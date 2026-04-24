import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight

import config
from seed import set_seed

from utils.dataloader import ViHSDDataset
from utils.char_vocab import build_char_vocab

from models.model import HybridHateSpeechModel
from models.phobert_model import PhoBERTModel
from models.visobert_model import ViSoBERTModel
from trainer import Trainer


def main():
    # ===== Định nghĩa tham số dòng lệnh =====
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_type",
        type=str,
        default="hybrid",
        choices=["phobert", "visobert", "hybrid"]
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="vinai/phobert-base"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume training from checkpoint"
    )

    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # Tạo thư mục lưu Confusion Matrix cho model tương ứng
    cm_folder = os.path.join(config.CM_DIR, args.model_type)
    os.makedirs(cm_folder, exist_ok=True)

    last_ckpt = os.path.join(config.SAVE_DIR, f"{args.model_type}_last.pt")
    best_ckpt = os.path.join(config.SAVE_DIR, f"{args.model_type}_best.pt")

    print("Model type:", args.model_type)
    print("Checkpoint:", last_ckpt)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ===== LOAD DỮ LIỆU ĐÃ LÀM SẠCH TỪ BƯỚC MERGE =====
    print("--- Loading pre-cleaned datasets ---")
    train_df = pd.read_csv(config.TRAIN_PATH) 
    dev_df = pd.read_csv(config.DEV_PATH)

    # CHỈ ÉP KIỂU, KHÔNG CHẠY LẠI PIPELINE
    train_texts = train_df["free_text"].astype(str).values
    train_labels = train_df["label_id"].values

    dev_texts = dev_df["free_text"].astype(str).values
    dev_labels = dev_df["label_id"].values

    # ===== BUILD/LOAD CHAR VOCAB =====
    vocab_path = os.path.join(config.SAVE_DIR, config.CHAR_VOCAB_FILE)
    if os.path.exists(vocab_path):
        print("Loading existing char vocab...")
        with open(vocab_path, "rb") as f:
            char_to_idx = pickle.load(f)
    else:
        print("Building new char vocab from cleaned data...")
        char_to_idx = build_char_vocab(train_texts)
        with open(vocab_path, "wb") as f:
            pickle.dump(char_to_idx, f)

    # ===== DATASET & DATALOADER =====
    train_dataset = ViHSDDataset(
        train_texts,
        train_labels,
        tokenizer,
        config.MAX_LEN,
        char_to_idx
    )

    dev_dataset = ViHSDDataset(
        dev_texts,
        dev_labels,
        tokenizer,
        config.MAX_LEN,
        char_to_idx
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, num_workers=2)

    # ===== KHỞI TẠO MODEL =====
    if args.model_type == "hybrid":
        model = HybridHateSpeechModel(
            args.model_name,
            len(char_to_idx) + 2 # +2 cho PAD và UNK
        )
    elif args.model_type == "phobert":
        model = PhoBERTModel(args.model_name)
    else:
        model = ViSoBERTModel(args.model_name)

    model.to(device)

   # ===== WEIGHTS & OPTIMIZER =====
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # ===== Loss function =====
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Thiết lập Optimizer với LR khác nhau cho BERT và các lớp tùy chỉnh
    if args.model_type == "hybrid":
        phobert_params = list(model.phobert.parameters())
        custom_params = [p for n, p in model.named_parameters() if "phobert." not in n]
        optimizer = optim.AdamW([
            {'params': phobert_params, 'lr': 1e-5}, 
            {'params': custom_params, 'lr': 3e-4} 
        ], weight_decay=0.01)
    else:
        # Đối với Baseline
        optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=0.01)

    # ===== Warmup Scheduler =====
    num_training_steps = len(train_loader) * config.EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps)

    # ===== RESUME TRAINING =====
    start_epoch = 0
    best_f1 = 0
    patience = 0 # Khởi tạo patience mặc định

    if args.resume and os.path.exists(last_ckpt):
        print("Loading checkpoint...")
        checkpoint = torch.load(last_ckpt, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # LOAD TRAINING STATE TRƯỚC ĐỂ LẤY START_EPOCH CHUẨN
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_f1"]
        patience = checkpoint.get("patience", 0)
        
        print(f"Resumed from epoch: {start_epoch}, Current Best F1: {best_f1:.4f}, Patience: {patience}")

        # XỬ LÝ SCHEDULER DỰA TRÊN START_EPOCH ĐÃ LOAD
        if "scheduler_state_dict" in checkpoint:
            remaining_epochs = config.EPOCHS - start_epoch
            if remaining_epochs > 0:
                remaining_steps = len(train_loader) * remaining_epochs
                num_warmup_steps = int(0.1 * remaining_steps)
                
                # Khởi tạo lại scheduler mới
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=remaining_steps
                )
                print(f"Scheduler re-initialized for remaining {remaining_epochs} epochs.")
            else:
                print("Training epochs completed.")
        else:
            print("No scheduler state, continuing...")
            
        print(f"Scheduler synchronized. Current LR: {optimizer.param_groups[0]['lr']:.2e}")

    # ===== KHỞI TẠO TRAINER =====
    trainer = Trainer(model, optimizer, criterion, device, scheduler, args.model_type)

    # ===== training loop =====
    for epoch in range(start_epoch, config.EPOCHS):

        train_loss = trainer.train_epoch(train_loader)

        # Lấy danh sách thực tế và dự đoán từ Trainer
        labels_all, preds, val_loss = trainer.eval_epoch(dev_loader)

        # Tính toán các chỉ số F1
        dev_f1 = f1_score(labels_all, preds, average="macro")
        report = classification_report(labels_all, preds, target_names=["Bình thường", "Gây hấn", "Tiêu cực"], output_dict=True, zero_division=0)
        
        f1_0 = report["Bình thường"]["f1-score"]
        f1_1 = report["Gây hấn"]["f1-score"]
        f1_2 = report["Tiêu cực"]["f1-score"]

        # Trích xuất Learning Rate hiện tại từ Optimizer
        lr_phobert = optimizer.param_groups[0]['lr']
        lr_custom = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else None

        # In ra bức tranh toàn cảnh (Overview)
        print("\n" + "="*60)
        print(f"EPOCH {epoch+1}/{config.EPOCHS} SUMMARY")
        print("-" * 60)
        print(f"• Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        print(f"• Dev F1: {dev_f1:.4f}")
        print(f"• Dev F1 (Lớp): Bình thường: {f1_0:.4f} | Gây hấn: {f1_1:.4f} | Tiêu cực: {f1_2:.4f}")
        
        if lr_custom:
            print(f"• L.R. PhoBERT: {lr_phobert:.2e} | L.R. Custom: {lr_custom:.2e}")
        else:
            print(f"• Learning Rate: {lr_phobert:.2e}")
        print("="*60)

        # In ma trận nhầm lẫn dạng số ở cuối cùng
        cm = confusion_matrix(labels_all, preds)
        target_names = ["Bình thường", "Gây hấn", "Tiêu cực"]
        cm_df = pd.DataFrame(
            cm, 
            index=target_names, 
            columns=target_names
        )
        print("-" * 60)
        print("[CONFUSION MATRIX]")
        print(cm_df)
        print("-" * 60)

        # Vẽ và lưu ảnh Confusion Matrix vào Drive
        cm = confusion_matrix(labels_all, preds)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=["Bình thường", "Gây hấn", "Tiêu cực"],
                    yticklabels=["Bình thường", "Gây hấn", "Tiêu cực"])
        plt.xlabel('Dự đoán (Predicted)')
        plt.ylabel('Thực tế (Actual)')
        plt.title(f'Confusion Matrix - {args.model_type.upper()} - Epoch {epoch+1}')
        plt.tight_layout()
        
        cm_path = os.path.join(cm_folder, f"epoch_{epoch+1}.png")
        plt.savefig(cm_path)
        plt.close() # Đóng figure để giải phóng RAM cho Epoch tiếp theo
        print(f"Đã lưu Confusion Matrix tại: {cm_path}")

        # Cập nhật logic patience trước khi đóng gói checkpoint
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience = 0
            print("--> Dev F1 improved. Saved best model.")
            
            # Chỉ lưu best_ckpt khi có cải thiện
            best_checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1": best_f1,
                "patience": patience
            }
            torch.save(best_checkpoint, best_ckpt)
        else:
            patience += 1
            print(f"--> Patience: {patience}/{config.PATIENCE}")

        # Tạo checkpoint mới với giá trị patience ĐÃ CẬP NHẬT
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_f1": best_f1,
            "patience": patience
        }

        # Lưu last_ckpt sau khi đã tính toán xong patience
        torch.save(checkpoint, last_ckpt)

        if patience >= config.PATIENCE:
            print("Early stopping")
            break
        
if __name__ == "__main__":
    main()