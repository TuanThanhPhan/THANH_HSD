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

    last_ckpt = os.path.join(config.SAVE_DIR, f"{args.model_type}_last_extend_ep50.pt")
    best_ckpt = os.path.join(config.SAVE_DIR, f"{args.model_type}_best_extend_ep50.pt")

    # Đường dẫn nạp model Baseline đã luyện ở GĐ1
    baseline_ckpt = os.path.join(config.SAVE_DIR, f"{args.model_type}_best_ep50.pt")

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

    # Nạp trọng số từ Model Baseline (GĐ1) trước khi bắt đầu train
    if not args.resume:
        if os.path.exists(baseline_ckpt):
            print(f"--- NẠP TRỌNG SỐ BASELINE: {baseline_ckpt} ---")
            checkpoint_base = torch.load(baseline_ckpt, map_location=device)
            # Kiểm tra xem file lưu là dict hay chỉ là weight
            if isinstance(checkpoint_base, dict) and "model_state_dict" in checkpoint_base:
                model.load_state_dict(checkpoint_base["model_state_dict"])
            else:
                model.load_state_dict(checkpoint_base)
        else:
            print("--- CẢNH BÁO: Không tìm thấy Baseline, sẽ train mới hoàn toàn ---")

   # ===== WEIGHTS & OPTIMIZER =====
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # ===== Loss function =====
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.15)

    # Thiết lập Optimizer với LR khác nhau cho BERT và các lớp tùy chỉnh
    if args.model_type == "hybrid":
        phobert_params = list(model.phobert.parameters())
        custom_params = [p for n, p in model.named_parameters() if "phobert." not in n]
        optimizer = optim.AdamW([
            {'params': phobert_params, 'lr': 5e-6}, 
            {'params': custom_params, 'lr': 1e-4} 
        ], weight_decay=0.05)
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
    
    # =========== RESUME ============
    start_epoch = 0
    best_f1 = 0
    patience = 0 

    if args.resume and os.path.exists(last_ckpt):
        print(f"--- Đang Resume từ checkpoint: {last_ckpt} ---")
        checkpoint = torch.load(last_ckpt, map_location=device)
        
        # 1. Nạp trọng số Model và Optimizer
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # 2. Nạp trạng thái Scheduler
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("-> Đã nạp lại trạng thái Scheduler từ file.")
        
        # 3. Nạp các biến trạng thái quan trọng
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_f1"]
        patience = checkpoint.get("patience", 0)
        
        print(f"-> Resume thành công: Epoch tiếp theo {start_epoch+1}, Best F1 hiện tại: {best_f1:.4f}, Patience: {patience}")
    # ============================================================================

    # ===== KHỞI TẠO TRAINER =====
    trainer = Trainer(model, optimizer, criterion, device, scheduler, args.model_type)

    # ===== TRAINING LOOP =====
    for epoch in range(start_epoch, config.EPOCHS):
        train_loss = trainer.train_epoch(train_loader)
        labels_all, preds, val_loss = trainer.eval_epoch(dev_loader)

        dev_f1 = f1_score(labels_all, preds, average="macro")
        report = classification_report(labels_all, preds, target_names=["Bình thường", "Gây hấn", "Tiêu cực"], output_dict=True, zero_division=0)
        
        lr_phobert = optimizer.param_groups[0]['lr']
        lr_custom = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else None

        # Hiển thị Summary 
        print("\n" + "="*60)
        print(f"EPOCH {epoch+1}/{config.EPOCHS} SUMMARY")
        print("-" * 60)
        print(f"• Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        print(f"• Dev F1: {dev_f1:.4f}")
        print(f"• Dev F1 (Lớp): BT: {report['Bình thường']['f1-score']:.4f} | GH: {report['Gây hấn']['f1-score']:.4f} | TC: {report['Tiêu cực']['f1-score']:.4f}")
        
        if lr_custom:
            print(f"• L.R. PhoBERT: {lr_phobert:.2e} | L.R. Custom: {lr_custom:.2e}")
        else:
            print(f"• Learning Rate: {lr_phobert:.2e}")
        print("="*60)

        # In Confusion Matrix dạng bảng
        cm = confusion_matrix(labels_all, preds)
        target_names = ["Bình thường", "Gây hấn", "Tiêu cực"]
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        print("-" * 60)
        print("[CONFUSION MATRIX]")
        print(cm_df)
        print("-" * 60)

        # Vẽ và lưu ảnh CM
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title(f'CM - {args.model_type.upper()} - Ep {epoch+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(cm_folder, f"epoch_{epoch+1}.png"))
        plt.close()

        # Cập nhật Best và Patience
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience = 0 # Reset patience khi đạt đỉnh mới
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                "best_f1": best_f1, "patience": patience
            }, best_ckpt)
            print("--> Dev F1 improved. Saved best model.")
        else:
            patience += 1 # Tăng patience nếu không cải thiện
            print(f"--> Patience: {patience}/{config.PATIENCE}")

        # LUÔN LƯU last_ckpt để resume nếu bị ngắt
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
            "best_f1": best_f1, "patience": patience
        }, last_ckpt)

        if patience >= config.PATIENCE:
            print("Early stopping triggered!")
            break

if __name__ == "__main__":
    main()