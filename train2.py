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
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_linear_schedule_with_warmup

import config
from seed import set_seed

from utils.dataloader import ViHSDDataset
from utils.char_vocab import build_char_vocab

from models.model import HybridHateSpeechModel
from models.phobert_model import PhoBERTModel
from models.visobert_model import ViSoBERTModel
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GĐ3: Model GĐ1 + adapt_train trên dev ViHSD")

    parser.add_argument("--model_type", type=str, default="hybrid", choices=["phobert", "visobert", "hybrid"])
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--resume", action="store_true", help="Resume từ checkpoint GĐ3")

    # Path dữ liệu — train trong SAVE_DIR, dev ViHSD gốc
    parser.add_argument(
        "--train_data",
        type=str,
        default=os.path.join(config.SAVE_DIR, "adapt_train.csv"),  
        help="Tập train GĐ3 (từ build_newtrain.py)"
    )
    parser.add_argument(
        "--dev_data",
        type=str,
        default=config.DEV_PATH,
        help="Tập dev ViHSD gốc để chọn best model"
    )

    # Path baseline GĐ1
    parser.add_argument(
        "--baseline_ckpt",
        type=str,
        default="hybrid_best_ep50.pt",
        help="Checkpoint model GĐ1 để nạp trước khi fine-tune"
    )

    # Hyperparams GĐ3 — fine-tune nhẹ, LR thấp hơn GĐ1
    parser.add_argument("--lr_phobert", type=float, default=5e-6, help="LR PhoBERT (GĐ1 dùng 1e-5)")
    parser.add_argument("--lr_custom", type=float, default=1e-5, help="LR custom head")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--l2_reg", type=float, default=1e-4, help="L2 reg về weight GĐ1 (chống forgetting)")
    parser.add_argument("--freeze_layers", type=int, default=8, help="Đóng băng N layers đầu PhoBERT (0=không freeze)")

    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.SAVE_DIR, exist_ok=True)

    # Tên checkpoint
    cm_folder = os.path.join(config.CM_DIR, args.model_type + "_extend")
    os.makedirs(cm_folder, exist_ok=True)
    last_ckpt = os.path.join(config.SAVE_DIR, f"{args.model_type}_extend_last.pt")
    best_ckpt = os.path.join(config.SAVE_DIR, f"{args.model_type}_extend_best.pt")
    baseline_ckpt = os.path.join(config.SAVE_DIR, args.baseline_ckpt)

    print("="*60)
    print("FINE-TUNE TRÊN ADAPT_TRAIN + ĐÁNH GIÁ DEV ViHSD")
    print("="*60)
    print(f"Model type : {args.model_type}")
    print(f"Baseline   : {baseline_ckpt}")
    print(f"Last ckpt  : {last_ckpt}")
    print(f"Best ckpt  : {best_ckpt}")
    print(f"LR PhoBERT : {args.lr_phobert}")
    print(f"LR Custom  : {args.lr_custom}")
    print(f"L2 reg     : {args.l2_reg}")
    print(f"Freeze     : {args.freeze_layers} layers")
    print("="*60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ==================== LOAD DATA ====================
    print(f"\n--- Loading GĐ3 train: {args.train_data} ---")
    train_df = pd.read_csv(args.train_data)
    train_df['free_text'] = train_df['free_text'].astype(str)

    print(f"--- Loading ViHSD dev: {args.dev_data} ---")
    dev_df = pd.read_csv(args.dev_data)
    dev_df['free_text'] = dev_df['free_text'].astype(str)

    train_texts = train_df["free_text"].values
    train_labels = train_df["label_id"].astype(int).values
    dev_texts = dev_df["free_text"].astype(str).values
    dev_labels = dev_df["label_id"].astype(int).values

    print(f"Train: {len(train_texts)} | Dev: {len(dev_texts)}")
    print("Train distribution:", dict(pd.Series(train_labels).value_counts().sort_index()))
    print("Dev distribution:", dict(pd.Series(dev_labels).value_counts().sort_index()))

    # ==================== CHAR VOCAB ====================
    vocab_path = os.path.join(config.SAVE_DIR, config.CHAR_VOCAB_FILE)
    if os.path.exists(vocab_path):
        with open(vocab_path, "rb") as f:
            char_to_idx = pickle.load(f)
    else:
        char_to_idx = build_char_vocab(train_texts)
        with open(vocab_path, "wb") as f:
            pickle.dump(char_to_idx, f)

    # ==================== DATALOADER ====================
    train_dataset = ViHSDDataset(train_texts, train_labels, tokenizer, config.MAX_LEN, char_to_idx)
    dev_dataset = ViHSDDataset(dev_texts, dev_labels, tokenizer, config.MAX_LEN, char_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=2)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, num_workers=2)

    # ==================== MODEL ====================
    if args.model_type == "hybrid":
        model = HybridHateSpeechModel(args.model_name, len(char_to_idx) + 2)
    elif args.model_type == "phobert":
        model = PhoBERTModel(args.model_name)
    else:
        model = ViSoBERTModel(args.model_name)
    model.to(device)

    # ==================== NẠP GĐ1 + FREEZE ====================
    baseline_weights = {}
    if not args.resume:
        if os.path.exists(baseline_ckpt):
            print(f"\n--- Nạp trọng số GĐ1: {baseline_ckpt} ---")
            ckpt = torch.load(baseline_ckpt, map_location=device)
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state)
            print("-> Đã nạp xong.")

            # Lưu baseline weights cho L2 reg
            if args.l2_reg > 0:
                for name, param in state.items():
                    baseline_weights[name] = param.clone().to(device)
                print(f"-> Đã lưu baseline weights cho L2 reg (λ={args.l2_reg})")
        else:
            print("[WARNING] Không tìm thấy baseline GĐ1!")

    # Đóng băng bottom layers
    if args.freeze_layers > 0 and hasattr(model, 'phobert'):
        print(f"\n--- Freezing {args.freeze_layers} bottom layers ---")
        frozen = 0
        for name, param in model.phobert.named_parameters():
            if "encoder.layer." in name:
                layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_idx < args.freeze_layers:
                    param.requires_grad = False
                    frozen += 1
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"-> Frozen {frozen} param groups. Trainable: {trainable:,}/{total:,} ({trainable/total*100:.1f}%)")

    # ==================== LOSS + CLASS WEIGHT ====================
    unique_labels = np.unique(train_labels)
    class_weights_auto = compute_class_weight(class_weight='balanced', classes=unique_labels, y=train_labels)
    class_weights = torch.tensor(class_weights_auto, dtype=torch.float32).to(device)
    print(f"\nClass weights (auto-balanced): {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    # ==================== OPTIMIZER GĐ3 ====================
    if args.model_type == "hybrid":
        phobert_params = [p for n, p in model.named_parameters() if "phobert." in n and p.requires_grad]
        custom_params = [p for n, p in model.named_parameters() if "phobert." not in n and p.requires_grad]
        optimizer = optim.AdamW([
            {'params': phobert_params, 'lr': args.lr_phobert},
            {'params': custom_params, 'lr': args.lr_custom}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr_phobert, weight_decay=args.weight_decay)

    # ==================== SCHEDULER ====================
    num_training_steps = len(train_loader) * config.EPOCHS
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # ==================== RESUME ====================
    start_epoch, best_f1, patience = 0, -1, 0
    if args.resume and os.path.exists(last_ckpt):
        print(f"\n--- Resume từ: {last_ckpt} ---")
        checkpoint = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_f1"]
        patience = checkpoint.get("patience", 0)
        print(f"-> Epoch tiếp: {start_epoch+1}, Best F1: {best_f1:.4f}, Patience: {patience}")

    # ==================== TRAINER ====================
    trainer = Trainer(model, optimizer, criterion, device, scheduler, args.model_type)

    # ==================== TRAINING LOOP GĐ3 ====================
    print(f"\n{'='*60}")
    print(f"BẮT ĐẦU TRAIN GĐ3 | Epochs: {config.EPOCHS} | Batch: {config.BATCH_SIZE} | Patience: {config.PATIENCE}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, config.EPOCHS):
        # Train
        train_loss = trainer.train_epoch(train_loader)

        # Thêm L2 reg loss (chỉ log, không backward riêng)
        l2_loss_val = 0.0
        if args.l2_reg > 0 and baseline_weights:
            l2_loss_val = 0.0
            for name, param in model.named_parameters():
                if name in baseline_weights and param.requires_grad:
                    l2_loss_val += torch.sum((param - baseline_weights[name]) ** 2)
            l2_loss_val = args.l2_reg * l2_loss_val

        # Eval trên dev ViHSD
        labels_all, preds, val_loss = trainer.eval_epoch(dev_loader)
        dev_f1 = f1_score(labels_all, preds, average="macro")
        report = classification_report(labels_all, preds, target_names=["Bình thường", "Gây hấn", "Tiêu cực"], output_dict=True, zero_division=0)

        lr_phobert = optimizer.param_groups[0]['lr']
        lr_custom = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else None

        # Log
        print("\n" + "="*60)
        print(f"EPOCH {epoch+1}/{config.EPOCHS} | GĐ3")
        print("-" * 60)
        print(f"• Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        if args.l2_reg > 0:
            print(f"• L2 reg: {l2_loss_val.item():.6f}")
        print(f"• Dev Macro F1: {dev_f1:.4f}")
        print(f"• Per-class F1: BT={report['Bình thường']['f1-score']:.4f} | GH={report['Gây hấn']['f1-score']:.4f} | TC={report['Tiêu cực']['f1-score']:.4f}")
        print(f"• Per-class Rec: BT={report['Bình thường']['recall']:.4f} | GH={report['Gây hấn']['recall']:.4f} | TC={report['Tiêu cực']['recall']:.4f}")
        print(f"• LR: PhoBERT={lr_phobert:.2e} | Custom={lr_custom:.2e}" if lr_custom else f"• LR: {lr_phobert:.2e}")
        print("="*60)

        # Confusion Matrix
        cm = confusion_matrix(labels_all, preds)
        target_names = ["Bình thường", "Gây hấn", "Tiêu cực"]
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        print("\n[CONFUSION MATRIX - ViHSD Dev]")
        print(cm_df)

        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title(f'CM GĐ3 - {args.model_type.upper()} - Ep {epoch+1}')
        plt.tight_layout()
        plt.savefig(os.path.join(cm_folder, f"epoch_{epoch+1}.png"))
        plt.close()

        # Save best (dựa trên dev ViHSD)
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1": best_f1, "patience": patience,
                "args": vars(args)
            }, best_ckpt)
            print(f"\n--> [BEST] Dev F1 improved to {best_f1:.4f}. Saved: {best_ckpt}")
        else:
            patience += 1
            print(f"\n--> Patience: {patience}/{config.PATIENCE}")

        # Save last
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_f1": best_f1, "patience": patience
        }, last_ckpt)

        if patience >= config.PATIENCE:
            print("\n[!] Early stopping!")
            break

    print(f"\n{'='*60}")
    print(f"GĐ3 HOÀN TẤT | Best Dev F1: {best_f1:.4f}")
    print(f"Best model: {best_ckpt}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
