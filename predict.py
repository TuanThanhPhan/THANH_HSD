import pickle
import torch
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import os

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

import config
from utils.dataloader import ViHSDDataset
from models.model import HybridHateSpeechModel
from models.phobert_model import PhoBERTModel
from models.visobert_model import ViSoBERTModel

def plot_confusion_matrix(y_true, y_pred, model_type, split):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Bình thường", "Gây hấn", "Tiêu cực"],
                yticklabels=["Bình thường", "Gây hấn", "Tiêu cực"])
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title(f'Confusion Matrix - {model_type.upper()} ({split.upper()})')
    plt.tight_layout()
    save_path = os.path.join(config.SAVE_DIR, f"{model_type}_{split}_confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Đã lưu Confusion Matrix tại: {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="hybrid", choices=["phobert", "visobert", "hybrid"])
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    args = parser.parse_args()

    if args.model_name is None:
        if args.model_type in ["phobert", "hybrid"]:
            args.model_name = "vinai/phobert-base"
        elif args.model_type == "visobert":
            args.model_name = "uitnlp/visobert" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ===== LOAD DỮ LIỆU =====
    print(f"--- Loading {args.split} dataset ---")
    if args.split == "dev":
        df = pd.read_csv(config.DEV_PATH)
    else:
        df = pd.read_csv(config.TEST_PATH)

    texts = df["free_text"].astype(str).values
    labels = df["label_id"].values

    vocab_path = os.path.join(config.SAVE_DIR, config.CHAR_VOCAB_FILE)
    with open(vocab_path, "rb") as f:
        char_to_idx = pickle.load(f)

    dataset = ViHSDDataset(texts, labels, tokenizer, config.MAX_LEN, char_to_idx)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE)

    # ===== KHỞI TẠO MODEL =====
    if args.model_type == "phobert":
        model = PhoBERTModel(args.model_name)
    elif args.model_type == "visobert":
        model = ViSoBERTModel(args.model_name)
    elif args.model_type == "hybrid":
        model = HybridHateSpeechModel(args.model_name, len(char_to_idx) + 2)

    model_path = os.path.join(config.SAVE_DIR, f"{args.model_type}_best.pt")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ===== DỰ ĐOÁN =====
    preds = []
    print(f"Predicting on {device}...")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            if args.model_type == "hybrid":
                char_in = batch["char_input"].to(device)
                logits = model(input_ids, mask, char_in)
            else:
                logits = model(input_ids, mask)

            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

    # ===== BÁO CÁO KẾT QUẢ =====
    print("\n--- Classification Report ---")
    # Sử dụng labels gốc để so sánh
    print(classification_report(labels, preds, target_names=["Bình thường", "Gây hấn", "Tiêu cực"], digits=4))

    plot_confusion_matrix(labels, preds, args.model_type, args.split)

if __name__ == "__main__":
    main()