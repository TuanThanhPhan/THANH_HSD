import pickle
import torch
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import os

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import config
from utils.dataloader import ViHSDDataset
from models.model import HybridHateSpeechModel

def plot_confusion_matrix(y_true, y_pred, model_type, data_name):
    """Vẽ và lưu ma trận nhầm lẫn dưới dạng ảnh"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=["Bình thường", "Gây hấn", "Tiêu cực"],
                yticklabels=["Bình thường", "Gây hấn", "Tiêu cực"])
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title(f'Confusion Matrix - {model_type.upper()} ({data_name})')
    plt.tight_layout()
    
    save_path = os.path.join(config.SAVE_DIR, f"{model_type}_{data_name}_confusion_matrix.png")
    plt.savefig(save_path)
    plt.show() 

def main():
    parser = argparse.ArgumentParser(description="Inference and Error Analysis on Real-world Data")
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base", help="Pretrained model base")
    parser.add_argument("--data_path", type=str, default="data/dataHSD.xlsx", help="Đường dẫn tới file dữ liệu thực tế")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ===== 1. LOAD DỮ LIỆU THỰC TẾ (EXCEL) =====
    print(f"--- Loading real-world dataset from {args.data_path} ---")
    try:
        df = pd.read_excel(args.data_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {args.data_path}. Vui lòng kiểm tra lại.")
        return

    # Lọc bỏ các dòng NaN
    df = df.dropna(subset=['free_text', 'label_id']).copy()
    
    texts = df["free_text"].astype(str).values
    labels = df["label_id"].astype(int).values

    # ===== 2. LOAD VOCAB & DATALOADER =====
    vocab_path = os.path.join(config.SAVE_DIR, config.CHAR_VOCAB_FILE)
    with open(vocab_path, "rb") as f:
        char_to_idx = pickle.load(f)

    dataset = ViHSDDataset(texts, labels, tokenizer, config.MAX_LEN, char_to_idx)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # ===== 3. KHỞI TẠO & LOAD MODEL HYBRID =====
    print("--- Khởi tạo và nạp trọng số mô hình Hybrid ---")
    model = HybridHateSpeechModel(args.model_name, len(char_to_idx) + 2)
    
    model_path = os.path.join(config.SAVE_DIR, "hybrid_best_ep50.pt")
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy trọng số tại {model_path}")
        return
        
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # ===== 4. DỰ ĐOÁN =====
    preds = []
    print(f"Predicting on {device}...")
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            char_in = batch["char_input"].to(device)
            
            logits = model(input_ids, mask, char_in)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

    # ===== 5. BÁO CÁO KẾT QUẢ TRÊN CONSOLE =====
    print("\n--- Classification Report ---")
    target_names = ["Bình thường", "Gây hấn", "Tiêu cực"]
    print(classification_report(labels, preds, target_names=target_names, digits=4))

    # Tính toán F1 Score
    macro_f1 = f1_score(labels, preds, average='macro')
    class_f1 = f1_score(labels, preds, average=None)
    f1_0, f1_1, f1_2 = class_f1[0], class_f1[1], class_f1[2]

    # In ra bức tranh toàn cảnh (Overview)
    print("\n" + "=" * 60)
    print("BỨC TRANH TOÀN CẢNH (OVERVIEW) - DỮ LIỆU THỰC TẾ")
    print("=" * 60)
    print(f"• Macro F1: {macro_f1:.4f}")
    print(f"• F1 (Lớp): Bình thường: {f1_0:.4f} | Gây hấn: {f1_1:.4f} | Tiêu cực: {f1_2:.4f}")
    
    # In ma trận nhầm lẫn dạng số ở cuối cùng
    cm = confusion_matrix(labels, preds)
    cm_df = pd.DataFrame(
        cm, 
        index=target_names, 
        columns=target_names
    )
    print("-" * 60)
    print("[CONFUSION MATRIX]")
    print(cm_df)
    print("-" * 60)

    # Vẽ và lưu ảnh ma trận nhầm lẫn
    plot_confusion_matrix(labels, preds, "hybrid", "dataHSD")

    # ===== 6. XUẤT FILE PHÂN TÍCH LỖI =====
    print("\n--- Đang trích xuất các mẫu dự đoán sai ---")
    df['predicted_label_id'] = preds
    df_errors = df[df['label_id'] != df['predicted_label_id']].copy()
    
    if len(df_errors) > 0:
        label_mapping = {0: "Bình thường", 1: "Gây hấn", 2: "Tiêu cực"}
        df_errors['Nhãn_Thực_Tế'] = df_errors['label_id'].map(label_mapping)
        df_errors['Nhãn_Dự_Đoán'] = df_errors['predicted_label_id'].map(label_mapping)
        
        cols = ['free_text', 'label_id', 'predicted_label_id', 'Nhãn_Thực_Tế', 'Nhãn_Dự_Đoán']
        other_cols = [c for c in df_errors.columns if c not in cols]
        df_errors = df_errors[cols + other_cols]

        error_save_path = os.path.join(config.SAVE_DIR, "hybrid_error_analysis_dataHSD.xlsx")
        df_errors.to_excel(error_save_path, index=False)
        print(f"[*] Đã xuất thành công {len(df_errors)} mẫu lỗi ra file: {error_save_path}")
    else:
        print("[*] Tuyệt vời! Không có mẫu nào bị dự đoán sai.")

if __name__ == "__main__":
    main()