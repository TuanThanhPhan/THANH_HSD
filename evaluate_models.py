import pickle
import torch
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import config
from utils.dataloader import ViHSDDataset
from models.model import HybridHateSpeechModel

def plot_confusion_matrix(y_true, y_pred, model_name, save_dir):
    """Vẽ và lưu ma trận nhầm lẫn tự động theo số lượng class thực tế"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    
    num_classes = cm.shape[0]
    class_labels = [f"Nhãn {i}" for i in range(num_classes)]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels,
                yticklabels=class_labels)
    plt.xlabel('Dự đoán (Predicted)')
    plt.ylabel('Thực tế (Actual)')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"cm_{model_name}.png")
    plt.savefig(save_path)
    plt.close()

def run_evaluation(model_path, loader, device, char_vocab_len, args):
    """Hàm chạy dự đoán cho một model cụ thể"""
    model = HybridHateSpeechModel(args.model_name, char_vocab_len + 2)
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file trọng số: {model_path}")
        return None

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            char_in = batch["char_input"].to(device)
            
            logits = model(input_ids, mask, char_in)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    return all_preds

def main():
    parser = argparse.ArgumentParser(description="So sánh 2 model Hybrid trên tập TestHSD")
    parser.add_argument("--model_name", type=str, default="vinai/phobert-base", help="Base model")

    parser.add_argument("--data_path", type=str, default="data/TestHSD.xlsx", help="Đường dẫn file test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 1. Load Data
    print(f"--- Đang tải dữ liệu: {args.data_path} ---")
    try:
        if args.data_path.endswith('.xlsx'):
            df = pd.read_excel(args.data_path)
        else:
            df = pd.read_csv(args.data_path)
    except Exception as e:
        print(f"❌ Lỗi khi đọc file: {e}")
        return
    
    df = df.dropna(subset=['free_text', 'label_id']).copy()
    texts = df["free_text"].astype(str).values
    labels = df["label_id"].astype(int).values

    # 2. Load Vocab & Dataloader
    vocab_path = os.path.join(config.SAVE_DIR, config.CHAR_VOCAB_FILE)
    with open(vocab_path, "rb") as f:
        char_to_idx = pickle.load(f)

    dataset = ViHSDDataset(texts, labels, tokenizer, config.MAX_LEN, char_to_idx)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 3. Danh sách model cần test
    model_files = {
        "Base_Model": "hybrid_best_ep50.pt",
        "Extended_Model": "hybrid_best_extend_ep50.pt"
    }

    results_summary = []

    for name, filename in model_files.items():
        full_path = os.path.join(config.SAVE_DIR, filename)
        print(f"\n Đang đánh giá model: {name} ({filename})...")
        
        preds = run_evaluation(full_path, loader, device, len(char_to_idx), args)
        
        if preds is not None:
            # Lưu kết quả dự đoán vào dataframe chính
            df[f'pred_{name}'] = preds
            
            # Tính chỉ số Macro-F1
            macro_f1 = f1_score(labels, preds, average='macro')
            report = classification_report(labels, preds, digits=4)
            print(report)

            results_summary.append({"Model": name, "Macro_F1": macro_f1})
            
            # Vẽ CM
            plot_confusion_matrix(labels, preds, name, config.SAVE_DIR)

    # 4. Tạo file Excel so sánh chi tiết 2 mô hình
    if "pred_Base_Model" in df.columns and "pred_Extended_Model" in df.columns:
        # Thêm cột cờ báo hiệu sự khác biệt giữa 2 model
        df['is_diff'] = df['pred_Base_Model'] != df['pred_Extended_Model']
        
        # Sắp xếp lại thứ tự cột cho trực quan
        core_cols = ['free_text', 'label_id', 'pred_Base_Model', 'pred_Extended_Model', 'is_diff']
        other_cols = [c for c in df.columns if c not in core_cols]
        df_export = df[core_cols + other_cols]
        
        compare_path = os.path.join(config.SAVE_DIR, "compare_models_results.xlsx")
        df_export.to_excel(compare_path, index=False)

    # 5. In bảng tổng kết
    print("\n" + "="*40)
    print(" BẢNG SO SÁNH KẾT QUẢ MACRO-F1")
    print("="*40)
    summary_df = pd.DataFrame(results_summary)
    summary_df['Macro_F1'] = summary_df['Macro_F1'].map('{:.4f}'.format)
    print(summary_df.to_string(index=False, justify='center'))
    print("="*40)

if __name__ == "__main__":
    main()