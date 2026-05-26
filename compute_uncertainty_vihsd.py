import os
import pickle
import torch
import pandas as pd
import numpy as np
import argparse

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import config
from seed import set_seed
from utils.dataloader import ViHSDDataset
from utils.char_vocab import build_char_vocab

from models.model import HybridHateSpeechModel
from models.phobert_model import PhoBERTModel
from models.visobert_model import ViSoBERTModel


def main():
    parser = argparse.ArgumentParser(
        description="Tính uncertainty (entropy) cho toàn bộ ViHSD train bằng model GĐ1"
    )

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
        "--baseline_ckpt",
        type=str,
        default="hybird_best_ep50.pt",
        help="Tên file checkpoint model GĐ1 trong thư mục SAVE_DIR"
    )

    parser.add_argument(
        "--train_data",
        type=str,
        default="data/ViHSD/train.csv",
        help="Đường dẫn tập train ViHSD gốc"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/vihsd_train_uncertainty.csv",
        help="File CSV đầu ra chứa uncertainty"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ==================== 1. LOAD DATA ====================
    print(f"[INFO] Loading train data: {args.train_data}")
    df = pd.read_csv(args.train_data)
    df['free_text'] = df['free_text'].astype(str)

    texts = df["free_text"].values
    labels = df["label_id"].astype(int).values
    print(f"[INFO] Total samples: {len(df)}")

    # ==================== 2. TOKENIZER & CHAR VOCAB ====================
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    vocab_path = os.path.join(config.SAVE_DIR, config.CHAR_VOCAB_FILE)
    if os.path.exists(vocab_path):
        print(f"[INFO] Loading char vocab from: {vocab_path}")
        with open(vocab_path, "rb") as f:
            char_to_idx = pickle.load(f)
    else:
        print("[INFO] Building char vocab from train texts...")
        char_to_idx = build_char_vocab(texts)
        os.makedirs(config.SAVE_DIR, exist_ok=True)
        with open(vocab_path, "wb") as f:
            pickle.dump(char_to_idx, f)
        print(f"[INFO] Saved char vocab to: {vocab_path}")

    # ==================== 3. DATASET & DATALOADER ====================
    dataset = ViHSDDataset(
        texts,
        labels,
        tokenizer,
        config.MAX_LEN,
        char_to_idx
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,      
        num_workers=2,
        drop_last=False     
    )

    # ==================== 4. KHỞI TẠO MODEL ====================
    if args.model_type == "hybrid":
        model = HybridHateSpeechModel(
            args.model_name,
            len(char_to_idx) + 2  # +2 cho PAD và UNK
        )
    elif args.model_type == "phobert":
        model = PhoBERTModel(args.model_name)
    else:
        model = ViSoBERTModel(args.model_name)

    model.to(device)

    # ==================== 5. NẠP TRỌNG SỐ GĐ1 ====================
    baseline_path = os.path.join(config.SAVE_DIR, args.baseline_ckpt)
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint GĐ1: {baseline_path}\n"
            f"Vui lòng kiểm tra lại tên file hoặc đường dẫn."
        )

    print(f"[INFO] Loading baseline checkpoint: {baseline_path}")
    checkpoint = torch.load(baseline_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    print("[INFO] Loaded GĐ1 weights successfully.")
    model.eval()

    # ==================== 6. TÍNH UNCERTAINTY ====================
    all_entropy = []
    all_confidence = []
    all_pred = []
    all_probs = []

    print(f"[INFO] Computing uncertainty for {len(dataset)} samples...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Chuyển batch lên device, bỏ qua 'labels'
            batch_inputs = {
                k: v.to(device)
                for k, v in batch.items()
                if k != "labels"
            }

            logits = model(**batch_inputs)
            probs = torch.softmax(logits, dim=-1)

            # Entropy: H = -sum(p * log(p))
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

            # Confidence = max probability
            confidence, pred_labels = torch.max(probs, dim=-1)

            all_entropy.extend(entropy.cpu().numpy().tolist())
            all_confidence.extend(confidence.cpu().numpy().tolist())
            all_pred.extend(pred_labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
                processed = min((batch_idx + 1) * args.batch_size, len(dataset))
                print(f"  Progress: {processed}/{len(dataset)}")

    # ==================== 7. LƯU KẾT QUẢ ====================
    df_out = df.copy().reset_index(drop=True)
    df_out['uncertainty'] = all_entropy
    df_out['confidence'] = all_confidence
    df_out['predicted_label'] = all_pred

    # (Tùy chọn) Lưu thêm xác suất từng nhãn để phân tích sau
    probs_array = np.array(all_probs)
    df_out['prob_label_0'] = probs_array[:, 0]
    df_out['prob_label_1'] = probs_array[:, 1]
    df_out['prob_label_2'] = probs_array[:, 2]

    # Tạo thư mục nếu chưa có
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_out.to_csv(args.output, index=False, encoding='utf-8-sig')

    print(f"\n[SUCCESS] Saved uncertainty file: {args.output}")
    print(f"Columns: {list(df_out.columns)}")

    # ==================== 8. THỐNG KÊ NHANH ====================
    print("\n[UNCERTAINTY STATISTICS BY LABEL]")
    print("-" * 50)
    for lbl in sorted(df_out['label_id'].unique()):
        sub = df_out[df_out['label_id'] == lbl]
        print(
            f"Label {lbl}:  "
            f"mean_unc={sub['uncertainty'].mean():.4f}  "
            f"std={sub['uncertainty'].std():.4f}  "
            f"mean_conf={sub['confidence'].mean():.4f}  "
            f"n={len(sub)}"
        )

    # Top 10 samples có uncertainty cao nhất (hard negatives)
    print("\n[TOP 10 HARDEST SAMPLES (highest uncertainty)]")
    print("-" * 50)
    top_hard = df_out.nlargest(10, 'uncertainty')[['free_text', 'label_id', 'predicted_label', 'uncertainty', 'confidence']]
    for idx, row in top_hard.iterrows():
        text = row['free_text'][:60] + "..." if len(row['free_text']) > 60 else row['free_text']
        print(f"  unc={row['uncertainty']:.4f} | true={int(row['label_id'])} | pred={int(row['predicted_label'])} | {text}")


if __name__ == "__main__":
    main()
