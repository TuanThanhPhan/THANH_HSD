import sys
import os
from pathlib import Path
import torch
import pickle
import numpy as np
from flask import Flask, render_template, request
from transformers import AutoTokenizer

# 1. Thiết lập đường dẫn gốc để import được config, models, utils
# Giả sử app.py nằm trong DEANTN_HSD/web demo/
current_dir = Path(__file__).resolve().parent
root_path = str(current_dir.parent) 
if root_path not in sys.path:
    sys.path.append(root_path)

# 2. Import module sau khi đã set path
try:
    import config
    from utils.cleantext import clean_text_pipeline
    from utils.dataloader import ViHSDDataset
    from models.model import HybridHateSpeechModel
except ImportError as e:
    print(f"❌ Lỗi Import: {e}. Hãy đảm bảo bạn đang ở đúng thư mục repo.")
    raise

# ===== INIT APP =====
# Chỉ định rõ thư mục template để tránh lỗi không tìm thấy file index.html
app = Flask(__name__, template_folder=str(current_dir / "templates"))

# ===== CONFIG =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "vinai/phobert-base"

LABEL_MAP = {0: "Bình thường", 1: "Gây hấn", 2: "Tiêu cực"}
COLORS = {0: "success", 1: "warning", 2: "danger"}

# ===== LOAD ASSETS =====
print("🔄 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load Char Vocab từ Drive dựa trên config.py
vocab_path = os.path.join(config.SAVE_DIR, config.CHAR_VOCAB_FILE)
print(f"🔄 Loading char vocab from: {vocab_path}")
with open(vocab_path, "rb") as f:
    char_to_idx = pickle.load(f)

# ===== LOAD MODEL =====
print("🔄 Loading model...")
# Model Hybrid dùng PhoBERT và CharCNN
model = HybridHateSpeechModel(MODEL_NAME, len(char_to_idx) + 2)

model_path = os.path.join(config.SAVE_DIR, "hybrid_best.pt") # File bạn để trên Drive
checkpoint = torch.load(model_path, map_location=DEVICE)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully!")

# ===== ROUTE =====
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        raw_text = request.form.get("content")
        if raw_text:
            cleaned_text = clean_text_pipeline(raw_text)
            
            # Sử dụng Dataset pipeline để chuẩn hóa đầu vào
            temp_dataset = ViHSDDataset(
                texts=[cleaned_text],
                labels=[0],
                tokenizer=tokenizer,
                max_len=config.MAX_LEN,
                char_to_idx=char_to_idx
            )
            data_item = temp_dataset[0]

            input_ids = data_item["input_ids"].unsqueeze(0).to(DEVICE)
            mask = data_item["attention_mask"].unsqueeze(0).to(DEVICE)
            char_tensor = data_item["char_input"].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(input_ids, mask, char_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)

            result = {
                "text": raw_text,
                "label": LABEL_MAP[pred_idx],
                "conf": round(probs[pred_idx] * 100, 2),
                "color": COLORS[pred_idx]
            }
    return render_template("index.html", result=result)

if __name__ == "__main__":
    # Để chạy trên Colab qua tunnel, host phải là 0.0.0.0
    app.run(host="0.0.0.0", port=5000)