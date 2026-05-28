import sys
import os
from pathlib import Path
import torch
import pickle
import numpy as np
import gradio as gr
from transformers import AutoTokenizer

# 1. Thiết lập đường dẫn gốc để import được config, models, utils
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

# ===== CONFIG =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "vinai/phobert-base"
LABEL_MAP = {0: "Bình thường", 1: "Gây hấn", 2: "Tiêu cực"}

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
model = HybridHateSpeechModel(MODEL_NAME, len(char_to_idx) + 2)
model_path = os.path.join(config.SAVE_DIR, "hybrid_best_extend.pt") 
checkpoint = torch.load(model_path, map_location=DEVICE)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()
print("✅ Model loaded successfully!")

# ===== PREDICTION FUNCTION =====
def predict_hate_speech(raw_text):
    if not raw_text or not raw_text.strip():
        return {"Vui lòng nhập văn bản": 1.0}

    # Tiền xử lý văn bản
    cleaned_text = clean_text_pipeline(raw_text)
    
    # Chuẩn hóa đầu vào
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

    # Dự đoán
    with torch.no_grad():
        logits = model(input_ids, mask, char_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Tạo dictionary chứa kết quả {Tên nhãn: Xác suất} để đưa vào gr.Label
    result_dict = {LABEL_MAP[i]: float(probs[i]) for i in range(len(LABEL_MAP))}
    return result_dict

# ===== GRADIO UI =====
# Sử dụng Blocks để custom giao diện giống HTML/Bootstrap cũ
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #0d6efd; font-weight: bold;">🛡️ DEMO HATE SPEECH DETECTION</h1>
            <p style="color: #6c757d; font-size: 1.1em;">Phát hiện phát ngôn tiêu cực trên mạng xã hội</p>
        </div>
        """
    )
    
    with gr.Row():
        # Cột bên trái: Nhập liệu
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                lines=6, 
                placeholder="Nhập nội dung vào đây...", 
                label="Nội dung cần phân tích:"
            )
            submit_btn = gr.Button("Kiểm tra ngay", variant="primary", size="lg")
            
        # Cột bên phải: Hiển thị kết quả & xác suất
        with gr.Column(scale=1):
            output_label = gr.Label(
                num_top_classes=3, 
                label="Kết quả dự đoán & Xác suất"
            )
            
    # Xử lý sự kiện click
    submit_btn.click(fn=predict_hate_speech, inputs=input_text, outputs=output_label)

if __name__ == "__main__":
    # share=True để tạo ra một public link (hữu ích khi bạn chạy trên Google Colab)
    demo.launch(server_name="0.0.0.0", server_port=5000, share=True)