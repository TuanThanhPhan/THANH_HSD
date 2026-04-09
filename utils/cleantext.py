import re
import os
from pyvi import ViTokenizer

def load_teencode_dict(file_name="teencode_dict.txt"):
    teencode_dict = {}
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, file_name)
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                # Dùng dấu phẩy ngăn cách theo file chuẩn của bạn
                parts = line.split(",")
                if len(parts) == 2:
                    key, value = parts
                    # .strip() để tránh khoảng trắng thừa, .lower() để đồng bộ
                    teencode_dict[key.strip().lower()] = value.strip()
        print(f"--- Đã nạp thành công từ điển: {len(teencode_dict)} từ ---")
    except FileNotFoundError:
        print(f"Cảnh báo cực nguy hiểm: Không tìm thấy file {path}")
        print("Mô hình sẽ không thể xử lý teencode!")
    return teencode_dict

# Load dictionary 1 lần duy nhất khi import module
teencode_dict = load_teencode_dict()

def clean_text_pipeline(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Xóa URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Xóa Email
    text = re.sub(r'\S+@\S+', '', text)
    # Xóa Mention (@user)
    text = re.sub(r'@\w+', '', text)
    
    # Chuẩn hóa ký tự lặp: Giữ lại 2 ký tự (nguuuu -> nguu) giúp CharCNN nhận diện cường độ cảm xúc
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    # Replace teencode: Tách từ theo khoảng trắng để tra từ điển
    words = text.split()
    text = " ".join([teencode_dict.get(w, w) for w in words])
    
    # Giữ lại chữ cái, số, dấu câu và Emoji
    text = re.sub(r'[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s!?.,\U0001F300-\U0001FAFF]', ' ', text)
    
    # Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()
    return text