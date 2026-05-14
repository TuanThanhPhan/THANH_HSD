import pandas as pd
import numpy as np
import os
import random
import re
from cleantext import clean_text_pipeline

# ====================== CẤU HÌNH ======================
BASE_PATH = r'D:\DE AN TOT NGHIEP\HSD_DEAN\data'
vihsd_path = os.path.join(BASE_PATH, "ViHSD")
error_file_path = os.path.join(BASE_PATH, "error_analysis.xlsx")
OUTPUT_FILE = os.path.join(BASE_PATH, "adapt_train.csv")

RANDOM_STATE = 42
TARGET_DISTRIBUTION = {
    0: 6000,   # Bình thường
    1: 5250,   # Gây hấn
    2: 3750    # Tiêu cực
}

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(RANDOM_STATE)

# ==================== CÁC HÀM TĂNG CƯỜNG DỮ LIỆU ====================
def augment_text(text, label):
    if not isinstance(text, str) or len(text) < 5:
        return [text]
    
    results = [text]
    words = text.split()
    
    # 1. Thêm từ lóng/Emoji vào cuối (Mô phỏng FB)
    slangs = [" vl", " vcl", " vãi", " @@", " :))", " :v", " !", " !!!"]
    if random.random() < 0.5:
        results.append(text + random.choice(slangs))

    # 2. Lặp âm cuối (Vocal Stretching - Rất phổ biến ở nhãn 1, 2)
    if len(words) > 0:
        target_idx = random.randint(0, len(words) - 1)
        if len(words[target_idx]) > 2:
            new_words = words.copy()
            new_words[target_idx] = new_words[target_idx] + new_words[target_idx][-1] * 3
            results.append(" ".join(new_words))

    # 3. Hoán đổi từ (Random Swap)
    if len(words) >= 4:
        new_words = words.copy()
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        results.append(" ".join(new_words))

    # 4. Typo nhẹ (Bỏ dấu hoặc viết sai)
    if random.random() < 0.3:
        results.append(text.replace("ch", "tr").replace("n", "nn"))

    return list(set(results)) # Xóa trùng

# ==================== MAIN PROCESSING ====================
# 1. Load data
df_vihsd = pd.read_csv(os.path.join(vihsd_path, "train.csv"))
df_error = pd.read_excel(error_file_path)

# 2. Clean sơ bộ
df_vihsd['free_text'] = df_vihsd['free_text'].astype(str).apply(clean_text_pipeline)
df_error['free_text'] = df_error['free_text'].astype(str).apply(clean_text_pipeline)

# 3. Thực hiện Augmentation trên Error Data
print("--- Đang tăng cường dữ liệu lỗi thực tế ---")
aug_list = []
for _, row in df_error.iterrows():
    augmented_variants = augment_text(row['free_text'], row['label_id'])
    for variant in augmented_variants:
        aug_list.append({'free_text': variant, 'label_id': row['label_id']})

df_error_aug = pd.DataFrame(aug_list).drop_duplicates(subset=['free_text'])
print(f"Số lượng Error Data sau khi tăng cường: {len(df_error_aug)}")

# 4. Merge theo tỷ lệ mục tiêu
final_train_list = []
for label, target_n in TARGET_DISTRIBUTION.items():
    # Lấy toàn bộ dữ liệu lỗi của nhãn này
    err_sub = df_error_aug[df_error_aug['label_id'] == label]
    
    # Tính số lượng cần lấy thêm từ ViHSD
    needed = target_n - len(err_sub)
    if needed > 0:
        vihsd_sub = df_vihsd[df_vihsd['label_id'] == label]
        vihsd_sampled = vihsd_sub.sample(n=min(needed, len(vihsd_sub)), random_state=RANDOM_STATE)
        combined = pd.concat([err_sub, vihsd_sampled])
    else:
        combined = err_sub # Nếu lỗi đã quá nhiều thì chỉ lấy lỗi
    
    final_train_list.append(combined)

final_df = pd.concat(final_train_list).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("\nTHỐNG KÊ TẬP TRAIN MỚI:")
print(final_df['label_id'].value_counts().sort_index())
final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"Đã lưu tại: {OUTPUT_FILE}")