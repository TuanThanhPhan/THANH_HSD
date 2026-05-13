import pandas as pd
import numpy as np
import os
import random
from cleantext import clean_text_pipeline

# ====================== CẤU HÌNH ======================
BASE_PATH = r'D:\DE AN TOT NGHIEP\HSD_DEAN\data'
vihsd_path = os.path.join(BASE_PATH, "ViHSD")
error_file_path = os.path.join(BASE_PATH, "error_analysis.xlsx")
OUTPUT_FILE = os.path.join(BASE_PATH, "adapt_train.csv")

RANDOM_STATE = 42

# ==================== TARGET MỚI ====================
TARGET_DISTRIBUTION = {
    0: 5300,   # Bình thường
    1: 5300,   # Gây hấn
    2: 4700    # Tiêu cực 
}

ERROR_MULTIPLIER = 2.5     

# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(RANDOM_STATE)

# 1. LOAD DATA
df_vihsd = pd.read_csv(os.path.join(vihsd_path, "train.csv"))
df_error = pd.read_excel(error_file_path)

print(f"ViHSD train gốc     : {len(df_vihsd):,} mẫu")
print(f"Error samples       : {len(df_error):,} mẫu")

# 2. CLEAN TEXT
for name, df in [("ViHSD", df_vihsd), ("Error", df_error)]:
    df['free_text'] = df['free_text'].astype(str).apply(clean_text_pipeline)
    print(f"✓ Cleaned {name}")

df_vihsd = df_vihsd[df_vihsd['free_text'].str.strip() != ""].drop_duplicates(subset=['free_text', 'label_id']).reset_index(drop=True)
df_error = df_error[df_error['free_text'].str.strip() != ""].drop_duplicates(subset=['free_text', 'label_id']).reset_index(drop=True)

# 3. SAMPLE ViHSD
vihsd_sampled = pd.DataFrame()
for label, n in TARGET_DISTRIBUTION.items():
    subset = df_vihsd[df_vihsd['label_id'] == label]
    sampled = subset.sample(n=min(n, len(subset)), random_state=RANDOM_STATE)
    vihsd_sampled = pd.concat([vihsd_sampled, sampled], ignore_index=True)

print("Phân bố ViHSD sampled:")
print(vihsd_sampled['label_id'].value_counts().sort_index())

# 4. AUGMENT ERROR
def basic_augment(text):
    if not isinstance(text, str) or len(text) < 10:
        return text
    words = text.split()
    aug = words.copy()
    
    if random.random() < 0.5:
        idx = random.randint(0, len(aug)-1)
        aug[idx] += random.choice([" lắm", " quá", " vl", " thật", " luôn"])
    
    if random.random() < 0.4 and len(aug) >= 4:
        i, j = random.sample(range(len(aug)), 2)
        aug[i], aug[j] = aug[j], aug[i]
    
    return ' '.join(aug)

aug_rows = []
for _, row in df_error.iterrows():
    text = row['free_text']
    label = row['label_id']
    aug_rows.append({'free_text': text, 'label_id': label})
    
    # Tiêu cực augment mạnh hơn
    num_aug = 2 if label == 2 else 1
    for _ in range(num_aug):
        aug_rows.append({'free_text': basic_augment(text), 'label_id': label})

df_error_aug = pd.DataFrame(aug_rows)
print(f"Error sau augment: {len(df_error_aug):,} mẫu")

# 5. MERGE
adapt_train = pd.concat([vihsd_sampled, df_error_aug], ignore_index=True)

final_train = pd.DataFrame()
for label, n in TARGET_DISTRIBUTION.items():
    subset = adapt_train[adapt_train['label_id'] == label]
    if len(subset) > n:
        subset = subset.sample(n=n, random_state=RANDOM_STATE)
    final_train = pd.concat([final_train, subset])

final_train = final_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# STATISTICS
print("\n" + "="*60)
print("FINAL DISTRIBUTION")
print("="*60)
print(final_train['label_id'].value_counts().sort_index())
print(f"Tổng mẫu: {len(final_train):,}")

final_train.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n✅ ĐÃ LƯU: {OUTPUT_FILE}")