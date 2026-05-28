import pandas as pd
import numpy as np
import os
import random
import config 

# ====================== CẤU HÌNH ĐÃ TỐI ƯU ======================
error_file_path = os.path.join(config.SAVE_DIR, "error_analysis.xlsx")
VIHSD_TRAIN = config.TRAIN_PATH
OUTPUT_FILE = os.path.join(config.SAVE_DIR, "adapt_train.csv")

RANDOM_STATE = 42

ERROR_OVERSAMPLE_MAP = {
    0: 2,  
    1: 4,  
    2: 3   
}

NORMAL_REPLAY = 5500     
AGGRESSIVE_REPLAY = 4000 
HATE_REPLAY = 1200       

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(RANDOM_STATE)

# ==================== AUGMENT ĐỀU NHẸ ====================
def augment_text(text, label):
    """Augment nhẹ, đồng đều cho cả 3 nhãn"""
    if not isinstance(text, str) or len(text) < 5:
        return [text]

    results = [text]
    words = text.split()

    # 1. Thêm từ lóng/Emoji (tất cả nhãn)
    slangs = [" vl", " vcl", " vãi", " @@", " :))", " :v", " !", " !!!"]
    if random.random() < 0.4:
        results.append(text + random.choice(slangs))

    # 2. Lặp âm cuối
    if len(words) > 0:
        target_idx = random.randint(0, len(words) - 1)
        if len(words[target_idx]) > 2:
            new_words = words.copy()
            new_words[target_idx] = new_words[target_idx] + new_words[target_idx][-1] * 2
            results.append(" ".join(new_words))

    # 3. Hoán đổi từ
    if len(words) >= 4:
        new_words = words.copy()
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        results.append(" ".join(new_words))

    # 4. Typo nhẹ
    if random.random() < 0.2:
        results.append(text.replace("ch", "tr").replace("n", "nn"))

    # 5. Thêm từ nhấn mạnh (chỉ nhãn 1,2)
    if label in [1, 2] and random.random() < 0.3:
        intensifiers = [" quá", " cực", " thật", " vl"]
        results.append(text + random.choice(intensifiers))

    return list(set(results))

# ==================== MAIN ====================
# 1. Load mẫu sai từ SAVE_DIR
print(f"--- Loading error samples from: {error_file_path} ---")
df_error = pd.read_excel(error_file_path)
df_error['free_text'] = df_error['free_text'].astype(str)

print(f"Raw error: {len(df_error)}")
print("Error distribution:")
err_counts = df_error['label_id'].value_counts().sort_index()
for lbl in [0, 1, 2]:
    print(f"  Label {lbl}: {err_counts.get(lbl, 0)} ({err_counts.get(lbl,0)/len(df_error)*100:.1f}%)")

# 2. Áp dụng Chiến lược Oversample theo class định sẵn
print(f"\n--- Class-specific Oversampling based on model weaknesses ---")
aug_list = []
for _, row in df_error.iterrows():
    lbl = row['label_id']
    # Lấy hệ số nhân bản riêng biệt cho từng nhãn từ cấu hình nâng cấp
    oversample_times = ERROR_OVERSAMPLE_MAP.get(lbl, 2)
    for _ in range(oversample_times):
        variants = augment_text(row['free_text'], lbl)
        for v in variants:
            aug_list.append({'free_text': v, 'label_id': lbl})

df_error_aug = pd.DataFrame(aug_list).drop_duplicates(subset=['free_text'])
print(f"After Adaptive Oversampling: {len(df_error_aug)}")
print("Error distribution after augment:")
err_aug_counts = df_error_aug['label_id'].value_counts().sort_index()
for lbl in [0, 1, 2]:
    print(f"  Label {lbl}: {err_aug_counts.get(lbl, 0)} ({err_aug_counts.get(lbl,0)/len(df_error_aug)*100:.1f}%)")

# 3. Load ViHSD train từ config.TRAIN_PATH
print(f"\n--- Loading ViHSD train: {VIHSD_TRAIN} ---")
df_vihsd = pd.read_csv(VIHSD_TRAIN)
df_vihsd['free_text'] = df_vihsd['free_text'].astype(str)

# Lọc mẫu không hợp lệ
text_lower = df_vihsd['free_text'].str.lower().str.strip()
has_error = text_lower.str.contains('#error!', na=False, regex=False)
has_nan = text_lower.isin(['nan', 'nat', 'none', 'null', ''])
too_short = df_vihsd['free_text'].str.len() < 5
mask_valid = ~(has_error | has_nan | too_short)
df_vihsd = df_vihsd[mask_valid].copy()

print(f"ViHSD valid: {len(df_vihsd)}")

# 4. Random stratified sampling với tỉ lệ Replay tối ưu mới
print(f"\n--- Random sampling ViHSD with updated ratios ---")
print(f"Target: Nhãn 0={NORMAL_REPLAY}, Nhãn 1={AGGRESSIVE_REPLAY}, Nhãn 2={HATE_REPLAY}")

replay_list = []
for lbl, target_n in [(0, NORMAL_REPLAY), (1, AGGRESSIVE_REPLAY), (2, HATE_REPLAY)]:
    pool = df_vihsd[df_vihsd['label_id'] == lbl]
    available = len(pool)
    n_take = min(target_n, available)
    sampled = pool.sample(n=n_take, random_state=RANDOM_STATE)
    replay_list.append(sampled)
    print(f"  Label {lbl}: took {n_take}/{target_n} (available: {available})")

df_replay = pd.concat(replay_list)[['free_text', 'label_id']].copy()

# 5. Merge tập dữ liệu mới nhất
print(f"\n{'='*60}")
print("MERGE: Adaptive Error Oversampling + Optimized ViHSD Replay")
print(f"{'='*60}")

df_error_clean = df_error_aug[['free_text', 'label_id']].copy()
final_df = pd.concat([df_error_clean, df_replay]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print("\nFINAL UPDATED TRAIN STATISTICS:")
final_counts = final_df['label_id'].value_counts().sort_index()
total = len(final_df)
for lbl in [0, 1, 2]:
    cnt = final_counts.get(lbl, 0)
    pct = cnt/total*100
    print(f"  Label {lbl}: {cnt:>5} samples ({pct:>5.1f}%)")

print(f"\n  TOTAL NEW TRAIN DATA: {total} samples")
print(f"  Error:Replay Ratio = 1:{len(df_replay)/len(df_error_clean):.2f}")

print(f"\n  Tỷ lệ phân phối lý tưởng so với TestHSD (Target: 52/33/15%):")
for lbl, test_pct in [(0, 52), (1, 33), (2, 15)]:
    train_pct = final_counts.get(lbl, 0)/total*100
    diff = train_pct - test_pct
    print(f"    Label {lbl}: New Train {train_pct:.1f}% vs Test {test_pct}% (diff: {diff:+.1f}%)")

print(f"{'='*60}")

# Đảm bảo thư mục data/ tồn tại và lưu tập train mới nhất
os.makedirs("data", exist_ok=True)
final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n[SUCCESS] Saved updated dataset to: {OUTPUT_FILE}")