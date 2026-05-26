import pandas as pd
import numpy as np
import os
import random
import config 

# ====================== CẤU HÌNH ======================
BASE_PATH = r'D:\DE AN TOT NGHIEP\HSD_DEAN\data'
error_file_path = os.path.join(BASE_PATH, "error_analysis.xlsx")

# File uncertainty từ compute_uncertainty_vihsd.py (lưu trong SAVE_DIR)
UNCERTAINTY_FILE = os.path.join(config.SAVE_DIR, "vihsd_train_uncertainty.csv")
OUTPUT_FILE = os.path.join(BASE_PATH, "adapt_train.csv")

RANDOM_STATE = 42
OLD_NEW_RATIO = 2.0   # Tỷ lệ mẫu cũ : mẫu error

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(RANDOM_STATE)

# ==================== CÁC HÀM TĂNG CƯỜNG DỮ LIỆU ====================
def augment_text(text, label):
    """Giữ nguyên logic augment cho 1936 mẫu sai thực tế"""
    if not isinstance(text, str) or len(text) < 5:
        return [text]

    results = [text]
    words = text.split()

    # 1. Thêm từ lóng/Emoji vào cuối
    slangs = [" vl", " vcl", " vãi", " @@", " :))", " :v", " !", " !!!"]
    if random.random() < 0.5:
        results.append(text + random.choice(slangs))

    # 2. Lặp âm cuối
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

    # 4. Typo nhẹ
    if random.random() < 0.3:
        results.append(text.replace("ch", "tr").replace("n", "nn"))

    return list(set(results))  # Xóa trùng

# ==================== MAIN PROCESSING ====================
# 1. Load 1936 mẫu sai từ error_analysis.xlsx
print("--- Loading 1936 error samples from real-world test ---")
df_error = pd.read_excel(error_file_path)
df_error['free_text'] = df_error['free_text'].astype(str)

print(f"Raw error samples: {len(df_error)}")
print("Error distribution (raw):")
print(df_error['label_id'].value_counts().sort_index())

# 2. Augment error data
print("\n--- Augmenting error samples ---")
aug_list = []
for _, row in df_error.iterrows():
    augmented_variants = augment_text(row['free_text'], row['label_id'])
    for variant in augmented_variants:
        aug_list.append({'free_text': variant, 'label_id': row['label_id']})

df_error_aug = pd.DataFrame(aug_list).drop_duplicates(subset=['free_text'])
print(f"Error samples after augmentation: {len(df_error_aug)}")
print("Error distribution after augment:")
print(df_error_aug['label_id'].value_counts().sort_index())

# ==================== 3. LOAD UNCERTAINTY FILE ====================
print(f"\n--- Loading uncertainty file: {UNCERTAINTY_FILE} ---")
if not os.path.exists(UNCERTAINTY_FILE):
    raise FileNotFoundError(
        f"Không tìm thấy {UNCERTAINTY_FILE}\n"
        f"Vui lòng chạy compute_uncertainty_vihsd.py trước."
    )

df_vihsd_unc = pd.read_csv(UNCERTAINTY_FILE)
print(f"Loaded ViHSD uncertainty: {len(df_vihsd_unc)} samples")

# ==================== 4. TÍNH SỐ MẪU CŨ CẦN LẤY ====================
n_error = len(df_error_aug)
n_old_target = int(n_error * OLD_NEW_RATIO)
print(f"\n{'='*60}")
print(f"Error (new) samples : {n_error}")
print(f"Old:new ratio      : 1:{OLD_NEW_RATIO}")
print(f"Old samples target : {n_old_target}")
print(f"{'='*60}")

# ==================== PER-LABEL UNCERTAINTY SAMPLING ====================
"""
VẤN ĐỀ: Nhãn 0 có uncertainty toàn cục rất cao (mean=0.96) vì model GĐ1 
gần như random guess với nhãn 0. Nếu sort uncertainty toàn cục, 
~80% mẫu cũ sẽ là nhãn 0 → GĐ3 bị nhãn 0 áp đảo, nhãn 1 (đang yếu nhất, 
recall=0.30) không được bổ sung đủ.

GIẢI PHÁP: Chia slot mẫu cũ theo từng nhãn riêng biệt (per-label).
Trong mỗi nhãn, chọn top uncertainty CAO NHẤT của nhãn đó.
Tỷ lệ slot được tính để:
- Nhãn 1 (Gây hấn) được ưu tiên cao nhất (vùng yếu nhất từ log test)
- Nhãn 0 không biến mất nhưng cũng không áp đảo
- Nhãn 2 được bổ sung hợp lý
"""

# Tính số lượng error data theo nhãn
error_counts = df_error_aug['label_id'].value_counts().sort_index()
n_err_0 = error_counts.get(0, 0)
n_err_1 = error_counts.get(1, 0)
n_err_2 = error_counts.get(2, 0)

print(f"\nError samples per label: 0={n_err_0}, 1={n_err_1}, 2={n_err_2}")

TARGET_PCT = {
    0: 0.30,   
    1: 0.55,   
    2: 0.15    
}

total_gd3 = n_error + n_old_target

# Tính số mẫu cũ cần lấy cho mỗi nhãn
target_per_label = {}
needed_from_old = {}
for lbl in [0, 1, 2]:
    target_total = int(total_gd3 * TARGET_PCT[lbl])
    have_error = error_counts.get(lbl, 0)
    need = max(0, target_total - have_error)
    target_per_label[lbl] = target_total
    needed_from_old[lbl] = need

# Điều chỉnh nếu tổng needed_from_old != n_old_target (làm tròn)
total_needed = sum(needed_from_old.values())
if total_needed != n_old_target:
    # Điều chỉnh nhãn 1 (ưu tiên) hoặc nhãn 0
    diff = n_old_target - total_needed
    needed_from_old[1] += diff  # Gán phần dư cho nhãn 1

print(f"\n[PER-LABEL ALLOCATION]")
print("-" * 60)
print(f"{'Label':<8} {'Target%':<10} {'TargetN':<10} {'Have(Error)':<14} {'Need(Old)':<10}")
print("-" * 60)
for lbl in [0, 1, 2]:
    print(f"{lbl:<8} {TARGET_PCT[lbl]*100:<10.1f} {target_per_label[lbl]:<10} {error_counts.get(lbl,0):<14} {needed_from_old[lbl]:<10}")
print("-" * 60)
print(f"Total old needed: {sum(needed_from_old.values())} (target: {n_old_target})")

# ==================== 5. CHỌN MẪU CŨ THEO TỪNG NHÃN ====================
final_old_list = []

for lbl in [0, 1, 2]:
    need = needed_from_old[lbl]
    if need <= 0:
        continue

    # Lọc pool theo nhãn
    pool = df_vihsd_unc[df_vihsd_unc['label_id'] == lbl].copy()

    if len(pool) == 0:
        print(f"[WARNING] No samples in ViHSD for label {lbl}")
        continue

    # Chọn top uncertainty CAO NHẤT trong nhãn này
    pool_sorted = pool.sort_values('uncertainty', ascending=False)
    n_take = min(need, len(pool_sorted))
    selected = pool_sorted.head(n_take)[['free_text', 'label_id', 'uncertainty']].copy()

    final_old_list.append(selected)
    print(f"\n[Label {lbl}] Selected {n_take}/{need} needed (pool size: {len(pool)})")
    print(f"  Uncertainty range: [{selected['uncertainty'].min():.4f}, {selected['uncertainty'].max():.4f}]")
    print(f"  Mean uncertainty: {selected['uncertainty'].mean():.4f}")

df_old = pd.concat(final_old_list, ignore_index=True) if final_old_list else pd.DataFrame()
df_old = df_old[['free_text', 'label_id']].copy()

print(f"\n{'='*60}")
print(f"[Old samples selected] Total: {len(df_old)}")
old_counts = df_old['label_id'].value_counts().sort_index()
for lbl in [0, 1, 2]:
    cnt = old_counts.get(lbl, 0)
    print(f"  Label {lbl}: {cnt} samples ({cnt/len(df_old)*100:.1f}%)")

# ==================== 6. MERGE & LƯU ====================
df_error_clean = df_error_aug[['free_text', 'label_id']].copy()
final_df = pd.concat([df_error_clean, df_old]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"\n{'='*60}")
print("FINAL GĐ3 TRAIN STATISTICS:")
print(f"{'='*60}")
final_counts = final_df['label_id'].value_counts().sort_index()
total = len(final_df)
for lbl in [0, 1, 2]:
    cnt = final_counts.get(lbl, 0)
    print(f"  Label {lbl}: {cnt} samples ({cnt/total*100:.1f}%)")
print(f"\n  TOTAL: {total} samples")
print(f"  Old:New = {len(df_old)}:{len(df_error_clean)} (1:{len(df_error_clean)/len(df_old):.2f})")
print(f"{'='*60}")

final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n[SUCCESS] Saved GĐ3 train to: {OUTPUT_FILE}")
