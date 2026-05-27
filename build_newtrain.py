import pandas as pd
import numpy as np
import os
import random
import config 

# ====================== CẤU HÌNH ======================
error_file_path = os.path.join(config.SAVE_DIR, "error_analysis.xlsx")
UNCERTAINTY_FILE = os.path.join(config.SAVE_DIR, "vihsd_train_uncertainty.csv")
OUTPUT_FILE = os.path.join(config.SAVE_DIR, "adapt_train.csv")

RANDOM_STATE = 42
OLD_NEW_RATIO = 2.0

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(RANDOM_STATE)

# ==================== AUGMENT MẠNH HƠN THEO NHÃN ====================
def augment_text(text, label):
    """
    Augment mạnh theo nhãn:
    - Nhãn 1 (Gây hấn): augment mạnh nhất (x4) vì là vùng yếu nhất
    - Nhãn 2 (Tiêu cực): augment vừa (x3)
    - Nhãn 0 (Bình thường): nhẹ (x2) vì đã nhiều và dễ
    """
    if not isinstance(text, str) or len(text) < 5:
        return [text]

    results = [text]
    words = text.split()

    # --- Các phép augment cơ bản (tất cả nhãn) ---

    # 1. Thêm từ lóng/Emoji vào cuối
    slangs = [" vl", " vcl", " vãi", " @@", " :))", " :v", " !", " !!!", " haha", " đmm"]
    if random.random() < 0.6:
        results.append(text + random.choice(slangs))

    # 2. Lặp âm cuối (Vocal Stretching)
    if len(words) > 0:
        target_idx = random.randint(0, len(words) - 1)
        if len(words[target_idx]) > 2:
            new_words = words.copy()
            new_words[target_idx] = new_words[target_idx] + new_words[target_idx][-1] * random.randint(2, 4)
            results.append(" ".join(new_words))

    # 3. Hoán đổi từ (Random Swap)
    if len(words) >= 4:
        new_words = words.copy()
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        results.append(" ".join(new_words))

    # 4. Typo nhẹ
    if random.random() < 0.4:
        typo_text = text
        if random.random() < 0.5:
            typo_text = typo_text.replace("ch", "tr")
        if random.random() < 0.5:
            typo_text = typo_text.replace("n", "nn")
        if random.random() < 0.3:
            typo_text = typo_text.replace("l", "n")
        results.append(typo_text)

    # --- Augment mạnh riêng cho nhãn 1 và 2 ---

    if label in [1, 2]:
        # 5. Xóa dấu câu ngẫu nhiên (mô phỏng viết vội)
        if random.random() < 0.5:
            no_punc = text.translate(str.maketrans('', '', '.,;:!?'))
            results.append(no_punc)

        # 6. Viết hoa ngẫu nhiên (mô phỏng cảm xúc mạnh)
        if random.random() < 0.4:
            words_upper = words.copy()
            for i in range(min(3, len(words_upper))):
                if random.random() < 0.5:
                    words_upper[i] = words_upper[i].upper()
            results.append(" ".join(words_upper))

        # 7. Thêm từ nhấn mạnh (nhãn 1 đặc biệt)
        if label == 1 and random.random() < 0.5:
            intensifiers = [" quá", " cực", " vcl", " thật", " kinh"]
            results.append(text + random.choice(intensifiers))

    # Nhãn 1: sinh thêm biến thể
    if label == 1 and len(results) < 5:
        # 8. Đảo ngược thứ tự câu (nếu đủ dài)
        if len(words) >= 5:
            rev_words = words[::-1]
            results.append(" ".join(rev_words))

    return list(set(results))

# ==================== MAIN PROCESSING ====================
print("--- Loading 1936 error samples ---")
df_error = pd.read_excel(error_file_path)
df_error['free_text'] = df_error['free_text'].astype(str)

print(f"Raw error samples: {len(df_error)}")
print("Error distribution (raw):")
print(df_error['label_id'].value_counts().sort_index())

# Augment error data
print("\n--- Augmenting error samples (stronger for label 1) ---")
aug_list = []
for _, row in df_error.iterrows():
    augmented_variants = augment_text(row['free_text'], row['label_id'])
    for variant in augmented_variants:
        aug_list.append({'free_text': variant, 'label_id': row['label_id']})

df_error_aug = pd.DataFrame(aug_list).drop_duplicates(subset=['free_text'])
print(f"Error samples after augmentation: {len(df_error_aug)}")
print("Error distribution after augment:")
print(df_error_aug['label_id'].value_counts().sort_index())

# ==================== LOAD UNCERTAINTY ====================
print(f"\n--- Loading uncertainty file: {UNCERTAINTY_FILE} ---")
if not os.path.exists(UNCERTAINTY_FILE):
    raise FileNotFoundError(f"Không tìm thấy {UNCERTAINTY_FILE}")

df_vihsd_unc = pd.read_csv(UNCERTAINTY_FILE)
print(f"Loaded ViHSD uncertainty: {len(df_vihsd_unc)} samples")

# ==================== TÍNH SỐ MẪU CŨ ====================
n_error = len(df_error_aug)
n_old_target = int(n_error * OLD_NEW_RATIO)
print(f"\n{'='*60}")
print(f"Error (new) samples : {n_error}")
print(f"Old:new ratio      : 1:{OLD_NEW_RATIO}")
print(f"Old samples target : {n_old_target}")
print(f"{'='*60}")

# ==================== PER-LABEL ALLOCATION ====================
error_counts = df_error_aug['label_id'].value_counts().sort_index()

# Điều chỉnh target: nhãn 1 nhiều hơn, nhãn 0 ít hơn
TARGET_PCT = {
    0: 0.25,   
    1: 0.60,   
    2: 0.15    
}

total_gd3 = n_error + n_old_target

target_per_label = {}
needed_from_old = {}
for lbl in [0, 1, 2]:
    target_total = int(total_gd3 * TARGET_PCT[lbl])
    have_error = error_counts.get(lbl, 0)
    need = max(0, target_total - have_error)
    target_per_label[lbl] = target_total
    needed_from_old[lbl] = need

# Điều chỉnh làm tròn
total_needed = sum(needed_from_old.values())
if total_needed != n_old_target:
    diff = n_old_target - total_needed
    needed_from_old[1] += diff

print(f"\n[PER-LABEL ALLOCATION]")
print("-" * 60)
print(f"{'Label':<8} {'Target%':<10} {'TargetN':<10} {'Have(Error)':<14} {'Need(Old)':<10}")
print("-" * 60)
for lbl in [0, 1, 2]:
    print(f"{lbl:<8} {TARGET_PCT[lbl]*100:<10.1f} {target_per_label[lbl]:<10} {error_counts.get(lbl,0):<14} {needed_from_old[lbl]:<10}")
print("-" * 60)
print(f"Total old needed: {sum(needed_from_old.values())} (target: {n_old_target})")

# ==================== CHỌN MẪU CŨ ====================
final_old_list = []

for lbl in [0, 1, 2]:
    need = needed_from_old[lbl]
    if need <= 0:
        continue

    pool = df_vihsd_unc[df_vihsd_unc['label_id'] == lbl].copy()
    if len(pool) == 0:
        print(f"[WARNING] No samples in ViHSD for label {lbl}")
        continue

    # Với nhãn 0: chọn uncertainty THẤP (easy samples, đỡ làm model bối rối)
    # Với nhãn 1,2: chọn uncertainty CAO (hard negatives)
    if lbl == 0:
        # Lấy mẫu nhãn 0 có uncertainty thấp nhất (model GĐ1 tự tin → pattern rõ ràng)
        pool_sorted = pool.sort_values('uncertainty', ascending=True)
        n_take = min(need, len(pool_sorted))
        selected = pool_sorted.head(n_take)[['free_text', 'label_id', 'uncertainty']].copy()
        print(f"\n[Label {lbl}] Selected {n_take} EASY samples (low uncertainty)")
    else:
        pool_sorted = pool.sort_values('uncertainty', ascending=False)
        n_take = min(need, len(pool_sorted))
        selected = pool_sorted.head(n_take)[['free_text', 'label_id', 'uncertainty']].copy()
        print(f"\n[Label {lbl}] Selected {n_take}/{need} HARD samples (high uncertainty)")

    print(f"  Uncertainty range: [{selected['uncertainty'].min():.4f}, {selected['uncertainty'].max():.4f}]")
    print(f"  Mean uncertainty: {selected['uncertainty'].mean():.4f}")
    final_old_list.append(selected)

df_old = pd.concat(final_old_list, ignore_index=True) if final_old_list else pd.DataFrame()
df_old = df_old[['free_text', 'label_id']].copy()

print(f"\n{'='*60}")
print(f"[Old samples selected] Total: {len(df_old)}")
old_counts = df_old['label_id'].value_counts().sort_index()
for lbl in [0, 1, 2]:
    cnt = old_counts.get(lbl, 0)
    print(f"  Label {lbl}: {cnt} samples ({cnt/len(df_old)*100:.1f}%)")

# ==================== MERGE & LƯU ====================
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

os.makedirs(config.SAVE_DIR, exist_ok=True)
final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n[SUCCESS] Saved GĐ3 train to: {OUTPUT_FILE}")
