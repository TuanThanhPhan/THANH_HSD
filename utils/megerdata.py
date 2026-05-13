import pandas as pd
import os
from cleantext import clean_text_pipeline


BASE_PATH = r'D:\DE AN TOT NGHIEP\HSD_DEAN\data'
vihsd_path = os.path.join(BASE_PATH, "ViHSD")
error_file_path = os.path.join(BASE_PATH, "error_analysis.xlsx")
OUTPUT_FILE = os.path.join(BASE_PATH, "adapt_train.csv")

RANDOM_STATE = 42

# Replay từ ViHSD
NORMAL_REPLAY = 2000
AGGRESSIVE_REPLAY = 1500
HATE_REPLAY = 1500

# Oversample dữ liệu lỗi
ERROR_OVERSAMPLE_TIMES = 3

# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("BƯỚC 1: LOAD DATA")
print("=" * 60)

# ----- Load ViHSD train -----
df_train_orig = pd.read_csv(
    os.path.join(vihsd_path, "train.csv")
)

# ----- Load error analysis -----
df_error = pd.read_excel(error_file_path)

# =========================================================
# KEEP REQUIRED COLUMNS
# =========================================================

required_cols = ['free_text', 'label_id']

df_train = df_train_orig[required_cols].copy()
df_error = df_error[required_cols].copy()

print(f"ViHSD Train: {len(df_train)}")
print(f"Error Samples: {len(df_error)}")

# =========================================================
# CLEAN TEXT
# =========================================================

print("\n" + "=" * 60)
print("BƯỚC 2: CLEAN TEXT")
print("=" * 60)

print("Cleaning ViHSD train...")

df_train['free_text'] = (
    df_train['free_text']
    .astype(str)
    .apply(clean_text_pipeline)
)

print("Cleaning error samples...")

df_error['free_text'] = (
    df_error['free_text']
    .astype(str)
    .apply(clean_text_pipeline)
)

# =========================================================
# REMOVE EMPTY TEXT
# =========================================================

df_train = df_train[
    df_train['free_text'].str.strip() != ""
].reset_index(drop=True)

df_error = df_error[
    df_error['free_text'].str.strip() != ""
].reset_index(drop=True)

# =========================================================
# REMOVE DUPLICATES
# =========================================================

print("\nRemoving duplicates...")

df_train = df_train.drop_duplicates(
    subset=['free_text', 'label_id']
)

df_error = df_error.drop_duplicates(
    subset=['free_text', 'label_id']
)

print(f"ViHSD after dedup: {len(df_train)}")
print(f"Error samples after dedup: {len(df_error)}")

# =========================================================
# STRATIFIED REPLAY FROM VIHSD
# =========================================================

print("\n" + "=" * 60)
print("BƯỚC 3: STRATIFIED REPLAY")
print("=" * 60)

normal_df = df_train[df_train['label_id'] == 0]
aggressive_df = df_train[df_train['label_id'] == 1]
hate_df = df_train[df_train['label_id'] == 2]

normal_sample = normal_df.sample(
    n=NORMAL_REPLAY,
    random_state=RANDOM_STATE
)

aggressive_sample = aggressive_df.sample(
    n=AGGRESSIVE_REPLAY,
    random_state=RANDOM_STATE
)

hate_sample = hate_df.sample(
    n=HATE_REPLAY,
    random_state=RANDOM_STATE
)

df_vihsd_replay = pd.concat(
    [
        normal_sample,
        aggressive_sample,
        hate_sample
    ],
    ignore_index=True
)

print(f"Replay samples: {len(df_vihsd_replay)}")

# =========================================================
# OVERSAMPLE ERROR SAMPLES
# =========================================================

print("\n" + "=" * 60)
print("BƯỚC 4: OVERSAMPLE ERROR DATA")
print("=" * 60)

df_error_boost = pd.concat(
    [df_error] * ERROR_OVERSAMPLE_TIMES,
    ignore_index=True
)

print(f"Error samples after oversample: {len(df_error_boost)}")

# =========================================================
# MERGE DATA
# =========================================================

print("\n" + "=" * 60)
print("BƯỚC 5: MERGE DATA")
print("=" * 60)

adapt_train = pd.concat(
    [
        df_vihsd_replay,
        df_error_boost
    ],
    ignore_index=True
)

# =========================================================
# FINAL SHUFFLE
# =========================================================

adapt_train = adapt_train.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# =========================================================
# LABEL DISTRIBUTION
# =========================================================

print("\nFinal label distribution:")

label_dist = (
    adapt_train['label_id']
    .value_counts()
    .sort_index()
)

print(label_dist)

print("\nLabel ratio:")

print(
    adapt_train['label_id']
    .value_counts(normalize=True)
    .sort_index()
)

print(f"\nTotal samples: {len(adapt_train)}")

# =========================================================
# SAVE
# =========================================================

print("\n" + "=" * 60)
print("BƯỚC 6: SAVE")
print("=" * 60)

adapt_train.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding='utf-8-sig'
)

print(f"\nSaved file:")
print(OUTPUT_FILE)

print("\nDONE.")