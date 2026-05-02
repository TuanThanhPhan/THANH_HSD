import pandas as pd
import os
from cleantext import clean_text_pipeline 

# --- Cấu hình đường dẫn ---
BASE_PATH = r'D:\DE AN TOT NGHIEP\HSD_DEAN\data'
vihsd_path = os.path.join(BASE_PATH, "ViHSD")
real_data_file = os.path.join(BASE_PATH, "dataHSD.xlsx")

print("--- Bước 1: Chia tách và Gộp dữ liệu ---")

# 1. Đọc dữ liệu
train_df = pd.read_csv(os.path.join(vihsd_path, "train.csv"))
df_real = pd.read_excel(real_data_file)

# 2. Chia tập thực tế: 2000 mẫu để train, 6764 mẫu làm tập TestHSD
df_subset_2k = df_real.sample(n=2000, random_state=42)
df_test_hsd  = df_real.drop(df_subset_2k.index)

# 3. Gộp vào tập Train gốc
train_extend = pd.concat([train_df, df_subset_2k], ignore_index=True)

print("--- Bước 2: Làm sạch dữ liệu mới ---")

# Chỉ làm sạch tập train_extend và TestHSD
train_extend['free_text'] = train_extend['free_text'].astype(str).apply(clean_text_pipeline)
df_test_hsd['free_text']  = df_test_hsd['free_text'].astype(str).apply(clean_text_pipeline)

# Xáo trộn tập train
train_extend = train_extend.sample(frac=1, random_state=42).reset_index(drop=True)

print("--- Bước 3: Lưu file vào thư mục data ---")

# Lưu 2 file quan trọng nhất
train_extend.to_csv(os.path.join(BASE_PATH, "train_extend.csv"), index=False, encoding='utf-8-sig')
df_test_hsd.to_csv(os.path.join(BASE_PATH, "TestHSD.csv"), index=False, encoding='utf-8-sig')

print(f"Xong! Đã tạo 'train_extend.csv' và 'TestHSD.csv' tại {BASE_PATH}")