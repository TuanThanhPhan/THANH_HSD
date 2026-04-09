import pandas as pd
import os
from cleantext import clean_text_pipeline 

# Đường dẫn
BASE_PATH = r'F:\DE AN TOT NGHIEP\HSD_DEAN_TN\data'
vihsd_path = os.path.join(BASE_PATH, "ViHSD")
extra_file = os.path.join(BASE_PATH, "addHSD.xlsx")
output_dir = os.path.join(BASE_PATH, "processed")

os.makedirs(output_dir, exist_ok=True)

print("--- Bước 1: Đang đọc dữ liệu ---")

# Đọc dữ liệu gốc ViHSD
train_df = pd.read_csv(os.path.join(vihsd_path, "train.csv"))
dev_df   = pd.read_csv(os.path.join(vihsd_path, "dev.csv"))
test_df  = pd.read_csv(os.path.join(vihsd_path, "test.csv"))

# Đọc dữ liệu bổ sung
extra_df = pd.read_excel(extra_file)
print(f"Số mẫu bổ sung thêm: {len(extra_df)}")

# Gộp dữ liệu vào tập Train
final_train = pd.concat([train_df, extra_df], ignore_index=True)

print("--- Bước 2: Đang thực hiện làm sạch dữ liệu ---")
# Ép kiểu string (.astype(str)) để tránh lỗi nếu có dòng trống (NaN) hoặc chỉ có số
final_train['free_text'] = final_train['free_text'].astype(str).apply(clean_text_pipeline)
dev_df['free_text']      = dev_df['free_text'].astype(str).apply(clean_text_pipeline)
test_df['free_text']     = test_df['free_text'].astype(str).apply(clean_text_pipeline)

# Xáo trộn dữ liệu Train (Shuffle) để tăng tính khách quan khi huấn luyện
final_train = final_train.sample(frac=1, random_state=42).reset_index(drop=True)

print("--- Bước 3: Đang lưu dữ liệu đã làm sạch ---")
# Lưu dataset mới vào thư mục processed
final_train.to_csv(os.path.join(output_dir, "final_train.csv"), index=False, encoding='utf-8-sig')
dev_df.to_csv(os.path.join(output_dir, "final_dev.csv"), index=False, encoding='utf-8-sig')
test_df.to_csv(os.path.join(output_dir, "final_test.csv"), index=False, encoding='utf-8-sig')

print(f"Hoàn thành!")
print(f"- Tổng số mẫu Train: {len(final_train)}")