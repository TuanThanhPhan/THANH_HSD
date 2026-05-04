import pandas as pd
import os
from cleantext import clean_text_pipeline 

# --- Cấu hình đường dẫn  ---
BASE_PATH = r'D:\DE AN TOT NGHIEP\HSD_DEAN\data'
vihsd_path = os.path.join(BASE_PATH, "ViHSD")
error_file_path = os.path.join(BASE_PATH, "error_analysis.xlsx")

print("--- Bước 1: Đọc và Đồng bộ hóa dữ liệu ---")

# 1. Đọc dữ liệu Train gốc từ ViHSD
df_train_orig = pd.read_csv(os.path.join(vihsd_path, "train.csv"))

# 2. Đọc file lỗi
df_error = pd.read_excel(error_file_path)

# Chỉ lấy 2 cột cần thiết để đảm bảo tính nhất quán khi gộp
df_error_subset = df_error[['free_text', 'label_id']].copy()
df_train_subset = df_train_orig[['free_text', 'label_id']].copy()

# 3. Gộp tập Train gốc với tập câu lỗi bổ sung
train_extend = pd.concat([df_train_subset, df_error_subset], ignore_index=True)

print(f"Số lượng Train gốc: {len(df_train_subset)}")
print(f"Số lượng bổ sung từ file lỗi: {len(df_error_subset)}")
print(f"Tổng cộng sau khi gộp: {len(train_extend)}")

print("--- Bước 2: Làm sạch dữ liệu và Xáo trộn ---")

# 4. Áp dụng pipeline làm sạch 
print("Đang thực hiện clean text...")
train_extend['free_text'] = train_extend['free_text'].astype(str).apply(clean_text_pipeline)

# Loại bỏ các dòng trống sau khi làm sạch
train_extend = train_extend[train_extend['free_text'].str.strip() != ""].reset_index(drop=True)

# 5. Xáo trộn tập dữ liệu (Shuffle)
train_extend = train_extend.sample(frac=1, random_state=42).reset_index(drop=True)

print("--- Bước 3: Xuất file kết quả ---")
# 6. Lưu file train_extend.csv vào thư mục data
output_file = os.path.join(BASE_PATH, "train_extend.csv")
train_extend.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"Xong! Đã tạo file '{output_file}' thành công.")