import numpy as np

# Tải tệp NPZ
data = np.load('D:\\uni\\face_recognition_project\\data\\processed\\face_embedding.npz')

# Kiểm tra các khóa trong tệp NPZ
print("Các khóa có trong tệp NPZ:", data.keys())

# Truy cập vào mảng arr_0
arr_0 = data['arr_0']

# Kiểm tra kích thước và một phần nội dung của arr_0
print("Kích thước của arr_0:", arr_0.shape)
print("Nội dung của arr_0 (một phần):", arr_0[:5])  # Hiển thị 5 mẫu đầu tiên