import numpy as np

# Nạp tệp face_dataset.npz
data = np.load('D:\uni\face_recognition_project\data\processed\face_dataset.npz')

# Kiểm tra các mảng trong tệp
for key in data:
    print(f"{key}: {data[key].shape}")

# Nếu bạn muốn kiểm tra dữ liệu cụ thể
# Ví dụ, kiểm tra một mảng cụ thể
if 'some_key' in data:  # Thay 'some_key' bằng tên của mảng bạn muốn kiểm tra
    print(data['some_key'])  # In ra dữ liệu của mảng đó
