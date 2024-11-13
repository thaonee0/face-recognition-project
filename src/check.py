import numpy as np

def check_npz_structure(npz_file_path):
    """Kiểm tra cấu trúc của tệp .npz"""
    # Tải tệp .npz
    data = np.load(npz_file_path)
    
    # In ra các khóa có trong tệp .npz
    print(f"Các khóa trong tệp: {data.files}")
    
    # In ra dữ liệu chi tiết của từng khóa
    for key in data.files:
        print(f"\nDữ liệu cho khóa '{key}':")
        print(data[key].shape)  # In ra kích thước của dữ liệu
        print(data[key])  # In ra dữ liệu (hoặc chỉ in phần nhỏ nếu cần)

# Đường dẫn tới tệp .npz cần kiểm tra
dataset_path = 'data/processed/face_dataset.npz'
check_npz_structure(dataset_path)
