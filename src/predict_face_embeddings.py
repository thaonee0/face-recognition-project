import sys
import os
import numpy as np
import tensorflow as tf
import cv2  # Thêm import OpenCV

# Thêm thư mục gốc vào đường dẫn tìm kiếm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from resnet50v2_model import ResNet50V2  # Import mô hình từ file resnet50v2_model.py

# Đường dẫn đến file dữ liệu khuôn mặt đã trích xuất
face_dataset_path = r'D:\uni\face_recognition_project\data\processed\face_dataset.npz'

# Đường dẫn đến file lưu embeddings
face_embedding_path = r'D:\uni\face_recognition_project\data\processed\face_embedding.npz'

# Tạo embeddings cho ảnh
def generate_embeddings(resnet_model, dataset):
    embeddings = []
    for face in dataset:
        # Thay đổi kích thước ảnh về 224x224 trước khi dự đoán
        face_resized = cv2.resize(face, (224, 224 , 3))
        embeddings.append(resnet_model.predict(np.expand_dims(face_resized, axis=0))[0])
    return np.asarray(embeddings)

if __name__ == '__main__':
    # Tải dữ liệu
    data = np.load(face_dataset_path)
    x_train = data['arr_0']  # Lấy khuôn mặt đã trích xuất

    # Tải mô hình ResNet50V2
    model = ResNet50V2(weights='imagenet')

    # Tính toán embeddings
    embeddings = generate_embeddings(model, x_train)

    # Lưu embeddings vào file .npz
    np.savez_compressed(face_embedding_path, embeddings)
    print("Embeddings đã được lưu tại:", face_embedding_path)