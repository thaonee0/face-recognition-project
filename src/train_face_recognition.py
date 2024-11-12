import numpy as np
from keras.models import load_model
import os# cmm

def load_data(face_dataset_path, face_embedding_path):
    # Tải dữ liệu khuôn mặt và embedding từ các tệp npz
    data = np.load(face_dataset_path)
    embeddings = np.load(face_embedding_path)

    return data['arr_0'], embeddings['arr_0']  # Giả sử dữ liệu và embeddings nằm trong arr_0

def save_embeddings(embeddings, labels, embedding_path, label_path):
    # Lưu embeddings và nhãn vào các tệp .npz
    np.save(embedding_path, embeddings)
    np.save(label_path, labels)
    print(f"Embeddings saved to {embedding_path}")
    print(f"Labels saved to {label_path}")

def main():
    # Đường dẫn đến các tệp dữ liệu của bạn
    face_dataset_path = 'D:\\FACENET\\face_recognition_project\\data\\face_dataset.npz'
    face_embedding_path = 'D:\\FACENET\\face_recognition_project\\data\\face_embedding.npz'

    # Tải dữ liệu và embeddings
    faces, embeddings = load_data(face_dataset_path, face_embedding_path)

    # Gán nhãn cho mỗi khuôn mặt (ví dụ: bạn có thể gán nhãn thủ công ở đây)
    # Lưu ý: labels cần phải khớp với mỗi khuôn mặt trong dataset của bạn.
    labels = np.array([f"Person_{i}" for i in range(len(faces))])

    # Lưu embeddings và nhãn vào tệp .npz
    save_embeddings(embeddings, labels, 'D:\\FACENET\\face_recognition_project\\data\\saved_embeddings.npy', 'D:\\FACENET\\face_recognition_project\\data\\saved_labels.npy')

if __name__ == '__main__':
    main()
