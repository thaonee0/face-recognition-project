import numpy as np
from keras.models import load_model
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
import pickle
import os

def load_data(face_dataset_path, face_embedding_path):
    # Tải dữ liệu khuôn mặt và embedding từ các tệp npz
    data = np.load(face_dataset_path)
    embeddings = np.load(face_embedding_path)

    return data['arr_0'], embeddings['arr_0']  # Giả sử dữ liệu và embeddings nằm trong arr_0

def train_svm(embeddings, labels):
    # Huấn luyện mô hình SVM với các embedding và nhãn
    svm_model = make_pipeline(LabelEncoder(), SVC(kernel='linear', probability=True))
    svm_model.fit(embeddings, labels)

    return svm_model

def save_model(model, model_path):
    # Lưu mô hình đã huấn luyện
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

def main():
    # Đường dẫn đến các tệp dữ liệu của bạn
    face_dataset_path = 'D:\\FACENET\\face_recognition_project\\data\\face_dataset.npz'
    face_embedding_path = 'D:\\FACENET\\face_recognition_project\\data\\face_embedding.npz'

    # Tải dữ liệu và embeddings
    faces, embeddings = load_data(face_dataset_path, face_embedding_path)

    # Gán nhãn cho mỗi khuôn mặt (ví dụ: bạn có thể gán nhãn thủ công ở đây)
    # Lưu ý: labels cần phải khớp với mỗi khuôn mặt trong dataset của bạn.
    labels = np.array([f"Person_{i}" for i in range(len(faces))])

    # Huấn luyện mô hình SVM để nhận diện khuôn mặt
    svm_model = train_svm(embeddings, labels)

    # Lưu mô hình đã huấn luyện
    save_model(svm_model, 'D:\\FACENET\\face_recognition_project\\models\\svm_model.pkl')

if __name__ == '__main__':
    main()
