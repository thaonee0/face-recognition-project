import os
import sys
import io
import numpy as np
import joblib
from numpy import load
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Đảm bảo output của hệ thống sử dụng encoding utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def load_data(face_dataset_path, face_embedding_path, index_file):
    index_file_path = os.path.join(face_dataset_path, index_file)
    if not os.path.exists(index_file_path):
        print(f"Lỗi: Không tìm thấy tệp index: {index_file_path}")
        return None, None, None

    index_data = load(index_file_path, allow_pickle=True)['index_data']
    print(f"Tải {len(index_data)} mục từ tệp index.") 

    x_faces, y_labels, x_embeddings = [], [], []
    missing_embeddings = []

    for data_info in index_data:
        file_path = data_info['file_path']
        subdir = data_info['name']

        face_data_path = os.path.join(face_dataset_path, f"{subdir}_faces.npz")
        if not os.path.exists(face_data_path):
            print(f"Cảnh báo: Không tìm thấy dữ liệu khuôn mặt cho lớp {subdir}.")
            continue

        face_data = load(face_data_path)
        faces, labels = face_data['faces'], face_data['labels']
        print(f"Tải {len(faces)} mẫu từ lớp {subdir}.")
        x_faces.extend(faces)
        y_labels.extend(labels)

        embedding_file_path = os.path.join(face_embedding_path, f"{subdir}_embedding.npz")
        if not os.path.exists(embedding_file_path):
            print(f"Cảnh báo: Thiếu embedding cho lớp {subdir}.")
            missing_embeddings.append(subdir)
            continue

        embedding_data = load(embedding_file_path)
        embeddings = embedding_data['faces']
        x_embeddings.extend(embeddings)

    x_faces, y_labels, x_embeddings = map(np.array, [x_faces, y_labels, x_embeddings])

    print(f"Tổng số mẫu khuôn mặt: {len(y_labels)}")
    print(f"Các nhãn duy nhất: {np.unique(y_labels)}")
    if missing_embeddings:
        print(f"Cảnh báo: Thiếu embedding cho các lớp sau: {missing_embeddings}")

    return x_faces, x_embeddings, y_labels

def train_svm(x_embeddings, y_labels):
    # Chia dữ liệu thành tập train và test
    x_train, x_test, y_train, y_test = train_test_split(
        x_embeddings, y_labels, test_size=0.2, random_state=42
    )
    print(f"Tổng số mẫu: {len(y_labels)}")
    print(f"Số mẫu train: {len(y_train)}")
    print(f"Số mẫu test: {len(y_test)}")

    # Chuẩn hóa dữ liệu train
    normalizer = Normalizer(norm='l2')
    x_train_normalized = normalizer.fit_transform(x_train)

    # Mã hóa nhãn train
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Huấn luyện mô hình SVM
    model = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probability=True,
        random_state=42
    )
    model.fit(x_train_normalized, y_train_encoded)

    # Trả về các đối tượng cần thiết
    return model, label_encoder, normalizer, x_test, y_test

def save_model(model, label_encoder, normalizer, output_model):
    """Lưu mô hình, label encoder và normalizer vào tệp."""
    # Kiểm tra và tạo thư mục nếu chưa tồn tại
    output_dir = os.path.dirname(output_model)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Đã tạo thư mục: {output_dir}")
    
    # Lưu mô hình
    data = {
        'model': model,
        'label_encoder': label_encoder,
        'normalizer': normalizer
    }
    joblib.dump(data, output_model)
    print(f"Đã lưu mô hình vào tệp: {output_model}")

def load_model(output_model):
    """Tải mô hình, label encoder và normalizer từ tệp."""
    if not os.path.exists(output_model):
        print(f"Lỗi: Không tìm thấy tệp mô hình tại {output_model}")
        return None
    data = joblib.load(output_model)
    print("Đã tải mô hình từ tệp.")
    return data['model'], data['label_encoder'], data['normalizer']

if __name__ == "__main__":
    # Đường dẫn dữ liệu
    face_dataset_path = r"D:\FACENET\face-recognition-project\data\processed\face_dataset"
    face_embedding_path = r"D:\FACENET\face-recognition-project\data\processed\face_embedding"
    output_model = r"D:\FACENET\face-recognition-project\data\model\svm_model.joblib"
    index_file = 'index.npz'

    # Load data
    x_faces, x_embeddings, y_labels = load_data(face_dataset_path, face_embedding_path, index_file)
    
    if x_faces is None or x_embeddings is None or y_labels is None:
        print("Lỗi: Dữ liệu không hợp lệ. Tệp index.npz có thể bị thiếu hoặc không đúng định dạng.")
        sys.exit(1)

    # Train the model
    model, label_encoder, normalizer, x_test, y_test = train_svm(x_embeddings, y_labels)

    # Lưu mô hình
    save_model(model, label_encoder, normalizer, output_model)

    # Chuẩn hóa dữ liệu kiểm tra
    x_test_normalized = normalizer.transform(x_test)

    # Dự đoán nhãn kiểm tra
    y_pred = model.predict(x_test_normalized)
    y_true = label_encoder.transform(y_test)

    # In báo cáo
    print("\nBáo cáo kết quả:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    print(f"Mô hình đã được lưu tại: {output_model}")