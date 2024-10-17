from random import choice
from numpy import load, expand_dims
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_data(face_dataset, face_embedding):
    data = load(face_dataset)
    embeddings = load(face_embedding)
    x_train, y_train = data['arr_0'], data['arr_1']
    x_embed, y_embed = embeddings['arr_0'], embeddings['arr_1']
    return x_train, y_train, x_embed, y_embed

def train_svm(x_train, y_train):
    # Chuyển đổi nhãn thành số
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)

    # Huấn luyện SVM
    svm = SVC(kernel='linear', probability=True)
    svm.fit(x_train, y_train_encoded)

    return svm, encoder

def test_model(svm, encoder, x_embed, y_embed):
    y_pred = svm.predict(x_embed)
    y_pred_inverse = encoder.inverse_transform(y_pred)

    # Tính độ chính xác
    accuracy = accuracy_score(y_embed, y_pred_inverse)
    return accuracy

if __name__ == '__main__':
    # Gán trực tiếp các đường dẫn cho dữ liệu khuôn mặt và embeddings
    face_dataset_path = r"D:\FACENET\face_recognition_project\data\processed\face_dataset.npz"
    face_embedding_path = r"D:\FACENET\face_recognition_project\data\processed\face_embedding.npz"

    # Tải dữ liệu
    x_train, y_train, x_embed, y_embed = load_data(face_dataset_path, face_embedding_path)

    # Huấn luyện và kiểm tra mô hình
    svm, encoder = train_svm(x_train, y_train)
    accuracy = test_model(svm, encoder, x_embed, y_embed)

    print(f"Độ chính xác của mô hình là: {accuracy:.2f}")
