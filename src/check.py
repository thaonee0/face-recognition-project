import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from utils import preprocess_face, get_embedding
import os

class FaceNetEvaluator:
    def __init__(self, model_path, images_path):
        self.model = load_model(model_path)
        self.images_path = images_path

    def load_data(self):
        """Load face images and labels from directory structure"""
        image_paths = []
        labels = []

        # Lặp qua thư mục con trong images_path để lấy ảnh và nhãn
        for label_dir in os.listdir(self.images_path):
            label_path = os.path.join(self.images_path, label_dir)
            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    image_paths.append(os.path.join(label_path, image_name))
                    labels.append(label_dir)

        faces, final_labels = [], []
        for image_path, label in zip(image_paths, labels):
            # Tiền xử lý ảnh
            face = preprocess_face(image_path)
            if face is not None:
                # Lấy embedding cho khuôn mặt
                embedding = get_embedding(self.model, face)
                faces.append(embedding)
                final_labels.append(label)
            else:
                print(f"Skipping invalid image: {image_path}")

        # Chuyển danh sách faces và labels thành numpy array
        faces = np.array(faces)
        final_labels = np.array(final_labels)
        
        # Mã hóa nhãn
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(final_labels)
        one_hot_labels = to_categorical(encoded_labels)
        
        # Lưu lại các nhãn cho dự đoán sau này
        np.save("label_names.npy", label_encoder.classes_)
        
        # Chia dữ liệu thành train và test
        return train_test_split(faces, one_hot_labels, test_size=0.2, random_state=42)

    def evaluate(self, X_test, y_test):
        """Đánh giá mô hình trên dữ liệu kiểm tra"""
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    model_path = 'face_recognition_classifier.h5'
    images_path = r'D:\FACENET\face-recognition-project\data\raw\val'  # Đảm bảo bạn chỉ định đúng đường dẫn tới dữ liệu kiểm tra
    
    evaluator = FaceNetEvaluator(model_path, images_path)
    X_train, X_test, y_train, y_test = evaluator.load_data()
    evaluator.evaluate(X_test, y_test)
