import numpy as np
import cv2
from tensorflow.keras.models import load_model
from utils import preprocess_face, get_embedding
from mtcnn import MTCNN
import os

class RealTimeRecognizer:
    def __init__(self, facenet_model_path, classifier_model_path, label_names_path):
        # Tải mô hình FaceNet và bộ phân loại
        self.facenet = load_model(facenet_model_path)
        self.classifier = load_model(classifier_model_path)
        self.label_names = np.load(label_names_path)
        self.detector = MTCNN()  # Sử dụng MTCNN để phát hiện khuôn mặt

    def recognize(self):
        cap = cv2.VideoCapture(0)  # Mở camera

        while cap.isOpened():
            ret, frame = cap.read()  # Đọc khung hình từ camera
            if not ret:
                break

            results = self.detector.detect_faces(frame)  # Phát hiện khuôn mặt trong khung hình
            for result in results:
                x, y, w, h = result['box']
                face = frame[y:y + h, x:x + w]  # Cắt khuôn mặt ra khỏi khung hình

                # Tiền xử lý khuôn mặt
                face = preprocess_face(face)
                if face is not None:
                    # Trích xuất embedding từ khuôn mặt
                    embedding = get_embedding(self.facenet, face)
                    # Dự đoán lớp của khuôn mặt
                    prediction = self.classifier.predict(np.expand_dims(embedding, axis=0))
                    label_index = np.argmax(prediction)  # Lấy chỉ số của lớp dự đoán
                    label_name = self.label_names[label_index]  # Lấy tên người dự đoán

                    # Vẽ hình chữ nhật quanh khuôn mặt và hiển thị tên
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Hiển thị video nhận diện
            cv2.imshow('Real-time Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Thoát khi nhấn 'q'
                break

        cap.release()  # Giải phóng tài nguyên camera
        cv2.destroyAllWindows()  # Đóng tất cả cửa sổ OpenCV


def preprocess_face(face):
    """ Tiền xử lý khuôn mặt trước khi truyền vào mô hình """
    if face is None or face.size == 0:  # Kiểm tra nếu ảnh không hợp lệ
        return None
    # Thực hiện các bước tiền xử lý khác ở đây (resize, normalization, v.v.)
    face = cv2.resize(face, (160, 160))  # Thay đổi kích thước ảnh
    face = (face / 255.0).astype(np.float32)  # Chuẩn hóa ảnh
    return face


def load_data(test_data_dir):
    """ Tải dữ liệu kiểm thử từ thư mục """
    X_test, y_test = [], []
    for label_name in os.listdir(test_data_dir):
        label_dir = os.path.join(test_data_dir, label_name)
        if os.path.isdir(label_dir):
            for image_name in os.listdir(label_dir):
                image_path = os.path.join(label_dir, image_name)
                print(f"Processing {image_path}...")  # Kiểm tra đường dẫn ảnh
                face = cv2.imread(image_path)
                if face is None:
                    print(f"Image {image_path} is not valid.")  # In thông báo nếu ảnh không hợp lệ
                    continue  # Bỏ qua ảnh không hợp lệ
                face = preprocess_face(face)
                if face is not None:
                    embedding = get_embedding(self.facenet, face)
                    X_test.append(embedding)
                    y_test.append(label_name)  # Lưu nhãn tương ứng
    return np.array(X_test), np.array(y_test)


if __name__ == "__main__":
    # Đường dẫn đến các mô hình đã huấn luyện
    facenet_model_path = r'D:\FACENET\face-recognition-project\models\facenet_keras.h5'
    classifier_model_path = 'face_recognition_classifier.h5'
    label_names_path = 'label_names.npy'  # Tên của file chứa nhãn
    test_data_dir = r'D:\FACENET\face-recognition-project\data\val'  # Đường dẫn đến dữ liệu kiểm thử

    # Khởi tạo và chạy nhận dạng
    recognizer = RealTimeRecognizer(facenet_model_path, classifier_model_path, label_names_path)
    recognizer.recognize()  # Bắt đầu nhận diện khuôn mặt theo thời gian thực

    # Đánh giá tỉ lệ đúng trên tập kiểm thử
    X_test, y_test = load_data(test_data_dir)
    prediction = classifier_model.predict(X_test)
    accuracy = np.mean(np.argmax(prediction, axis=1) == np.array([np.argmax(label) for label in y_test]))
    print(f"Accuracy: {accuracy * 100:.2f}%")
