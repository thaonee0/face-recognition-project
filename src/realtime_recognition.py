import numpy as np
import cv2
from tensorflow.keras.models import load_model
from utils import preprocess_face, get_embedding
from mtcnn import MTCNN

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

if __name__ == "__main__":
    # Đường dẫn đến các mô hình đã huấn luyện
    facenet_model_path = r'D:\FACENET\face-recognition-project\models\facenet_keras.h5'
    classifier_model_path = 'face_recognition_classifier.h5'
    label_names_path = 'label_names.npy'  # Tên của file chứa nhãn

    # Khởi tạo và chạy nhận dạng
    recognizer = RealTimeRecognizer(facenet_model_path, classifier_model_path, label_names_path)
    recognizer.recognize()  # Bắt đầu nhận diện khuôn mặt theo thời gian thực
