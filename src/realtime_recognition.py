import cv2
import numpy as np
import pickle
import mysql.connector
from resnet50v2_model import CustomResNet50V2  # Nhập mô hình từ tệp resnet50v2_model.py

class RealtimeRecognition:
    def __init__(self):
        # Tải mô hình ResNet50V2
        self.resnet_model = CustomResNet50V2(weights='imagenet')

        # Tải mô hình SVM
        with open('D:\\uni\\face_recognition_project\\models\\svm_model.pkl', 'rb') as f:
            self.svm_model = pickle.load(f)

        # Tải label encoder
        with open('D:\\uni\\face_recognition_project\\models\\label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)

        # Kết nối đến cơ sở dữ liệu
        self.db_connection = mysql.connector.connect(
            host="localhost",
            user="root",  # Thay đổi username nếu cần
            password="",  # Thay đổi password nếu cần
            database="diem_danh"
        )
        self.cursor = self.db_connection.cursor()

        print("Models and database connected successfully.")

    def process_frame(self, frame):
        # Tiền xử lý khung hình
        face_position = self.detect_face(frame)
        if face_position is not None:
            (face, (x, y, w, h)) = face_position
            embedding = self.get_face_embedding(face)
            name, probability = self.recognize_face(embedding)
            self.display_recognition(frame, name, probability, (x, y, w, h))
        return frame

    def detect_face(self, frame):
        # Chức năng phát hiện khuôn mặt
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = frame[y:y + h, x:x + w]  # Trả về khuôn mặt phát hiện được
                return (face, (x, y, w, h))  # Trả về khuôn mặt và tọa độ

        return None

    def get_face_embedding(self, face):
        # Tiền xử lý khuôn mặt và lấy đặc trưng từ mô hình ResNet50V2
        face = cv2.resize(face, (224, 224))  # Kích thước phù hợp với ResNet50V2
        face = np.expand_dims(face, axis=0)  # Thêm batch dimension
        face = face / 255.0  # Chuẩn hóa giá trị pixel
        embedding = self.resnet_model.predict(face)

        # Giảm số lượng đặc trưng của embedding
        if embedding.shape[1] > 50:
            embedding = embedding[:, :50]  # Giữ lại 20 đặc trưng đầu tiên

        return embedding

    def recognize_face(self, embedding):
        # Dự đoán bằng mô hình SVM
        prediction = self.svm_model.predict(embedding)
        probability = self.svm_model.predict_proba(embedding)
        # Lấy tên từ dự đoán
        if len(prediction) > 0:
            name = self.label_encoder.inverse_transform([prediction[0]])[0]  # Giá trị dự đoán là nhãn
        else:
            name = "Unknown"  # Không có dự đoán

        # Lấy xác suất cao nhất
        prob = probability[0].max() if len(probability) > 0 else 0.0

        return name, prob

    def display_recognition(self, frame, name, probability, face_position):
        # Hiển thị ô vuông xanh quanh khuôn mặt và tên cũng như xác suất trên khung hình
        (x, y, w, h) = face_position
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Vẽ ô vuông xanh

        # Hiển thị tên và xác suất trên khung hình
        text = f"{name} ({probability:.2f})"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def run(self):
        # Chạy webcam để nhận diện khuôn mặt
        cap = cv2.VideoCapture(0)  # Sử dụng webcam
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break
            
            processed_frame = self.process_frame(frame)
            cv2.imshow('Real-time Face Recognition', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.cursor.close()  # Đóng con trỏ
        self.db_connection.close()  # Đóng kết nối cơ sở dữ liệu

if __name__ == "__main__":
    recognizer = RealtimeRecognition()
    recognizer.run()