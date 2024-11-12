import cv2
import numpy as np
from keras.models import load_model

class RealtimeRecognition:
    def __init__(self):
        # Tải mô hình FaceNet
        self.facenet_model = load_model('D:\\FACENET\\face_recognition_project\\models\\facenet_keras.h5')

        print("Model loaded successfully.")

    def process_frame(self, frame):
        # Tiền xử lý khung hình
        face_position = self.detect_face(frame)
        if face_position is not None:
            (face, (x, y, w, h)) = face_position
            embedding = self.get_face_embedding(face)
            self.display_recognition(frame, embedding, (x, y, w, h))
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
        # Tiền xử lý khuôn mặt và lấy embedding từ mô hình FaceNet
        face = cv2.resize(face, (160, 160))  # Kích thước phù hợp với FaceNet
        face = np.expand_dims(face, axis=0)  # Thêm batch dimension
        face = face / 255.0  # Chuẩn hóa giá trị pixel
        embedding = self.facenet_model.predict(face)

        return embedding

    def display_recognition(self, frame, embedding, face_position):
        # Hiển thị ô vuông xanh quanh khuôn mặt trên khung hình
        (x, y, w, h) = face_position
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Vẽ ô vuông xanh

        # Hiển thị embedding trên khung hình
        text = f"Embedding: {embedding.flatten()[:5]}"  # Hiển thị một phần embedding (ví dụ 5 giá trị đầu tiên)
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

if __name__ == "__main__":
    recognizer = RealtimeRecognition()
    recognizer.run()
