import cv2
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load
from mtcnn.mtcnn import MTCNN

# Đường dẫn tới các mô hình
facenet_model_path = r'D:\FACENET\face-recognition-project\models\facenet_keras.h5'
svm_model_path = r'D:\FACENET\face-recognition-project\models\svm_model.joblib'

# Tải các mô hình
facenet_model = load_model(facenet_model_path)
svm_model = load(svm_model_path)

# Hàm lấy embedding cho khuôn mặt
def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = facenet_model.predict(samples)
    return yhat[0]

# Nhận diện thời gian thực
def recognize_faces():
    detector = MTCNN()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Phát hiện khuôn mặt
        results = detector.detect_faces(frame)
        for result in results:
            x1, y1, width, height = result['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = frame[y1:y2, x1:x2]

            # Resize và chuẩn hóa khuôn mặt
            try:
                face = cv2.resize(face, (160, 160))
                embedding = get_embedding(face)

                # Dự đoán nhãn
                label = svm_model.predict([embedding])[0]
                confidence = max(svm_model.predict_proba([embedding])[0])

                # Hiển thị kết quả
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error processing face: {e}")

        cv2.imshow("Face Recognition", frame)

        # Thoát nếu nhấn phím Q
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
