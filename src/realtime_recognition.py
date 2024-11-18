import sys
import io
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load
from mtcnn.mtcnn import MTCNN
import threading

# Đảm bảo output của hệ thống sử dụng encoding utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Đường dẫn tới các mô hình
facenet_model_path = r'D:\FACENET\face-recognition-project\models\facenet_keras.h5'
svm_model_path = r'D:\FACENET\face-recognition-project\models\svm_model.joblib'

# Tải mô hình FaceNet
facenet_model = load_model(facenet_model_path)
print("Đã tải mô hình FaceNet.")

# Tải mô hình SVM, label encoder, và normalizer
svm_data = load(svm_model_path)
svm_model = svm_data['model']
label_encoder = svm_data['label_encoder']
normalizer = svm_data['normalizer']
print("Đã tải mô hình SVM, Label Encoder và Normalizer.")

# Hàm lấy embedding cho khuôn mặt
def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = facenet_model.predict(samples)
    return yhat[0]

def recognize_faces_from_camera(callback):
    cap = cv2.VideoCapture(0)  # Mở camera

    detector = MTCNN()  # Khởi tạo bộ phát hiện khuôn mặt

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể nhận khung hình từ camera.")
            break

        # Phát hiện khuôn mặt
        results = detector.detect_faces(frame)
        if len(results) > 0:
            for face in results:
                x, y, w, h = face['box']
                x, y = max(0, x), max(0, y)  # Bảo đảm giá trị không âm
                cropped_face = frame[y:y + h, x:x + w]

                # Tiền xử lý khuôn mặt
                cropped_face = cv2.resize(cropped_face, (160, 160))
                embedding = get_embedding(cropped_face)
                embedding_normalized = normalizer.transform([embedding])

                # Dự đoán với SVM
                pred_label = svm_model.predict(embedding_normalized)
                pred_name = label_encoder.inverse_transform(pred_label)[0]

                # Dự đoán xác suất
                pred_prob = svm_model.predict_proba(embedding_normalized)
                max_prob = np.max(pred_prob)  # Xác suất cao nhất
                prob_percentage = max_prob * 100  # Chuyển đổi thành phần trăm

                if prob_percentage > 70:
                    # Gửi thông tin về GUI khi nhận diện thành công
                    callback(pred_name)  # Gửi tên

                # Hiển thị khung và nhãn
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{pred_name} ({prob_percentage:.2f}%)", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Hàm gọi khi nhận diện thành công
def start_recognition(callback):
    threading.Thread(target=recognize_faces_from_camera, args=(callback,)).start()

if __name__ == "__main__":
    start_recognition(lambda name: print(f"{name}"))
