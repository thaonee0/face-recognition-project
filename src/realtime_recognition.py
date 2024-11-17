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

def recognize_faces_from_camera(model_path, face_extractor, video_source=0):
    # Tải mô hình
    model, label_encoder, normalizer = load_model(model_path)
    if model is None:
        print("Không thể tải mô hình. Thoát chương trình.")
        return

    # Mở camera
    cap = cv2.VideoCapture(video_source)
    print("Mở camera... Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể nhận khung hình từ camera.")
            break

        # Phát hiện và trích xuất khuôn mặt từ khung hình
        faces = face_extractor(frame)
        for face in faces:
            # Tiền xử lý khuôn mặt và tính embedding
            face_embedding = compute_face_embedding(face)  # Hàm này là của bạn
            face_embedding_normalized = normalizer.transform([face_embedding])

            # Dự đoán nhãn
            pred_label = model.predict(face_embedding_normalized)
            pred_name = label_encoder.inverse_transform(pred_label)[0]

            # Hiển thị tên trên khung hình
            x, y, w, h = face['box']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, pred_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Nhận diện khuôn mặt", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
