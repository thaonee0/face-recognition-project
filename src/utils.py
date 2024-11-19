import cv2
import numpy as np
from mtcnn import MTCNN

# Tiền xử lý khuôn mặt trực tiếp từ mảng ảnh (dành cho webcam)
def preprocess_face(face):
    detector = MTCNN()  # Khởi tạo detector MTCNN
    results = detector.detect_faces(face)

    if len(results) > 0:  # Kiểm tra có khuôn mặt nào được phát hiện không
        # Lặp qua tất cả các khuôn mặt được phát hiện
        for result in results:
            x, y, w, h = result['box']
            face_cropped = face[y:y + h, x:x + w]  # Cắt khuôn mặt
            if face_cropped.size == 0:  # Kiểm tra xem ảnh cắt có hợp lệ không
                return None  # Trả về None nếu không có khuôn mặt hợp lệ
            face_resized = cv2.resize(face_cropped, (160, 160))  # Resize khuôn mặt
            face_resized = face_resized.astype('float32') / 255.0  # Chuẩn hóa pixel về [0, 1]
            face_resized = (face_resized - face_resized.mean()) / face_resized.std()  # Chuẩn hóa trung bình và độ lệch chuẩn
            return face_resized
    return None  # Nếu không phát hiện khuôn mặt nào

def get_embedding(model, face):
    face = np.expand_dims(face, axis=0)  # Thêm chiều batch cho mô hình
    embedding = model.predict(face)
    return embedding[0]
