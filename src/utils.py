import cv2
import numpy as np
from mtcnn import MTCNN

# Tiền xử lý khuôn mặt trực tiếp từ mảng ảnh (dành cho webcam)
def preprocess_face(face):
    detector = MTCNN()
    results = detector.detect_faces(face)

    if results:
        x, y, w, h = results[0]['box']
        face = face[y:y + h, x:x + w]
        face = cv2.resize(face, (160, 160))
        face = face.astype('float32') / 255.0  # Chuẩn hóa giá trị pixel về [0, 1]
        face = (face - face.mean()) / face.std()  # Chuẩn hóa trung bình và độ lệch chuẩn
        return face
    return None

def get_embedding(model, face):
    face = np.expand_dims(face, axis=0)  # Thêm chiều batch cho mô hình
    embedding = model.predict(face)
    return embedding[0]
