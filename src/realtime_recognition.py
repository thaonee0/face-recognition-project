import numpy as np
import cv2
from tensorflow.keras.models import load_model
import mysql.connector
from mysql.connector import Error
from datetime import datetime, date
from mtcnn import MTCNN
import tensorflow as tf
import os

def preprocess_face(img, required_size=(160, 160)):
    """Resize và chuẩn hóa ảnh khuôn mặt"""
    try:
        if img is None:
            return None
            
        # Chuyển đổi BGR sang RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize ảnh
        img_resized = cv2.resize(img_rgb, required_size)
        
        # Chuyển đổi sang float32 và chuẩn hóa
        face_pixels = img_resized.astype('float32')
        
        # Chuẩn hóa 
        mean = face_pixels.mean()
        std = face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        
        # Mở rộng chiều
        face_pixels = np.expand_dims(face_pixels, axis=0)
        
        return face_pixels
        
    except Exception as e:
        print(f"Lỗi trong quá trình xử lý ảnh: {str(e)}")
        return None

def get_embedding(model, face_pixels):
    """Trích xuất đặc trưng khuôn mặt sử dụng FaceNet"""
    try:
        # Dự đoán
        yhat = model.predict(face_pixels)
        return yhat[0]
        
    except Exception as e:
        print(f"Lỗi khi trích xuất embedding: {str(e)}")
        return None

class RealTimeRecognizer:
    def __init__(self, facenet_model_path, classifier_model_path, label_names_path, db_config):
        """
        Khởi tạo Real-time Face Recognition system
        
        Parameters:
        - facenet_model_path: Đường dẫn đến model FaceNet
        - classifier_model_path: Đường dẫn đến model phân loại
        - label_names_path: Đường dẫn đến file nhãn
        - db_config: Cấu hình kết nối database
        """
        self.facenet = load_model(facenet_model_path, compile=False)
        self.classifier = load_model(classifier_model_path, compile=False)
        self.label_names = np.load(label_names_path)
        self.detector = MTCNN()
        self.db_config = db_config
        self.attended_students = set()

    def extract_student_info(self, label_name):
        """Trích xuất tên và mã số sinh viên từ nhãn"""
        parts = label_name.split('-')
        name = parts[0]
        mssv = parts[1] if len(parts) > 1 else ""
        return name, mssv

    def is_already_attended(self, student_id):
        """Kiểm tra sinh viên đã điểm danh trong ngày chưa"""
        try:
            connection = mysql.connector.connect(**self.db_config)
            if connection.is_connected():
                cursor = connection.cursor()
                
                today = date.today().strftime('%Y-%m-%d')
                query = """
                    SELECT COUNT(*) 
                    FROM diem_danh 
                    WHERE ten_sinh_vien = %s 
                    AND DATE(ngay_gio_diem_danh) = %s
                """
                cursor.execute(query, (student_id, today))
                count = cursor.fetchone()[0]
                
                cursor.close()
                connection.close()
                
                return count > 0
                
        except Error as e:
            print(f"Lỗi khi kiểm tra điểm danh: {e}")
            return False

    def check_and_update_database(self, label_name):
        """Kiểm tra và cập nhật thông tin điểm danh vào database"""
        try:
            student_name, mssv = self.extract_student_info(label_name)
            
            connection = mysql.connector.connect(**self.db_config)
            if connection.is_connected():
                cursor = connection.cursor(dictionary=True)

                # Lấy thông tin sinh viên từ bảng sinhvien
                query = """
                    SELECT id, ten_sinh_vien, lop, khoa, mssv 
                    FROM sinhvien 
                    WHERE ten_sinh_vien = %s
                """
                cursor.execute(query, (student_name,))
                student = cursor.fetchone()

                if student:
                    student_id = student['id']
                    
                    # Kiểm tra đã điểm danh chưa
                    if student_id in self.attended_students or self.is_already_attended(student_id):
                        print(f"Sinh viên {student['ten_sinh_vien']} - MSSV: {student['mssv']} đã được điểm danh hôm nay")
                        return

                    # Thêm điểm danh mới
                    now = datetime.now()
                    ngay_gio_diem_danh = now.strftime('%Y-%m-%d %H:%M:%S')

                    insert_query = """
                        INSERT INTO diem_danh (ten_sinh_vien, ngay_gio_diem_danh)
                        VALUES (%s, %s)
                    """
                    cursor.execute(insert_query, (student_id, ngay_gio_diem_danh))
                    connection.commit()

                    self.attended_students.add(student_id)
                    print(f"""
                    Đã điểm danh thành công:
                    - Tên: {student['ten_sinh_vien']}
                    - MSSV: {student['mssv']}
                    - Lớp: {student['lop']}
                    - Khoa: {student['khoa']}
                    - Thời gian: {ngay_gio_diem_danh}
                    """)
                else:
                    print(f"Không tìm thấy sinh viên {student_name} trong cơ sở dữ liệu")

                cursor.close()

        except Error as e:
            print(f"Đã xảy ra lỗi khi kết nối MySQL: {e}")

        finally:
            if 'connection' in locals() and connection.is_connected():
                connection.close()

    def recognize(self):
        """Thực hiện nhận diện khuôn mặt thời gian thực"""
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.detector.detect_faces(frame)
            for result in results:
                x, y, w, h = result['box']
                face = frame[y:y + h, x:x + w]

                face_processed = preprocess_face(face)
                if face_processed is not None:
                    embedding = get_embedding(self.facenet, face_processed)
                    prediction = self.classifier.predict(np.expand_dims(embedding, axis=0))
                    label_index = np.argmax(prediction)
                    label_name = self.label_names[label_index]
                    confidence = np.max(prediction)

                    # Hiển thị tên và MSSV
                    display_name, mssv = self.extract_student_info(label_name)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Hiển thị tên phía trên hình chữ nhật
                    cv2.putText(frame, f"{display_name}", (x, y - 25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    # Hiển thị MSSV phía dưới tên
                    cv2.putText(frame, f"MSSV: {mssv}", (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if confidence > 0.55:  # Ngưỡng tin cậy
                        print(f"Dự đoán: {display_name} (MSSV: {mssv}) với độ chính xác {confidence:.2f}")
                        self.check_and_update_database(label_name)

            cv2.imshow('Real-time Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Cấu hình đường dẫn
    facenet_model_path = 'models/facenet_keras.h5'
    classifier_model_path = 'face_recognition_classifier.h5'
    label_names_path = 'label_names.npy'

    # Cấu hình database
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': '',
        'database': 'diem_danh'
    }

    # Khởi tạo và chạy hệ thống
    recognizer = RealTimeRecognizer(facenet_model_path, classifier_model_path, label_names_path, db_config)
    recognizer.recognize()