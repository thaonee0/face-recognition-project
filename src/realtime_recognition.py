import mysql.connector
from mysql.connector import Error
from datetime import datetime, date
from tkinter import messagebox
import numpy as np
from mtcnn import MTCNN
from keras.models import load_model
import cv2

class RealTimeRecognizer:
    def __init__(self, facenet_model_path, classifier_model_path, label_names_path, db_config):
        self.facenet = load_model(facenet_model_path, compile=False)
        self.classifier = load_model(classifier_model_path, compile=False)
        self.label_names = np.load(label_names_path)
        self.detector = MTCNN()
        self.db_config = db_config
        self.attended_students = set()  # Để theo dõi những sinh viên đã điểm danh trong ngày

    def preprocess_face(self, img, required_size=(160, 160)):
        if img is None:
            return None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, required_size)
        face_pixels = img_resized.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        face_pixels = np.expand_dims(face_pixels, axis=0)
        return face_pixels

    def get_embedding(self, face_pixels):
        return self.facenet.predict(face_pixels)[0]

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
                    
                    # Hiển thị thông báo thành công trên giao diện Tkinter
                    messagebox.showinfo("Điểm danh thành công", f"""
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

    def recognize_frame(self, frame):
        results = self.detector.detect_faces(frame)
        recognition_results = []

        for result in results:
            x, y, w, h = result['box']
            face = frame[y:y + h, x:x + w]
            face_processed = self.preprocess_face(face)

            if face_processed is not None:
                embedding = self.get_embedding(face_processed)
                prediction = self.classifier.predict(np.expand_dims(embedding, axis=0))
                label_index = np.argmax(prediction)
                label_name = self.label_names[label_index]
                confidence = np.max(prediction)

                if confidence > 0.50:
                    name, mssv = label_name.split('-')
                    recognition_results.append({'name': name, 'mssv': mssv, 'box': (x, y, w, h), 'confidence': confidence})
                    
                    # Cập nhật thông tin điểm danh vào database
                    self.check_and_update_database(label_name)

        return recognition_results
