import mysql.connector
from mysql.connector import Error
from datetime import datetime, date
import os

class DatabaseHandler:
    def __init__(self):
        self.connection = None
        self.connect_to_database()

    def connect_to_database(self):
        """Kết nối tới cơ sở dữ liệu."""
        try:
            self.connection = mysql.connector.connect(
                host="localhost",
                user="root",  # Thay đổi username nếu cần
                password="",  # Thay đổi password nếu cần
                database="diem_danh"
            )
            if self.connection.is_connected():
                print("Database connection successful")
        except Error as e:
            print(f"Lỗi kết nối database: {e}")

    def reconnect(self):
        """Tự động kết nối lại nếu kết nối bị mất."""
        if not self.connection or not self.connection.is_connected():
            print("Kết nối cơ sở dữ liệu đã mất, thử kết nối lại...")
            self.connect_to_database()

    def add_student(self, ten_sinh_vien, lop, khoa, anh_dai_dien, mssv):
        """Thêm sinh viên mới vào cơ sở dữ liệu."""
        self.reconnect()
        try:
            cursor = self.connection.cursor()
            sql = """INSERT INTO sinhvien (ten_sinh_vien, lop, khoa, anh_dai_dien, mssv) 
                     VALUES (%s, %s, %s, %s, %s)"""
            values = (ten_sinh_vien, lop, khoa, anh_dai_dien, mssv)
            cursor.execute(sql, values)
            self.connection.commit()
            return cursor.lastrowid
        except Error as e:
            print(f"Lỗi thêm sinh viên: {e}")
            return None
        finally:
            if cursor:
                cursor.close()

    """
    def get_sv_from_name(self, name):
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute("SELECT ten_sinh_vien FROM sinhvien WHERE ten_sinh_vien = %s", (name,))
        student = cursor.fetchone()
        cursor.close()
        
        if student:
            return student['ten_sinh_vien']
        else:
            return None 
    """

    def check_attendance(self, recognized_name):
        """Kiểm tra và thực hiện điểm danh."""
        self.reconnect()
        try:
            cursor = self.connection.cursor(dictionary=True)

            # Tách recognized_name thành ten_sinh_vien và mssv
            ten_sinh_vien, mssv = recognized_name.split('-')  # Tách từ recognized_name
            
            print(f"Đang kiểm tra điểm danh cho sinh viên: {ten_sinh_vien} với MSSV: {mssv}")
            
            # Lấy thông tin sinh viên từ bảng sinhvien theo ten_sinh_vien
            cursor.execute("SELECT * FROM sinhvien WHERE ten_sinh_vien = %s AND mssv = %s", (ten_sinh_vien, mssv))
            student = cursor.fetchone()
            
            if not student:
                print(f"Không tìm thấy sinh viên {ten_sinh_vien} - {mssv} trong cơ sở dữ liệu.")
                return False, f"Không tìm thấy sinh viên {ten_sinh_vien} - {mssv} trong cơ sở dữ liệu."
            
            print(f"Thông tin sinh viên: {student}")
            
            # Kiểm tra đã điểm danh hôm nay chưa
            today = date.today().strftime('%Y-%m-%d')
            cursor.execute("""
                SELECT * FROM diem_danh 
                WHERE id_sinh_vien = %s AND DATE(ngay_gio_diem_danh) = %s
            """, (student['id'], today))
            
            existing_attendance = cursor.fetchone()
            if existing_attendance:
                print(f"Sinh viên {ten_sinh_vien} đã điểm danh hôm nay.")
                return False, f"Sinh viên {ten_sinh_vien} đã điểm danh hôm nay."
            
            # Thêm điểm danh mới
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Thêm điểm danh cho sinh viên {student['ten_sinh_vien']} vào lúc {now}")
            cursor.execute("""
                INSERT INTO diem_danh (id_sinh_vien, ngay_gio_diem_danh)
                VALUES (%s, %s)
            """, (student['id'], now))
            
            self.connection.commit()
            
            success_message = f"""Điểm danh thành công:
            - Tên: {student['ten_sinh_vien']}
            - MSSV: {student.get('mssv', 'N/A')}
            - Lớp: {student['lop']}
            - Thời gian: {now}"""
            
            print(success_message)
            return True, success_message

        except Error as e:
            print(f"Lỗi khi điểm danh: {str(e)}")  # In lỗi đúng cách
            return False, f"Lỗi khi điểm danh: {str(e)}"
        finally:
            if cursor:
                cursor.close()



    def close(self):
        """Đóng kết nối tới cơ sở dữ liệu."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Đã đóng kết nối database.")
