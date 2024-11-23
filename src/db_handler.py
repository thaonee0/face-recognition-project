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
                print("Kết nối database thành công")
        except Error as e:
            print(f"Lỗi kết nối database: {e}")

    def reconnect(self):
        """Tự động kết nối lại nếu kết nối bị mất."""
        if not self.connection or not self.connection.is_connected():
            print("Kết nối cơ sở dữ liệu đã mất, thử kết nối lại...")
            self.connect_to_database()

    def add_student(self, ten_sinh_vien, lop, khoa, anh_dai_dien):
        """Thêm sinh viên mới vào cơ sở dữ liệu."""
        self.reconnect()
        try:
            cursor = self.connection.cursor()
            sql = """INSERT INTO sinhvien (ten_sinh_vien, lop, khoa, anh_dai_dien) 
                     VALUES (%s, %s, %s, %s)"""
            values = (ten_sinh_vien, lop, khoa, anh_dai_dien)
            cursor.execute(sql, values)
            self.connection.commit()
            return cursor.lastrowid
        except Error as e:
            print(f"Lỗi thêm sinh viên: {e}")
            return None
        finally:
            if cursor:
                cursor.close()

    def get_student_by_name(self, name):
        """Lấy thông tin sinh viên dựa vào tên."""
        self.reconnect()
        try:
            cursor = self.connection.cursor(dictionary=True)
            sql = "SELECT * FROM sinhvien WHERE ten_sinh_vien = %s"
            cursor.execute(sql, (name,))
            return cursor.fetchone()
        except Error as e:
            print(f"Lỗi truy vấn sinh viên: {e}")
            return None
        finally:
            if cursor:
                cursor.close()

    def check_attendance(self, ten_sinh_vien):
        """Kiểm tra và thực hiện điểm danh."""
        self.reconnect()
        try:
            cursor = self.connection.cursor(dictionary=True)

            print(f"Đang kiểm tra điểm danh cho sinh viên: {ten_sinh_vien}")
            
            # Lấy thông tin sinh viên từ bảng sinhvien
            cursor.execute("SELECT * FROM sinhvien WHERE ten_sinh_vien = %s", (ten_sinh_vien,))
            student = cursor.fetchone()
            
            if not student:
                print(f"Không tìm thấy sinh viên {ten_sinh_vien} trong cơ sở dữ liệu.")
                return False, f"Không tìm thấy sinh viên {ten_sinh_vien} trong cơ sở dữ liệu."
            
            print(f"Thông tin sinh viên: {student}")
            
            # Kiểm tra đã điểm danh hôm nay chưa
            today = date.today().strftime('%Y-%m-%d')
            cursor.execute("""
                SELECT * FROM diem_danh 
                WHERE ten_sinh_vien = %s AND DATE(ngay_gio_diem_danh) = %s
            """, (student['ten_sinh_vien'], today))
            
            existing_attendance = cursor.fetchone()
            if existing_attendance:
                print(f"Sinh viên {ten_sinh_vien} đã điểm danh hôm nay.")
                return False, f"Sinh viên {ten_sinh_vien} đã điểm danh hôm nay."
            
            # Thêm điểm danh mới
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Thêm điểm danh cho sinh viên {student['ten_sinh_vien']} vào lúc {now}")
            cursor.execute("""
                INSERT INTO diem_danh (ten_sinh_vien, ngay_gio_diem_danh)
                VALUES (%s, %s)
            """, (student['ten_sinh_vien'], now))
            
            self.connection.commit()
            
            success_message = f"""Điểm danh thành công:
            - Tên: {student['ten_sinh_vien']}
            - MSSV: {student.get('mssv', 'N/A')}
            - Lớp: {student['lop']}
            - Thời gian: {now}"""
            
            print(success_message)
            return True, success_message
                
        except Error as e:
            print(f"Lỗi khi điểm danh: {e}")
            return False, f"Lỗi khi điểm danh: {e}"
        finally:
            if cursor:
                cursor.close()

    def close(self):
        """Đóng kết nối tới cơ sở dữ liệu."""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("Đã đóng kết nối database.")
