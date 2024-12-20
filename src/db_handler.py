import mysql.connector
from mysql.connector import Error
import os

class DatabaseHandler:
    def __init__(self):
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

    def add_student(self, ten_sinh_vien, lop, khoa, anh_dai_dien):
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

    def add_attendance(self, id_sinh_vien):
        try:
            cursor = self.connection.cursor()
            sql = "INSERT INTO diem_danh (id_sinh_vien) VALUES (%s)"
            cursor.execute(sql, (id_sinh_vien,))
            self.connection.commit()
        except Error as e:
            print(f"Lỗi thêm điểm danh: {e}")
        finally:
            if cursor:
                cursor.close()

    def get_student_by_name(self, name):
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

    def close(self):
        if self.connection.is_connected():
            self.connection.close()