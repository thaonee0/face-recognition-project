import mysql.connector
from mysql.connector import Error
import os
from tkinter import messagebox

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
                print("Kết nối database thành công")
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
    
    def record_attendance(self, recognized_name, recognized_prob):
        """Lưu thông tin điểm danh vào cơ sở dữ liệu."""
        if recognized_name and recognized_prob:
            # Tìm kiếm sinh viên trong cơ sở dữ liệu
            student = self.get_student_by_name(recognized_name)
            if student:
                student_id = student['id']
                
                # Lưu thông tin điểm danh vào bảng diem_danh
                try:
                    cursor = self.connection.cursor()
                    sql = "INSERT INTO diem_danh (id_sinh_vien) VALUES (%s)"
                    cursor.execute(sql, (student_id,))
                    self.connection.commit()
                    messagebox.showinfo("Điểm danh", f"Đã lưu điểm danh thành công cho {recognized_name}")
                except Error as e:
                    print(f"Lỗi khi lưu điểm danh: {e}")
            else:
                messagebox.showwarning("Không tìm thấy sinh viên", "Không có sinh viên với tên này!")
        else:
            messagebox.showerror("Lỗi", "Chưa nhận diện được sinh viên!")


    def close(self):
        if self.connection.is_connected():
            self.connection.close()