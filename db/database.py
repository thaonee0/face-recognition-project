import mysql.connector
from mysql.connector import Error

class Database:
    def __init__(self, host, database, user, password):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.create_connection()

    def create_connection(self):
        """Tạo kết nối đến cơ sở dữ liệu MySQL."""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password
            )
            if self.connection.is_connected():
                print("Đã kết nối thành công đến cơ sở dữ liệu")
        except Error as e:
            print(f"Lỗi khi kết nối đến cơ sở dữ liệu: {e}")

    def create_table(self):
        """Tạo bảng nếu chưa tồn tại."""
        create_table_query = """
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            student_id VARCHAR(20) NOT NULL,
            student_name VARCHAR(100) NOT NULL,
            class_number VARCHAR(10) NOT NULL,
            mssv VARCHAR(20) NOT NULL,
            date DATE NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        cursor = self.connection.cursor()
        cursor.execute(create_table_query)
        self.connection.commit()
        print("Bảng 'attendance' đã được tạo hoặc đã tồn tại.")

    def insert_attendance(self, student_id, student_name, class_number, mssv, date):
        """Thêm thông tin điểm danh vào bảng attendance."""
        insert_query = """
        INSERT INTO attendance (student_id, student_name, class_number, mssv, date)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor = self.connection.cursor()
        cursor.execute(insert_query, (student_id, student_name, class_number, mssv, date))
        self.connection.commit()
        print("Thông tin điểm danh đã được thêm thành công.")

    def fetch_attendance(self):
        """Lấy thông tin điểm danh từ bảng attendance."""
        select_query = "SELECT * FROM attendance"
        cursor = self.connection.cursor()
        cursor.execute(select_query)
        records = cursor.fetchall()
        print("Thông tin điểm danh:")
        for row in records:
            print(row)

    def close_connection(self):
        """Đóng kết nối đến cơ sở dữ liệu."""
        if self.connection.is_connected():
            self.connection.close()
            print("Kết nối đến cơ sở dữ liệu đã được đóng.")

# Ví dụ sử dụng
if __name__ == "__main__":
    db = Database(host="localhost", database="your_database_name", user="your_username", password="your_password")
    db.create_table()
    # Ghi điểm danh cho sinh viên
    db.insert_attendance("SV001", "Nguyễn Văn A", " lớp 1", "MSSV001", "2024-10-13")
    db.insert_attendance("SV002", "Trần Thị B", " lớp 1", "MSSV002", "2024-10-13")
    # Lấy và hiển thị thông tin điểm danh
    db.fetch_attendance()
    db.close_connection()
