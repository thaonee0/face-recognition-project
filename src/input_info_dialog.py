import tkinter as tk
from tkinter import simpledialog, messagebox
from db_handler import DatabaseHandler

class InputInfoDialog(simpledialog.Dialog):
    def __init__(self, parent, run_capture_script):
        self.parent = parent
        self.run_capture_script = run_capture_script
        self.top = tk.Toplevel(parent)
        self.top.title("Nhập Thông Tin")
        
        # Tạo labels và entries
        tk.Label(self.top, text="MSSV:").grid(row=0, column=0, padx=5, pady=5)
        self.mssv_entry = tk.Entry(self.top)
        self.mssv_entry.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(self.top, text="Tên:").grid(row=1, column=0, padx=5, pady=5)
        self.name_entry = tk.Entry(self.top)
        self.name_entry.grid(row=1, column=1, padx=5, pady=5)
        
        tk.Label(self.top, text="Lớp:").grid(row=2, column=0, padx=5, pady=5)
        self.class_entry = tk.Entry(self.top)
        self.class_entry.grid(row=2, column=1, padx=5, pady=5)
        
        tk.Label(self.top, text="Khoa:").grid(row=3, column=0, padx=5, pady=5)
        self.faculty_entry = tk.Entry(self.top)
        self.faculty_entry.grid(row=3, column=1, padx=5, pady=5)

        self.ok_button = tk.Button(self.top, text="OK", command=self.process_input)
        self.ok_button.grid(row=4, column=0, columnspan=2, pady=10)

    def process_input(self):
        mssv = self.mssv_entry.get()
        name = self.name_entry.get()
        class_name = self.class_entry.get()
        faculty = self.faculty_entry.get()
        
        if mssv and name and class_name and faculty:
            folder_name = f"{name}-{mssv}"
            # Tạo đường dẫn ảnh đại diện
            avatar_path = r"D:\FACENET\face-recognition-project\data\raw\{folder_name}\{folder_name}_1.jpg"
            
            # Lưu vào database
            db = DatabaseHandler()
            student_id = db.add_student(name, class_name, faculty, avatar_path, mssv)
            db.close()
            
            if student_id:
                self.top.destroy()
                # Chạy chức năng chụp hình với folder_name
                self.run_capture_script(folder_name, class_name)
            else:
                messagebox.showerror("Lỗi", "Không thể lưu thông tin vào database")
        else:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập đầy đủ thông tin")