import tkinter as tk
from tkinter import Label, Button, Frame, messagebox
import cv2
from PIL import Image, ImageTk
import os
import sys
import io
import subprocess
import threading

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thêm đường dẫn gốc của project vào PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Sửa lại imports để không sử dụng src.
from input_info_dialog import InputInfoDialog
from capture_and_save_face import capture_and_save_face
from realtime_recognition import start_recognition
from db_handler import DatabaseHandler  

class FaceCapturingApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Face Recognition - Nhóm 7")
        self.window.geometry("1000x800")

        # Khởi tạo DatabaseHandler
        self.db = DatabaseHandler()
        
        self.camera_on = False
        self.current_person = None
        self.image_count = 0
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.recognition_mode = False
        self.recognized_name = None
        self.recognized_prob = None
        self.recognizer = None
        
        self.setup_gui()

    def setup_gui(self):
        title_label = Label(self.window, text="Face Recognition - Nhóm 7", font=("Helvetica", 34))
        title_label.pack(pady=(60, 5))

        self.left_frame = Frame(self.window, width=800, height=580)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.left_frame.pack_propagate(False)

        self.camera_label = Label(self.left_frame, bg="black")
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        right_frame = Frame(self.window)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        #Bật Camera Button
        self.toggle_button = Button(right_frame, text="Bật Camera", 
                                    command=self.toggle_camera, 
                                    height=2, width=20, 
                                    bg="#EECAD5", fg="white",
                                    font=("Helvetica", 14, "bold"))
        self.toggle_button.pack(pady=10)

        #Nhập Thông tin Button
        self.info_button = Button(right_frame, text="Nhập Thông Tin", 
                                  command=self.input_info, 
                                  height=2, width=20, 
                                  bg="#87A2FF", fg="white",
                                  font=("Helvetica", 14, "bold"))
        self.info_button.pack(pady=10)

        #Training Button
        self.training_button = Button(right_frame, text="Huấn luyện", 
                              command=self.run_training_pipeline, 
                              height=2, width=20, 
                              bg="#705C53", fg="white",
                              font=("Helvetica", 14, "bold"))
        self.training_button.pack(pady=10)

        #Nhận diện Button
        self.recognition_button = Button(right_frame, text="Nhận diện", 
                                         command=self.toggle_recognition, 
                                         height=2, width=20,
                                         bg="#7E60BF", fg="white",
                                         font=("Helvetica", 14, "bold"))
        self.recognition_button.pack(pady=10)

        #Điểm danh Button
        self.attendance_button = Button(right_frame, text="Điểm danh", 
                                        command=self.toggle_attendance,
                                         height=2, width=20,
                                         bg="#347928", fg="white",
                                         font=("Helvetica", 14, "bold"))
        self.attendance_button.pack(pady=10)

        # Button xác nhận điểm danh (ẩn cho đến khi có kết quả)
        """self.confirm_button = Button(right_frame, text="Xác nhận", state="disabled",
                                        height=2, width=20,
                                        font=("Helvetica", 14, "bold"),
                                        command=self.confirm_attend)
        self.confirm_button.pack(pady=20)"""

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def toggle_camera(self):
        if self.camera_on:
            # Tắt camera
            self.camera_on = False
            self.cap.release()
            self.camera_label.config(image="")
            self.toggle_button.config(text="Bật Camera")
        else:
            # Bật lại camera
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Khởi tạo lại đối tượng camera
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", "Không thể mở camera. Vui lòng kiểm tra thiết bị!")
                return
            self.camera_on = True
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Giảm độ phân giải ngang
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Giảm độ phân giải dọc
            self.cap.set(cv2.CAP_PROP_FPS, 8)  # Giảm FPS 
            self.show_camera()
            self.toggle_button.config(text="Tắt Camera")

    def show_camera(self):
        if self.camera_on:
            ret, frame = self.cap.read()
            if ret:
                if self.recognition_mode and self.recognizer:
                    frame = self.recognizer.process_frame(frame)
                    
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((self.camera_label.winfo_width(), 
                                self.camera_label.winfo_height()), 
                                Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
            self.camera_label.after(10, self.show_camera)  # Gọi lại sau 10ms để tránh lag


    def input_info(self):
        dialog = InputInfoDialog(self.window, self.run_capture_process)

    def run_capture_process(self, ten, lop):
        self.current_person = ten
        if not self.camera_on:
            messagebox.showerror("Lỗi", "Vui lòng bật camera trước!")
            return
        messagebox.showinfo("Thông báo", "Bắt đầu chụp 50 ảnh tự động.")
        capture_and_save_face(self.current_person, self.toggle_camera)

    def run_training_pipeline(self, event=None):  # Thêm self để liên kết class
        try:
            # Đường dẫn tuyệt đối đến các file script
            extract_script = r"D:\FACENET\face-recognition-project\src\extract_face_dataset.py"
            embedding_script = r"D:\FACENET\face-recognition-project\src\predict_face_embedding.py"
            classification_script = r"D:\FACENET\face-recognition-project\src\classification.py"

            # Chạy lần lượt từng script
            subprocess.run(["python", extract_script], check=True)
            print("Đã chạy extract_face_dataset.py thành công")

            subprocess.run(["python", embedding_script], check=True)
            print("Đã chạy predict_face_embedding.py thành công")

            subprocess.run(["python", classification_script], check=True)
            print("Đã chạy classification.py thành công")

            # Thông báo hoàn thành
            messagebox.showinfo("Thành công", "Quá trình huấn luyện hoàn tất!")
        except subprocess.CalledProcessError as e:
            print(f"Lỗi khi chạy file: {e}")
            messagebox.showerror("Lỗi", f"Quá trình huấn luyện thất bại.\nChi tiết lỗi: {e}")

    def toggle_recognition(self):
        if not self.camera_on:
            messagebox.showerror("Lỗi", "Vui lòng bật camera trước!")
            return
            
        """Bắt đầu nhận diện khuôn mặt."""
        def on_recognition(name, prob):
            """Xử lý kết quả nhận diện khuôn mặt."""
            self.window.after(0, self.update_recognition_result, name, prob)
        
        def recognition_thread():
            start_recognition(on_recognition)

        # Chạy nhận diện trong một thread riêng biệt
        threading.Thread(target=recognition_thread, daemon=True).start()

    def update_recognition_result(self, name, prob):
        """Cập nhật kết quả nhận diện trên GUI"""
        self.recognized_name = name
        self.recognized_prob = prob
        if prob > 70:
            self.attendance_button.config(state="normal")  # Kích hoạt button điểm danh
        else:
            self.attendance_button.config(state="disabled")

    def toggle_attendance(self): 
        if self.recognized_name:
            # Tách recognized_name thành ten_sinh_vien và mssv
            try:
                ten_sinh_vien, mssv = self.recognized_name.split('-')  # Tách theo dấu '-'
                full_name_with_mssv = f"{ten_sinh_vien}-{mssv}"  # Tạo lại chuỗi đầy đủ
            except ValueError:
                messagebox.showerror("Lỗi", "Tên sinh viên hoặc MSSV không hợp lệ!")
                return
            
            messagebox.showinfo("Điểm danh", f"{full_name_with_mssv}")
            success, message = self.db.check_attendance(full_name_with_mssv)
            if success:
                messagebox.showinfo("Thông báo", message)
            else:
                messagebox.showerror("Thông báo", message)
        else:
            messagebox.showerror("Lỗi", "Chưa nhận diện được khuôn mặt!")

    
    def run(self):
        self.window.mainloop()

    def __del__(self):
        if self.camera_on:
            self.cap.release()
        # Đóng kết nối database
        self.db.close()

if __name__ == "__main__":
    app = FaceCapturingApp()
    app.run()

"""    def confirm_attend(self):
        if self.recognized_name:
            success, message = self.db.check_attendance(self.recognized_name)
            # Ghi nhận vào cơ sở dữ liệu hoặc lưu lại thông tin người điểm danh
            if success:
                messagebox.showinfo("Thông báo", message)
            else:
                messagebox.showerror("Thông báo", message)
        else:
            messagebox.showerror("Lỗi", "Chưa nhận diện được khuôn mặt!")"""