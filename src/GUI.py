import tkinter as tk
from tkinter import Label, Button, Frame, messagebox
import cv2
from PIL import Image, ImageTk
import os
import sys
import subprocess

# Thêm đường dẫn gốc của project vào PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Sửa lại imports để không sử dụng src.
from input_info_dialog import InputInfoDialog
from capture_and_save_face import capture_and_save_face
from realtime_recognition import start_recognition

class FaceCapturingApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Xử lý ảnh - Nhóm 8")
        self.window.geometry("1000x800")
        
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
        title_label = Label(self.window, text="Xử Lý Ảnh - Nhóm 8", font=("Helvetica", 34))
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
                                    bg="#4CAF50", fg="white",
                                    font=("Helvetica", 14, "bold"))
        self.toggle_button.pack(pady=10)

        #Nhập Thông tin Button
        self.info_button = Button(right_frame, text="Nhập Thông Tin", 
                                  command=self.input_info, 
                                  height=2, width=20, 
                                  bg="#FFC107", fg="white",
                                  font=("Helvetica", 14, "bold"))
        self.info_button.pack(pady=10)

        #Training Button
        self.training_button = Button(right_frame, text="Training", 
                                     command=self.run_training_pipeline, 
                                     height=2, width=20, 
                                     bg="#FFC107", fg="white",
                                     font=("Helvetica", 14, "bold"))
        self.training_button.pack(pady=10)

        #Nhận diện Button
        self.recognition_button = Button(right_frame, text="Nhận diện", 
                                         command=self.toggle_recognition, 
                                         height=2, width=20,
                                         bg="#9C27B0", fg="white",
                                         font=("Helvetica", 14, "bold"))
        self.recognition_button.pack(pady=10)

        #Điểm danh Button
        self.attendance_button = Button(right_frame, text="Điểm danh", 
                                         command=self.toggle_attendance, 
                                         height=2, width=20,
                                         bg="#9C27B0", fg="white",
                                         font=("Helvetica", 14, "bold"))
        self.attendance_button.pack(pady=10)

        # Button xác nhận điểm danh (ẩn cho đến khi có kết quả)
        self.confirm_button = tk.Button(self.window, text="Xác nhận", state="disabled", 
                                           command=self.confirm_attend)
        self.confirm_button.pack(pady=20)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    def toggle_camera(self):
        if self.camera_on:
            self.camera_on = False
            self.cap.release()
            self.camera_label.config(image="")
            self.toggle_button.config(text="Bật Camera")
        else:
            self.camera_on = True
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Giảm độ phân giải ngang
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Giảm độ phân giải dọc
            self.cap.set(cv2.CAP_PROP_FPS, 10)  # Giảm FPS xuống 15
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
        self.setup_capture_directories()
        capture_and_save_face(self.current_person, self.toggle_camera)

    def setup_capture_directories(self):
        train_dir = os.path.join(self.base_path, 'data', 'raw', 'train', self.current_person)
        
        os.makedirs(train_dir, exist_ok=True)
        
        return train_dir
    
    def run_training_pipeline():
        try:
            # Chạy file extract_face_dataset.py
            subprocess.run(["python", "extract_face_dataset.py"], check=True)
            print("Đã lấy dataset người dùng")

            # Chạy file predict_face_embedding.py
            subprocess.run(["python", "predict_face_embedding.py"], check=True)
            print("Đã trích xuất embedding thành công")

            # Chạy file classification.py
            subprocess.run(["python", "classification.py"], check=True)
            print("Đã hoàn thành phân loại")

            # Thông báo thành công
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
            self.recognized_name = name
            self.recognized_prob = prob
            if prob > 70:
                self.attendance_button.config(state="normal")  # Kích hoạt button điểm danh
            else:
                self.attendance_button.config(state="disabled")

        # Gọi hàm nhận diện từ file realtime_recognition.py
        start_recognition(on_recognition)

    def toggle_attendance(self):
        if self.recognized_name:
            messagebox.showinfo("Điểm danh thành công", f"{self.recognized_name}")
            self.confirm_button.config(state="normal") 
        else:
            messagebox.showerror("Lỗi", "Chưa nhận diện được khuôn mặt!")

    def confirm_attend(self):
        if self.recognized_name:
            # Ghi nhận vào cơ sở dữ liệu hoặc lưu lại thông tin người điểm danh
            print(f"{self.recognized_name} đã được điểm danh.")
            # Thực hiện lưu vào cơ sở dữ liệu ở đây
            messagebox.showinfo("Thông báo", f"{self.recognized_name} đã điểm danh thành công")
    
    def run(self):
        self.window.mainloop()

    def __del__(self):
        if self.camera_on:
            self.cap.release()

if __name__ == "__main__":
    app = FaceCapturingApp()
    app.run()
