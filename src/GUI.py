import tkinter as tk
from tkinter import messagebox
from tkinter import Frame, Label, Button
import cv2
from PIL import Image, ImageTk
from realtime_recognition import RealTimeRecognizer
from sklearn.preprocessing import LabelEncoder
import subprocess
import os
from input_info_dialog import InputInfoDialog  # Import InputInfoDialog
import capture_and_save_face  # Import capture_and_save_face.py

class FaceRecognitionApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("NHẬN DIỆN KHUÔN MẶT")
        self.window.geometry("1200x800")

        self.camera_on = False
        self.recognition_mode = False
        
        # Cấu hình kết nối MySQL
        self.db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '',
            'database': 'diem_danh'
        }

        # Khởi tạo RealTimeRecognizer
        self.recognizer = RealTimeRecognizer(
            facenet_model_path="models/facenet_keras.h5",
            classifier_model_path="face_recognition_classifier.h5",
            label_names_path="label_names.npy",
            db_config=self.db_config
        )

        # Cài đặt camera
        self.cap = cv2.VideoCapture(0)

        # Cài đặt giao diện
        self.setup_gui()

    def setup_gui(self):
        """Cài đặt giao diện chính"""
        title_label = tk.Label(self.window, text="Hệ Thống Nhận Dạng Khuôn Mặt", font=("Helvetica", 24))
        title_label.pack(pady=10)

        # Khung hiển thị camera
        self.camera_frame = Frame(self.window, width=800, height=600, bg="black")
        self.camera_frame.pack(pady=10)
        self.camera_label = Label(self.camera_frame)
        self.camera_label.pack()

        # Khung chứa các nút điều khiển
        button_frame = Frame(self.window)
        button_frame.pack(side=tk.BOTTOM, fill='x', pady=50)  # Nút ở phía dưới cùng với khoảng cách 10px

        # Đặt các nút vào button_frame và sắp xếp theo hàng ngang
        self.toggle_camera_btn = Button(button_frame, text="BẬT CAMERA", command=self.toggle_camera, width=20, height=4, bg="green", fg="white")
        self.toggle_camera_btn.pack(side=tk.LEFT, padx=5)

        self.recognition_btn = Button(button_frame, text="NHẬN DẠNG", command=self.toggle_recognition, width=20, height=4, bg="blue", fg="white")
        self.recognition_btn.pack(side=tk.LEFT, padx=5)

        self.input_info_btn = Button(button_frame, text="NHẬP THÔNG TIN", command=self.open_input_info_dialog, width=20, height=4, bg="orange", fg="white")
        self.input_info_btn.pack(side=tk.LEFT, padx=5)

        self.train_btn = Button(button_frame, text="HUẤN LUYỆN", command=self.train_model, width=20, height=4, bg="purple", fg="white")
        self.train_btn.pack(side=tk.LEFT, padx=5)

        self.exit_btn = Button(button_frame, text="THOÁT", command=self.window.destroy, width=20, height=4, bg="red", fg="white")
        self.exit_btn.pack(side=tk.LEFT, padx=5)

        button_frame.place(relx=0.5, rely=0.9, anchor="center") 

    def open_input_info_dialog(self):
        """Mở cửa sổ nhập thông tin sinh viên"""
        def run_capture_script(student_id, student_name, class_name, department, image_dir):
            # Chạy script capture ảnh sau khi nhập thông tin
            capture_and_save_face.capture_face(student_id, student_name, class_name, department, image_dir)
        
        # Truyền hàm run_capture_script vào InputInfoDialog
        dialog = InputInfoDialog(self.window, run_capture_script)
        dialog.run()  # Giả sử InputInfoDialog có phương thức run để hiển thị cửa sổ

    def toggle_camera(self):
        """Bật/Tắt camera"""
        if self.camera_on:
            self.camera_on = False
            self.cap.release()
            self.camera_label.config(image="")
            self.toggle_camera_btn.config(text="BẬT CAMERA", bg="green")
        else:
            self.camera_on = True
            self.show_camera()
            self.toggle_camera_btn.config(text="TẮT CAMERA", bg="gray")

    def show_camera(self):
        """Hiển thị camera và nhận diện khuôn mặt nếu cần"""
        if self.camera_on:
            ret, frame = self.cap.read()
            if ret:
                if self.recognition_mode:
                    recognition_results = self.recognizer.recognize_frame(frame)
                    for res in recognition_results:
                        x, y, w, h = res['box']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, f"{res['name']} ({res['mssv']})", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = img
                self.camera_label.configure(image=img)

            self.camera_label.after(10, self.show_camera)

    def toggle_recognition(self):
        """Bật/Tắt chế độ nhận diện khuôn mặt"""
        if not self.camera_on:
            messagebox.showerror("Lỗi", "Hãy bật camera trước!")
            return

        self.recognition_mode = not self.recognition_mode
        self.recognition_btn.config(
            text="TẮT NHẬN DẠNG" if self.recognition_mode else "NHẬN DẠNG",
            bg="gray" if self.recognition_mode else "blue"
        )

    def train_model(self):
        """Chạy file huấn luyện mô hình"""
        try:
            # Chạy file huấn luyện mô hình (train_facenet.py)
            subprocess.run(["python", "D:/FACENET/face-recognition-project/src/train_facenet.py"], check=True)
            messagebox.showinfo("Thông báo", "Huấn luyện mô hình thành công!")
        except subprocess.CalledProcessError:
            messagebox.showerror("Lỗi", "Không thể huấn luyện mô hình.")

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()
