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
from resnet50v2_model import ResNet50V2
from input_info_dialog import InputInfoDialog
from capture_and_save_face import capture_and_save_face
from realtime_recognition import RealtimeRecognition

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

        self.toggle_button = Button(right_frame, text="Bật Camera", 
                                    command=self.toggle_camera, 
                                    height=2, width=20, 
                                    bg="#4CAF50", fg="white",
                                    font=("Helvetica", 14, "bold"))
        self.toggle_button.pack(pady=10)

        self.info_button = Button(right_frame, text="Nhập Thông Tin", 
                                  command=self.input_info, 
                                  height=2, width=20, 
                                  bg="#FFC107", fg="white",
                                  font=("Helvetica", 14, "bold"))
        self.info_button.pack(pady=10)

        self.extract_button = Button(right_frame, text="Trích Xuất", 
                                     command=self.extract_faces, 
                                     height=2, width=20, 
                                     bg="#FFC107", fg="white",
                                     font=("Helvetica", 14, "bold"))
        self.extract_button.pack(pady=10)

        self.predict_button = Button(right_frame, text="Dự Đoán", 
                                     command=self.predict_faces, 
                                     height=2, width=20,
        bg="#9C27B0", fg="white",
                                     font=("Helvetica", 14, "bold"))
        self.predict_button.pack(pady=10)

        self.compare_button = Button(right_frame, text="So Sánh", 
                                     command=self.compare_faces, 
                                     height=2, width=20, 
                                     bg="#F44336", fg="white",
                                     font=("Helvetica", 14, "bold"))
        self.compare_button.pack(pady=10)

        self.recognition_button = Button(right_frame, text="Nhận Dạng", 
                                         command=self.toggle_recognition, 
                                         height=2, width=20, 
                                         bg="#E91E63", fg="white",
                                         font=("Helvetica", 14, "bold"))
        self.recognition_button.pack(pady=10)

        self.cap = cv2.VideoCapture(0)

    def toggle_camera(self):
        if self.camera_on:
            self.camera_on = False
            self.cap.release()
            self.camera_label.config(image="")
            self.toggle_button.config(text="Bật Camera")
        else:
            self.camera_on = True
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
            self.camera_label.after(10, self.show_camera)

    def toggle_recognition(self):
        if not self.camera_on:
            messagebox.showerror("Lỗi", "Vui lòng bật camera trước!")
            return
            
        self.recognition_mode = not self.recognition_mode
        if self.recognition_mode:
            self.recognizer = RealtimeRecognition()
            self.recognition_button.config(text="Tắt Nhận Dạng")
        else:
            self.recognizer = None
            self.recognition_button.config(text="Nhận Dạng")

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
        val_dir = os.path.join(self.base_path, 'data', 'raw', 'val', self.current_person)
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        return train_dir, val_dir

    def extract_faces(self):
        try:
            extract_script = os.path.join(self.base_path, "src", "extract_faces_dataset.py")
            subprocess.run(['python', extract_script], check=True)
            messagebox.showinfo("Thành công", "Đã trích xuất khuôn mặt thành công!")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Lỗi", f"Lỗi khi trích xuất khuôn mặt: {str(e)}")

    def predict_faces(self):
        try:
            predict_script = os.path.join(self.base_path, "src", "predict_face_embeddings.py")
            subprocess.run(['python', predict_script], check=True)
            messagebox.showinfo("Thành công", "Đã dự đoán khuôn mặt thành công!")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Lỗi", f"Lỗi khi dự đoán khuôn mặt: {str(e)}")

    def compare_faces(self):
        try:
            compare_script = os.path.join(self.base_path, "src", "realtime_recognition.py")
            subprocess.run(['python', compare_script], check=True)
            messagebox.showinfo("Thành công", "Đã so sánh khuôn mặt thành công!")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Lỗi", f"Lỗi khi so sánh khuôn mặt: {str(e)}")

    def run(self):
        self.window.mainloop()

    def __del__(self):
        if self.camera_on:
            self.cap.release()

if __name__ == "__main__":
    app = FaceCapturingApp()
    app.run()