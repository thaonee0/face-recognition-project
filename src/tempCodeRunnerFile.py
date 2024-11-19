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

# Import các module liên quan
from input_info_dialog import InputInfoDialog
from capture_and_save_face import capture_and_save_face
from train_facenet import FaceNetAutoUpdater
from realtime_recognition import RealTimeRecognizer

class FaceRecognitionApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Xử lý ảnh - Nhóm 8")
        self.window.geometry("1000x800")
        
        self.camera_on = False
        self.current_person = None
        self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.recognition_mode = False
        self.recognizer = None
        
        # Khởi tạo các đường dẫn cho model
        self.facenet_model_path = os.path.join(project_root, 'models', 'facenet_keras.h5')
        self.classifier_model_path = os.path.join(project_root, 'face_recognition_classifier.h5')
        self.label_names_path = os.path.join(project_root, 'label_names.npy')
        self.train_dir = os.path.join(project_root, 'data', 'raw', 'train')
        
        # Cấu hình MySQL
        self.db_config = {
            'user': 'root',
            'password': '',
            'host': 'localhost',
            'database': 'diem_danh'
        }
        
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

        # Nút Bật Camera
        self.toggle_button = Button(right_frame, text="Bật Camera", 
                                    command=self.toggle_camera, 
                                    height=2, width=20, 
                                    bg="#4CAF50", fg="white",
                                    font=("Helvetica", 14, "bold"))
        self.toggle_button.pack(pady=10)

        # Nút Nhập Thông Tin
        self.info_button = Button(right_frame, text="Nhập Thông Tin", 
                                  command=self.input_info, 
                                  height=2, width=20, 
                                  bg="#FFC107", fg="white",
                                  font=("Helvetica", 14, "bold"))
        self.info_button.pack(pady=10)

        # Nút Huấn Luyện
        self.train_button = Button(right_frame, text="Huấn Luyện", 
                                   command=self.train_model,
                                   height=2, width=20, 
                                   bg="#FF5722", fg="white",
                                   font=("Helvetica", 14, "bold"))
        self.train_button.pack(pady=10)

        # Nút Nhận Dạng
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

    def train_model(self):
        try:
            # Tạo instance của FaceNetAutoUpdater
            updater = FaceNetAutoUpdater(
                facenet_model_path=self.facenet_model_path,
                existing_classifier_path=self.classifier_model_path,
                train_dir=self.train_dir,
                output_model_path=self.classifier_model_path
            )
            
            # Thực hiện quá trình huấn luyện
            new_faces, new_labels, num_classes = updater.prepare_new_data()
            
            if new_faces is not None:
                updater.build_updated_classifier(num_classes)
                updater.train(new_faces, new_labels)
                updater.save_model()
                messagebox.showinfo("Thành công", "Đã huấn luyện mô hình thành công!")
            else:
                messagebox.showinfo("Thông báo", "Không có dữ liệu mới để huấn luyện!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi huấn luyện mô hình: {str(e)}")

    def toggle_recognition(self):
        if not self.camera_on:
            messagebox.showerror("Lỗi", "Vui lòng bật camera trước!")
            return

        self.recognition_mode = not self.recognition_mode
        if self.recognition_mode:
            try:
                # Đóng cửa sổ GUI hiện tại
                self.window.quit()
                self.window.destroy()

                # Mở cửa sổ nhận diện khuôn mặt mới sau khi đóng GUI
                subprocess.run([sys.executable, r'D:\FACENET\face-recognition-project\src\realtime_recognition.py'])
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể khởi tạo nhận dạng: {str(e)}")
                self.recognition_mode = False
                return
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

    def run(self):
        self.window.mainloop()

    def __del__(self):
        if self.camera_on:
            self.cap.release()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()
