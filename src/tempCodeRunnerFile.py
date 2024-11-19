import tkinter as tk
from tkinter import Label, Button, Frame, messagebox
import cv2
from PIL import Image, ImageTk
from realtime_recognition import RealTimeRecognizer

class FaceRecognitionApp:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Xử lý ảnh - Nhóm 8")
        self.window.geometry("1200x800")

        # Trạng thái
        self.camera_on = False
        self.recognition_mode = False
        self.recognizer = RealTimeRecognizer(
            facenet_model_path="models/facenet_keras.h5",
            classifier_model_path="face_recognition_classifier.h5",
            label_names_path="label_names.npy"
        )
        
        # Cài đặt camera
        self.cap = cv2.VideoCapture(0)
        
        # Cấu hình giao diện
        self.setup_gui()

    def setup_gui(self):
        """Cài đặt giao diện chính"""
        title_label = Label(self.window, text="Hệ Thống Nhận Dạng Khuôn Mặt", font=("Helvetica", 24))
        title_label.pack(pady=10)

        # Khung hiển thị camera
        self.camera_frame = Frame(self.window, width=800, height=600, bg="black")
        self.camera_frame.pack(pady=10)
        self.camera_label = Label(self.camera_frame)
        self.camera_label.pack()

        # Khung chứa nút điều khiển
        button_frame = Frame(self.window)
        button_frame.pack(pady=20)

        # Nút bật/tắt camera
        self.toggle_camera_btn = Button(button_frame, text="Bật Camera", command=self.toggle_camera, width=20, bg="green", fg="white")
        self.toggle_camera_btn.grid(row=0, column=0, padx=10, pady=10)

        # Nút bật/tắt nhận diện
        self.recognition_btn = Button(button_frame, text="Nhận Dạng", command=self.toggle_recognition, width=20, bg="blue", fg="white")
        self.recognition_btn.grid(row=0, column=1, padx=10, pady=10)

        # Nút thoát
        self.exit_btn = Button(button_frame, text="Thoát", command=self.window.destroy, width=20, bg="red", fg="white")
        self.exit_btn.grid(row=0, column=2, padx=10, pady=10)

    def toggle_camera(self):
        if self.camera_on:
            self.camera_on = False
            self.cap.release()
            self.camera_label.config(image="")
            self.toggle_camera_btn.config(text="Bật Camera", bg="green")
        else:
            self.camera_on = True
            self.show_camera()
            self.toggle_camera_btn.config(text="Tắt Camera", bg="gray")

    def show_camera(self):
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
        if not self.camera_on:
            messagebox.showerror("Lỗi", "Hãy bật camera trước!")
            return
        
        self.recognition_mode = not self.recognition_mode
        self.recognition_btn.config(
            text="Tắt Nhận Dạng" if self.recognition_mode else "Nhận Dạng",
            bg="gray" if self.recognition_mode else "blue"
        )

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()
