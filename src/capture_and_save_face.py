import cv2
import os
from tkinter import messagebox

def capture_and_save_face(name, camera_control_callback):
    train_dir = f'D:\\FACENET\\face_recognition_project\\data\\raw\\{name}'
    
    os.makedirs(train_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)  # Mở camera
    count = 0

    while count < 50:  # Chụp 50 hình ảnh
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Lỗi", "Không thể đọc từ camera.")
            break

        # Lưu hình ảnh vào thư mục train và val
        img_path_train = os.path.join(train_dir, f"{name}_{count + 1}.jpg")
        
        cv2.imwrite(img_path_train, frame)  # Lưu hình ảnh vào train
        print(f"Hình ảnh đã được lưu: {img_path_train}")

        count += 1

    cap.release()  # Giải phóng camera
    camera_control_callback()  # Gọi hàm tắt camera
    messagebox.showinfo("Thành công", "Đã chụp và lưu 50 hình ảnh thành công.")