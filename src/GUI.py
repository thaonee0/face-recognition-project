import tkinter as tk
from tkinter import Label, Button, Frame
import cv2
from PIL import Image, ImageTk

# Biến kiểm soát camera
camera_on = False

# Hàm để hiển thị camera
def show_camera():
    if camera_on:
        ret, frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            # Đặt kích thước cho hình ảnh để khớp với kích thước của camera_label
            img = img.resize((camera_label.winfo_width(), camera_label.winfo_height()), Image.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)
            camera_label.imgtk = imgtk
            camera_label.configure(image=imgtk)
        camera_label.after(10, show_camera)

# Hàm để bật/tắt camera
def toggle_camera():
    global camera_on
    if camera_on:
        camera_on = False
        cap.release()
        camera_label.config(image="")
        toggle_button.config(text="Bật Camera")
    else:
        camera_on = True
        cap.open(0)
        show_camera()
        toggle_button.config(text="Tắt Camera")

# Hàm để chụp hình từ camera
def capture_image():
    if camera_on:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite("captured_image.png", frame)
            print("Hình đã được chụp và lưu.")
    else:
        print("Camera chưa được bật.")

# Hàm để trích xuất dữ liệu
def extract():
    print("Trích xuất dữ liệu...")

# Hàm để dự đoán
def predict():
    print("Dự đoán...")

# Hàm để so sánh trích xuất với dự đoán
def compare():
    print("So sánh trích xuất với dự đoán...")

# Tạo cửa sổ Tkinter
window = tk.Tk()
window.title("Xử lý ảnh - Nhóm 8")
window.geometry("1000x800")

# Tiêu đề 
title_label = Label(window, text="Xử Lý Ảnh - Nhóm 8", font=("Helvetica", 34))
title_label.pack(pady=(60, 5))

# Frame để chứa camera ở góc trái
left_frame = Frame(window, width=800, height=580)  # Thiết lập kích thước cố định cho frame
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Không cho phép frame thay đổi kích thước theo nội dung
left_frame.pack_propagate(False)

# Label để hiển thị camera
camera_label = Label(left_frame, bg="black")  # Đặt nền đen cho label
camera_label.pack(fill=tk.BOTH, expand=True)  # Label sẽ chiếm toàn bộ khung chứa

# Frame để chứa các nút ở góc phải
right_frame = Frame(window)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# Nút bật/tắt camera
toggle_button = Button(right_frame, text="Bật Camera", command=toggle_camera, height=2, width=20, bg="#4CAF50", fg="white",font=("Helvetica", 14, "bold"))
toggle_button.pack(pady=10)

# Nút ấn để chụp hình
capture_button = Button(right_frame, text="Chụp Hình", command=capture_image, height=2, width=20, bg="#2196F3", fg="white",font=("Helvetica", 14, "bold"))
capture_button.pack(pady=10)

# Nút trích xuất
extract_button = Button(right_frame, text="Trích Xuất", command=extract, height=2, width=20, bg="#FFC107", fg="white",font=("Helvetica", 14, "bold"))
extract_button.pack(pady=10)

# Nút dự đoán
predict_button = Button(right_frame, text="Dự Đoán", command=predict, height=2, width=20, bg="#9C27B0", fg="white",font=("Helvetica", 14, "bold"))
predict_button.pack(pady=10)

# Nút so sánh trích xuất với dự đoán
compare_button = Button(right_frame, text="So Sánh", command=compare, height=2, width=20, bg="#F44336", fg="white",font=("Helvetica", 14, "bold"))
compare_button.pack(pady=10)

# Khởi tạo camera nhưng chưa mở
cap = cv2.VideoCapture()

# Khởi động giao diện Tkinter
window.mainloop()

# Giải phóng camera sau khi đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
