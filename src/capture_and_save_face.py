import cv2
import os

# Danh sách các celebrity hợp lệ
valid_celebrities = ['celebrity_1', 'celebrity_2']  # Thay đổi theo danh sách thật của bạn

def capture_and_save_face(name, save_dir='data/raw/val/'):
    # Kiểm tra xem tên celebrity có hợp lệ không
    if name not in valid_celebrities:
        print("Tên celebrity không hợp lệ! Vui lòng nhập lại.")
        return

    # Tạo đường dẫn lưu hình ảnh
    save_path = os.path.join(save_dir, name)

    # Kiểm tra xem thư mục có tồn tại không, nếu không thì tạo
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Khởi động webcam
    cap = cv2.VideoCapture(0)

    # Kiểm tra xem webcam có mở thành công không
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Chụp hình, nhấn 'c' để lưu, nhấn 'q' để thoát.")
    
    while True:
        ret, frame = cap.read()  # Đọc khung hình từ webcam
        if not ret:
            print("Error: Could not read frame.")
            break

        # Hiển thị hình ảnh
        cv2.imshow('Capture Face', frame)

        # Nhấn 'c' để chụp và lưu hình ảnh, nhấn 'q' để thoát
        key = cv2.waitKey(1)
        if key == ord('c'):
            # Tạo tên file hình ảnh
            file_name = f"{name}.jpg"
            # Lưu hình ảnh
            cv2.imwrite(os.path.join(save_path, file_name), frame)
            print(f"Hình ảnh đã được lưu: {file_name}")
        elif key == ord('q'):
            break

    # Giải phóng webcam và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_name = input("Nhập tên celebrity để lưu hình ảnh: ")
    capture_and_save_face(user_name)
