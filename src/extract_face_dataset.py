import os
import sys
import io
import numpy as np
from os import listdir
from os.path import join, isdir
from PIL import Image
from numpy import asarray, savez_compressed
from mtcnn.mtcnn import MTCNN

# Đảm bảo output của hệ thống sử dụng encoding utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import tensorflow as tf

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)

    if len(results) == 0:
        return None

    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# Tải tất cả khuôn mặt từ thư mục
def load_faces(directory):
    faces = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Đường dẫn đầy đủ
            path = os.path.join(directory, filename)
            # Trích xuất khuôn mặt
            face = extract_face(path)
            # Lưu khuôn mặt nếu có
            if face is not None:
                faces.append(face)
    return faces

def load_and_save_dataset(input_folder, output_folder, index_file):
    index_data = []

    # Đọc tệp index.npz nếu đã tồn tại
    index_file_path = join(output_folder, index_file)
    if os.path.exists(index_file_path):
        index_data = list(np.load(index_file_path, allow_pickle=True)['index_data'])

    # Kiểm tra và loại bỏ thông tin lỗi thời trong index_data
    updated_index_data = []
    for entry in index_data:
        if os.path.exists(entry['file_path']):
            updated_index_data.append(entry)
        else:
            print(f"Tệp {entry['file_path']} không tồn tại. Đã loại bỏ khỏi index.")

    # Lưu lại index.npz đã cập nhật (loại bỏ thông tin lỗi thời)
    savez_compressed(index_file_path, index_data=updated_index_data)
    print(f"Đã cập nhật index.npz với thông tin chính xác.")

    # Cập nhật index_data sau khi lọc
    index_data = updated_index_data

    # Kiểm tra nếu tất cả dataset đã tồn tại
    all_files_exist = True
    for subdir in os.listdir(input_folder):
        path = join(input_folder, subdir)
        if not isdir(path):
            continue

        output_file = join(output_folder, f"{subdir}_faces.npz")
        if not os.path.exists(output_file):
            all_files_exist = False
            break  # Không cần kiểm tra thêm nếu phát hiện tệp thiếu

    if all_files_exist:
        print("Tất cả tệp dataset đã tồn tại, không cần xử lý thêm.")
        return False  # Dừng chương trình nếu tất cả dữ liệu đã tồn tại

    # Xử lý các thư mục con (lớp dữ liệu)
    for subdir in os.listdir(input_folder):
        path = join(input_folder, subdir)
        if not isdir(path):
            continue

        # Tạo đường dẫn tệp kết quả cho từng lớp
        output_file = join(output_folder, f"{subdir}_faces.npz")

        # Kiểm tra nếu tệp dataset đã tồn tại
        if os.path.exists(output_file):
            print(f"Tệp {output_file} đã tồn tại.")
            continue

        # Tải tất cả khuôn mặt trong thư mục con
        subdir_faces = load_faces(path)
        if len(subdir_faces) == 0:
            print(f"Không có hình ảnh khuôn mặt trong thư mục {subdir}.")
            continue

        subdir_labels = [subdir] * len(subdir_faces)

        faces_array = asarray(subdir_faces)
        labels_array = asarray(subdir_labels)

        # Lưu khuôn mặt và nhãn vào tệp .npz
        savez_compressed(output_file, faces=faces_array, labels=labels_array)
        print(f"> Đã lưu {len(subdir_faces)} ví dụ cho lớp: {subdir} vào tệp {output_file}.")

        # Thêm thông tin vào index_data
        index_data.append({
            'name': subdir,
            'num_images': len(faces_array),
            'file_path': output_file
        })

    # Lưu lại file index.npz với dữ liệu mới
    savez_compressed(index_file_path, index_data=index_data)
    print(f"Tạo tệp index: {index_file_path}")

def main():
    # Đường dẫn tập huấn luyện
    train_folder = r'D:\FACENET\face-recognition-project\data\raw'
    # Đường dẫn lưu file đầu ra
    output_folder = r'D:\FACENET\face-recognition-project\data\processed\face_dataset'
    index_file = 'index.npz'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Kiểm tra thư mục đầu vào
    print(f"Thư mục train_folder: {os.listdir(train_folder)}")

    # Tải tập dữ liệu huấn luyện
    print("Đang tải và xử lý tập huấn luyện...") 
    load_and_save_dataset(train_folder, output_folder, index_file)

if __name__ == '__main__':
    main()
