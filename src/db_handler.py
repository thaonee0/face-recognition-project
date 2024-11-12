from os import listdir
from os.path import isdir, join
from PIL import Image
from numpy import savez_compressed, asarray
from mtcnn.mtcnn import MTCNN
import argparse
import tensorflow as tf

# Cấu hình GPU (nếu có)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Trích xuất khuôn mặt từ ảnh
def extract_face(filename, required_size=(224, 224 , 3)):  # Thay đổi kích thước ở đây
    image = Image.open(filename).convert('RGB')
    pixels = asarray(image)

    # Phát hiện khuôn mặt với MTCNN
    detector = MTCNN()
    results = detector.detect_faces(pixels)

    if results:
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face).resize(required_size)  # Kích thước đã thay đổi
        return asarray(image)

# Lưu dữ liệu khuôn mặt
def process_faces(input_folder, output_file):
    faces, labels = [], []
    for subdir in listdir(input_folder):
        path = join(input_folder, subdir)
        if not isdir(path):
            continue
        for filename in listdir(path):
            face = extract_face(join(path, filename))
            if face is not None:
                faces.append(face)
                labels.append(subdir)
    savez_compressed(output_file, asarray(faces), asarray(labels))

if __name__ == '__main__':
    # Gán trực tiếp đường dẫn đến thư mục chứa dữ liệu ảnh và file nén đầu ra
    input_folder = r'D:\FACENET\face_recognition_project\data\raw\val'
    output_file = r'D:\FACENET\face_recognition_project\data\processed\face_dataset.npz'
    
    process_faces(input_folder, output_file)