import sys
import io
import os
from numpy import load, asarray, savez_compressed, expand_dims
from tensorflow.keras.models import load_model
import tensorflow as tf

# Tắt GPU
tf.config.set_visible_devices([], 'GPU')

# Đảm bảo output của hệ thống sử dụng encoding utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Đường dẫn tới mô hình pre-trained
facenet_model_path = r'D:\FACENET\face-recognition-project\models\facenet_keras.h5'

# Tải mô hình đã huấn luyện
model = load_model(facenet_model_path)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Lấy embedding cho một khuôn mặt
def get_embedding(model, face_pixels):
    # Chuẩn hóa giá trị pixel
    face_pixels = face_pixels.astype('float32')
    # Chuẩn hóa giá trị pixel trên toàn bộ kênh (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # Chuyển khuôn mặt thành một mẫu đơn
    samples = expand_dims(face_pixels, axis=0)
    # Dự đoán để lấy embedding
    yhat = model.predict(samples)
    return yhat[0]

def main():
    # Đường dẫn dữ liệu và mô hình
    face_dataset_path = r'D:\FACENET\face-recognition-project\data\processed\face_dataset'
    face_embeddings_path = r'D:\FACENET\face-recognition-project\data\processed\face_embedding'
    index_file = 'index.npz'
    
    # Tải tệp index.npz để lấy danh sách các tệp .npz
    index_file_path = os.path.join(face_dataset_path, index_file)
    if not os.path.exists(index_file_path):
        print(f"Không tìm thấy tệp index: {index_file_path}")
        return
    
    # Load thông tin từ tệp index
    index_data = load(index_file_path, allow_pickle=True)['index_data']
    print(f"Tải {len(index_data)} mục từ tệp index.") 

    all_embeddings_exist = True
    for data_info in index_data:
        file_path = data_info['file_path']
        subdir = data_info['name']
        
        embedding_file_path = os.path.join(face_embeddings_path, f"{subdir}_embedding.npz")
        if not os.path.exists(embedding_file_path):
            all_embeddings_exist = False
            break  # Không cần kiểm tra thêm nếu phát hiện tệp thiếu

    if all_embeddings_exist:
        print("Tất cả tệp embedding đã tồn tại. Không cần xử lý thêm.")
        return  # Dừng chương trình nếu tất cả embedding đã tồn tại
    
    # Lặp qua từng tệp .npz từ index_data
    for data_info in index_data:
        file_path = data_info['file_path']
        subdir = data_info['name']

        # Kiểm tra trực tiếp xem tệp embedding đã tồn tại chưa
        embedding_file_path = os.path.join(face_embeddings_path, f"{subdir}_embedding.npz")
        if os.path.exists(embedding_file_path):
            print(f"Embedding cho lớp {subdir} đã tồn tại")
            continue
        
        # Tải dữ liệu khuôn mặt từ tệp .npz
        data = load(file_path)
        faces = data['faces']
        labels = data['labels']
        
        print(f"Tải dữ liệu khuôn mặt cho lớp {subdir} từ tệp {file_path}: {faces.shape}") 
        
        # Chuyển mỗi khuôn mặt trong bộ dữ liệu thành một embedding (bao gồm cả train và test)
        newFaces = [get_embedding(model, face_pixels) for face_pixels in faces]
        newFaces = asarray(newFaces)
        print(f"Embedding cho dữ liệu của lớp {subdir}: {newFaces.shape}") 
        
        # Lưu embedding vào tệp nén mới
        embedding_file_path = os.path.join(face_embeddings_path, f"{subdir}_embedding.npz")
        savez_compressed(embedding_file_path, faces=newFaces, labels=labels)
        print(f"Đã lưu embedding cho lớp {subdir} vào {embedding_file_path}")  

if __name__ == '__main__':
    main()