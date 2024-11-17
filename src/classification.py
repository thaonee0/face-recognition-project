import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from joblib import dump

# Đường dẫn tới các tệp embedding
embedding_path = r'D:\FACENET\face-recognition-project\data\processed\face_embedding'

# Tải dữ liệu embeddings
def load_embeddings(embedding_path):
    embeddings = []
    labels = []
    for file in os.listdir(embedding_path):
        if file.endswith("_embedding.npz"):
            data = np.load(os.path.join(embedding_path, file))
            embeddings.extend(data['faces'])
            labels.extend(data['labels'])
    return np.array(embeddings), np.array(labels)

def main():
    # Tải embeddings và nhãn
    X, y = load_embeddings(embedding_path)
    print(f"Loaded {len(X)} embeddings with labels")

    # Khởi tạo và huấn luyện mô hình SVM
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)

    # Báo cáo kết quả
    print("Training completed.")
    print(classification_report(y, model.predict(X)))

    # Lưu mô hình
    model_path = r'D:\FACENET\face-recognition-project\models\svm_model.joblib'
    dump(model, model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    main()
