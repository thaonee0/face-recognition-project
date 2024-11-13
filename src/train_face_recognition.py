import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import os

class FaceNetClassifier:
    def __init__(self, embedding_size=128):
        self.embedding_size = embedding_size
        self.model = None

    def load_data(self, embeddings_path):
        """Load face embeddings and labels"""
        # Tải dữ liệu từ file .npz
        data = np.load(embeddings_path)
        embeddings = data['embeddings']
        labels = data['labels']
        
        # Mã hóa nhãn thành dạng one-hot
        label_encoder = {label: idx for idx, label in enumerate(np.unique(labels))}
        encoded_labels = np.array([label_encoder[label] for label in labels])
        one_hot_labels = to_categorical(encoded_labels)
        
        # Chia bộ dữ liệu thành tập huấn luyện và kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(
            embeddings, one_hot_labels,
            test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, len(label_encoder)

    def build_model(self, num_classes):
        """Build MLP classifier model"""
        model = Sequential()
        model.add(Dense(512, activation='relu', input_dim=self.embedding_size))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        self.model = model
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, X_train, y_train, X_test, y_test):
        """Train the model"""
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=10, batch_size=32
        )
        return history

    def save_model(self, model_path):
        """Save the trained model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Create directory if it doesn't exist
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

def main():
    classifier = FaceNetClassifier()

    # Tải dữ liệu từ file .npz chứa embeddings và nhãn
    embeddings_path = 'D:/FACENET/face-recognition-project/data/processed/face_embedding.npz'
    X_train, X_test, y_train, y_test, num_classes = classifier.load_data(embeddings_path)
    
    # Xây dựng và huấn luyện mô hình
    classifier.build_model(num_classes)
    classifier.train(X_train, y_train, X_test, y_test)
    
    # Lưu mô hình sau khi huấn luyện vào thư mục "trained"
    model_path = 'D:/FACENET/face-recognition-project/models/trained/face_recognition_model.h5'
    classifier.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
