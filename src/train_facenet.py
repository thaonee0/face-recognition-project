import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from utils import preprocess_face, get_embedding

class FaceNetTrainer:
    def __init__(self, facenet_model_path, images_path, output_model_path):
        self.facenet = load_model(facenet_model_path)
        self.images_path = images_path
        self.output_model_path = output_model_path

    def load_data(self):
        """Load face images and labels from directory structure"""
        image_paths = []
        labels = []

        for label_dir in os.listdir(self.images_path):
            label_path = os.path.join(self.images_path, label_dir)
            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    image_paths.append(os.path.join(label_path, image_name))
                    labels.append(label_dir)

        faces, final_labels = [], []
        for image_path, label in zip(image_paths, labels):
            face = preprocess_face(image_path)
            if face is None:
                print(f"Lỗi khi xử lý ảnh: {image_path}")
                continue  # Bỏ qua ảnh này nếu không xử lý được
            if face is not None:
                embedding = get_embedding(self.facenet, face)
                if embedding is None:
                    print(f"Lỗi khi tạo embedding cho ảnh: {image_path}")
                    continue

                faces.append(embedding)
                final_labels.append(label)

        faces = np.array(faces)
        final_labels = np.array(final_labels)
        
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(final_labels)
        one_hot_labels = to_categorical(encoded_labels)
        
        np.save("label_names.npy", label_encoder.classes_)
        
        return train_test_split(faces, one_hot_labels, test_size=0.2, random_state=42), len(label_encoder.classes_)

    def build_classifier(self, num_classes):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(128,)))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        
        model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model

    def train(self, X_train, y_train, X_test, y_test):
        history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
        return history

    def save_model(self):
        self.model.save(self.output_model_path)
        print(f"Model saved to {self.output_model_path}")

if __name__ == "__main__":
    facenet_model_path = r'D:\FACENET\face-recognition-project\models\facenet_keras.h5'
    images_path = 'D:/FACENET/face-recognition-project/data/raw/train'
    output_model_path = 'face_recognition_classifier.h5'
    
    trainer = FaceNetTrainer(facenet_model_path, images_path, output_model_path)
    (X_train, X_test, y_train, y_test), num_classes = trainer.load_data()
    trainer.build_classifier(num_classes)
    trainer.train(X_train, y_train, X_test, y_test)
    trainer.save_model()
