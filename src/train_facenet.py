import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from utils import preprocess_face, get_embedding

class FaceNetAutoUpdater:
    def __init__(self, facenet_model_path, existing_classifier_path, train_dir, output_model_path):
        """
        Khởi tạo class với các tham số cần thiết
        """
        # Load model FaceNet
        self.facenet = load_model(facenet_model_path)

        # Load model phân loại đã train trước đó
        # Trong TensorFlow 2.6, không sử dụng experimental.optimizer và compile lại
        self.classifier = load_model(existing_classifier_path, compile=False)
        self.classifier.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Đường dẫn dữ liệu
        self.train_dir = train_dir
        self.output_model_path = output_model_path

        # Load danh sách nhãn đã có
        self.existing_labels = np.load("label_names.npy", allow_pickle=True)
        print("Danh sách người đã có:", self.existing_labels)

    def find_new_people(self):
        """
        Tìm những người mới trong thư mục train bằng cách so sánh với danh sách cũ
        """
        # Lấy danh sách tất cả người trong thư mục train
        all_people = [d for d in os.listdir(self.train_dir)
                      if os.path.isdir(os.path.join(self.train_dir, d))]

        # Tìm những người mới (không có trong danh sách cũ)
        new_people = [p for p in all_people if p not in self.existing_labels]

        if not new_people:
            print("Không tìm thấy người mới trong thư mục train!")
            return None, None

        print(f"Tìm thấy {len(new_people)} người mới:", new_people)

        # Lấy đường dẫn ảnh của người mới
        new_image_paths = []
        new_labels = []

        for person in new_people:
            person_dir = os.path.join(self.train_dir, person)
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                new_image_paths.append(image_path)
                new_labels.append(person)

        return new_image_paths, new_labels

    def prepare_new_data(self):
        """
        Chuẩn bị dữ liệu cho người mới:
        - Tìm người mới
        - Tạo embedding cho ảnh mới
        - Thêm nhãn mới vào danh sách nhãn
        """
        # Tìm người mới
        new_image_paths, new_labels = self.find_new_people()
        if new_image_paths is None:
            return None, None, None

        # Tạo embedding cho ảnh mới
        new_faces = []
        final_new_labels = []

        print(f"Đang xử lý {len(new_image_paths)} ảnh của người mới...")

        for image_path, label in zip(new_image_paths, new_labels):
            face = preprocess_face(image_path)
            if face is not None:
                embedding = get_embedding(self.facenet, face)
                new_faces.append(embedding)
                final_new_labels.append(label)
            else:
                print(f"Không thể xử lý ảnh: {image_path}")

        if not new_faces:
            print("Không có ảnh nào được xử lý thành công!")
            return None, None, None

        # Chuyển đổi sang numpy array
        new_faces = np.array(new_faces)
        final_new_labels = np.array(final_new_labels)

        # Cập nhật danh sách nhãn
        all_labels = np.concatenate([self.existing_labels, np.unique(final_new_labels)])
        all_labels = np.unique(all_labels)

        # Mã hóa nhãn
        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels)
        encoded_new_labels = label_encoder.transform(final_new_labels)
        one_hot_new_labels = to_categorical(encoded_new_labels, num_classes=len(all_labels))

        # Lưu danh sách nhãn mới
        np.save("label_names.npy", label_encoder.classes_)
        print(f"Đã cập nhật danh sách người: {len(all_labels)} người")

        return new_faces, one_hot_new_labels, len(all_labels)

    def build_updated_classifier(self, num_classes):
        """
        Tạo model phân loại mới với số lượng class đã cập nhật
        """
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(128,)))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Sao chép trọng số từ các lớp của model cũ
        for i in range(len(model.layers) - 1):
            model.layers[i].set_weights(self.classifier.layers[i].get_weights())

        self.model = model

    def train(self, X_train, y_train, epochs=5):
        """
        Train model với dữ liệu mới
        """
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_split=0.2
        )
        return history

    def save_model(self):
        """
        Lưu model đã cập nhật
        """
        self.model.save(self.output_model_path)
        print(f"Đã lưu model tại {self.output_model_path}")


def main():
    # Khai báo đường dẫn
    facenet_model_path = r'D:\FACENET\face-recognition-project\models\facenet_keras.h5'
    existing_classifier_path = 'face_recognition_classifier.h5'
    train_dir = r'D:\FACENET\face-recognition-project\data\raw\train'
    output_model_path = 'face_recognition_classifier.h5'

    # Khởi tạo class updater
    updater = FaceNetAutoUpdater(
        facenet_model_path,
        existing_classifier_path,
        train_dir,
        output_model_path
    )

    # Tìm và chuẩn bị dữ liệu người mới
    print("Đang tìm và chuẩn bị dữ liệu người mới...")
    new_faces, new_labels, num_classes = updater.prepare_new_data()

    if new_faces is not None:
        # Tạo và cập nhật model
        print("Đang cập nhật model...")
        updater.build_updated_classifier(num_classes)

        # Train model với dữ liệu mới
        print("Đang train model với dữ liệu mới...")
        updater.train(new_faces, new_labels)

        # Lưu model đã cập nhật
        updater.save_model()
    else:
        print("Không có dữ liệu mới để cập nhật!")


if __name__ == "__main__":
    main()
