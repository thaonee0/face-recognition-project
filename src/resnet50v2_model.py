from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.applications import ResNet50V2
import numpy as np

class CustomResNet50V2:
    """Class to instantiate the ResNet50V2 architecture."""

    def __init__(self, include_top=False, weights="imagenet", input_tensor=None,
                 input_shape=(224, 224, 3), pooling='avg', classes=1000):
        self.model = self._build_model(include_top, weights, input_tensor, input_shape, pooling, classes)

    def _build_model(self, include_top, weights, input_tensor, input_shape, pooling, classes):
        """Builds the ResNet50V2 model."""
        base_model = ResNet50V2(
            include_top=include_top,
            weights=weights,
            input_tensor=input_tensor,
            input_shape=input_shape,
            pooling=pooling,
            classes=classes,
        )

        # Chỉ giữ lại phần backbone của ResNet50V2 và loại bỏ các lớp phân loại
        x = base_model.output
        # Tạo lớp Flatten để lấy embedding
        x = Flatten()(x)  # Chuyển đổi đầu ra thành vector một chiều
        x = Dense(20, activation='relu')(x)  # Lấy 20 đặc trưng

        # Tạo mô hình mới
        model = Model(inputs=base_model.input, outputs=x)

        return model

    def predict(self, x):
        return self.model.predict(x)

# Hàm tiền xử lý
def preprocess_input(x, data_format=None):
    from keras.applications.imagenet_utils import preprocess_input as imagenet_preprocess
    return imagenet_preprocess(x, data_format=data_format, mode="tf")

# Tạo mô hình
def create_resnet50v2_model():
    model = CustomResNet50V2(weights="imagenet")
    # Kiểm tra kích thước đầu ra
    dummy_input = np.random.rand(1, 224, 224, 3)  # Giả lập đầu vào
    embedding = model.predict(dummy_input)
    print("Kích thước đầu ra embedding:", embedding.shape)  # Kiểm tra kích thước đầu ra

    # Lưu mô hình vào tệp .h5
    model.model.save('D:\\FACENET\\face_recognition_project\\models\\resnet50v2_model.h5')
    print("Mô hình đã được lưu thành công tại 'D:\\FACENET\\face_recognition_project\\models\\resnet50v2_model.h5'")
    
    return model

# Kiểm tra xem có tồn tại mô hình không
if __name__ == "__main__":
    model = create_resnet50v2_model()
    print("ResNet50V2 model has been created.")