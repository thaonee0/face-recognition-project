from keras.src.api_export import keras_export
from keras.src.applications import imagenet_utils
from keras.src.applications import resnet
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os

@keras_export(
    [
        "keras.applications.ResNet50V2",
        "keras.applications.resnet_v2.ResNet50V2",
    ]
)
class ResNet50V2:
    """Class to instantiate the ResNet50V2 architecture."""

    def __init__(self, include_top=True, weights="imagenet", input_tensor=None,
                 input_shape=None, pooling=None, classes=1000, classifier_activation="softmax"):
        self.model = self._build_model(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation)

    def _build_model(self, include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation):
        """Builds the ResNet50V2 model."""
        def stack_fn(x):
            x = resnet.stack_residual_blocks_v2(x, 64, 3, name="conv2")
            x = resnet.stack_residual_blocks_v2(x, 128, 4, name="conv3")
            x = resnet.stack_residual_blocks_v2(x, 256, 6, name="conv4")
            return resnet.stack_residual_blocks_v2(x, 512, 3, stride1=1, name="conv5")

        return resnet.ResNet(
            stack_fn,
            True,
            True,
            name="resnet50v2",
            weights_name="resnet50v2",
            include_top=include_top,
            weights=weights,
            input_tensor=input_tensor,
            input_shape=input_shape,
            pooling=pooling,
            classes=classes,
            classifier_activation=classifier_activation,
        )

    def predict(self, x):
        return self.model.predict(x)

# Hàm tiền xử lý
@keras_export("keras.applications.resnet_v2.preprocess_input")
def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(x, data_format=data_format, mode="tf")

# Hàm giải mã dự đoán
@keras_export("keras.applications.resnet_v2.decode_predictions")
def decode_predictions(preds, top=5):
    return imagenet_utils.decode_predictions(preds, top=top)

# Tạo mô hình
model = ResNet50V2(weights='imagenet')

# Đường dẫn đến dữ liệu hình ảnh
train_image_dir = r"D:\FACENET\face_recognition_project\data\raw\train"  # Đường dẫn hình ảnh huấn luyện
val_image_dir = r"D:\FACENET\face_recognition_project\data\raw\val"      # Đường dẫn hình ảnh kiểm thử

# Tiền xử lý hình ảnh
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Đặt kích thước hình ảnh
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
    img_array = preprocess_input(img_array)  # Tiền xử lý hình ảnh
    return img_array

# Dự đoán
def predict(img_path):
    img_array = prepare_image(img_path)
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=5)
    return decoded_predictions

# Ví dụ sử dụng
if __name__ == "__main__":
    # Duyệt qua các hình ảnh trong thư mục huấn luyện
    print("Dự đoán cho các hình ảnh trong thư mục huấn luyện:")
    for img_file in os.listdir(train_image_dir):
        img_path = os.path.join(train_image_dir, img_file)  # Đường dẫn đầy đủ đến hình ảnh
        if os.path.isfile(img_path):  # Kiểm tra xem có phải là file hình ảnh không
            predictions = predict(img_path)
            print(f"Predictions for {img_file}: {predictions}")

    # Duyệt qua các hình ảnh trong thư mục kiểm thử
    print("Dự đoán cho các hình ảnh trong thư mục kiểm thử:")
    for img_file in os.listdir(val_image_dir):
        img_path = os.path.join(val_image_dir, img_file)  # Đường dẫn đầy đủ đến hình ảnh
        if os.path.isfile(img_path):  # Kiểm tra xem có phải là file hình ảnh không
            predictions = predict(img_path)
            print(f"Predictions for {img_file}: {predictions}")
