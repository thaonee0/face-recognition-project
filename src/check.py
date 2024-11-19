from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Tải mô hình với optimizer chuẩn
classifier_model_path = 'face_recognition_classifier.h5'
model = load_model(classifier_model_path, custom_objects={'Adam': Adam})
