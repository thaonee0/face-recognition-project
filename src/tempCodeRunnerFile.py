import cv2
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import os

class RealtimeRecognition:
    def __init__(self, model_dir):
        # Paths for the model
        self.model_path = os.path.join(model_dir, 'face_recognition_model.h5')
        
        # Load model
        self.model = tf.keras.models.load_model(self.model_path)
        
        # Initialize MTCNN detector
        self.detector = MTCNN()

        # Load precomputed embeddings and labels
        embedding_data = np.load(r'D:\FACENET\face-recognition-project\data\processed\face_embedding.npz')
        self.embeddings = embedding_data['embeddings']
        self.labels = embedding_data['labels']

        # Normalize the embeddings
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        print("Model and embeddings loaded successfully.")

    def preprocess_face(self, face_img):
        """Preprocess the face image for embedding calculation"""
        # Resize to 160x160 (FaceNet requirement)
        face_img = cv2.resize(face_img, (160, 160))
        
        # Convert BGR to RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Normalize: Standardization (zero mean, unit variance)
        face_img = face_img.astype('float32')
        mean, std = face_img.mean(), face_img.std()
        face_img = (face_img - mean) / std
        
        # L2 Normalization
        norm = np.sqrt(np.sum(np.square(face_img)))
        face_img = face_img / norm
        
        return face_img

    def predict_face(self, face):
        """Recognize the face by comparing with precomputed embeddings"""
        # Preprocess face (resize, normalize, etc.)
        processed_face = self.preprocess_face(face)
        
        # Tạo embedding từ mô hình face_recognition_model.h5 (mô hình đã được huấn luyện từ trước)
        # processed_face phải có dạng (1, 160, 160, 3) sau khi tiền xử lý.
        embedding = self.model.predict(np.expand_dims(processed_face, axis=0))  # Đảm bảo kích thước là (1, 160, 160, 3)
        
        # L2 normalization cho embedding
        embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

        # Tính khoảng cách Euclidean giữa embedding của khuôn mặt hiện tại và các embedding đã lưu
        distances = np.linalg.norm(self.embeddings - embedding, axis=1)
        
        # Tìm label với khoảng cách nhỏ nhất
        best_idx = np.argmin(distances)
        confidence = 1 / (distances[best_idx] + 1e-6)  # Confidence là nghịch đảo của khoảng cách

        if confidence > 0.7:  # Ngưỡng độ tin cậy
            return self.labels[best_idx], confidence
        else:
            return "Unknown", confidence

    def detect_face(self, frame):
        """Detect faces in the frame"""
        # Use MTCNN to detect faces in the frame
        results = self.detector.detect_faces(frame)
        
        faces = []
        face_positions = []
        
        for result in results:
            if result['confidence'] >= 0.9:  # Confidence threshold
                x, y, w, h = result['box']
                face = frame[y:y + h, x:x + w]
                faces.append(face)
                face_positions.append((x, y, w, h))
        
        return zip(faces, face_positions)

    def draw_detection(self, frame, face_pos, name, confidence):
        """Draw recognition result on the frame"""
        x, y, w, h = face_pos
        
        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Label with name and confidence
        label = f"{name} ({confidence * 100:.1f}%)"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def run(self):
        """Run real-time face recognition"""
        cap = cv2.VideoCapture(0)  # Open webcam
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Detect faces in the frame
            detected_faces = self.detect_face(frame)
            
            if detected_faces:
                for face, face_pos in detected_faces:
                    # Recognize face
                    name, confidence = self.predict_face(face)
                    
                    # Draw results on the frame
                    self.draw_detection(frame, face_pos, name, confidence)
            
            else:
                print("No face detected in the frame.")
            
            # Show frame with recognition results
            cv2.imshow("Realtime Face Recognition", frame)
            
            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Run recognition
if __name__ == "__main__":
    model_dir = r'D:\FACENET\face-recognition-project\models\trained'  # Path to directory containing your trained model and dataset subfolders
    recognition_system = RealtimeRecognition(model_dir)
    recognition_system.run()
