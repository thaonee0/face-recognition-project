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

        # Create labels based on subdirectories in dataset folder (using folder names as labels)
        self.class_names = sorted([name for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, name))])
        
        print("Model and dependencies loaded successfully.")

    def preprocess_face(self, face_img):
        """Preprocess the face image for prediction"""
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

    def detect_face(self, frame):
        """Detect faces in the frame"""
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        results = self.detector.detect_faces(rgb_frame)
        
        detected_faces = []
        for result in results:
            x, y, w, h = result['box']
            confidence = result['confidence']
            
            # Only use faces with high confidence
            if confidence > 0.95:
                x, y = max(0, x), max(0, y)  # Ensure coordinates are not negative
                face = frame[y:y+h, x:x+w]  # Crop the face
                
                detected_faces.append((face, (x, y, w, h)))
        
        return detected_faces

    def predict_face(self, face):
        """Recognize the face"""
        # Preprocess
        processed_face = self.preprocess_face(face)
        
        # Generate embedding using FaceNet
        embedding = np.expand_dims(processed_face, axis=0)
        
        # Predict class
        predictions = self.model.predict(embedding)
        best_idx = np.argmax(predictions[0])
        confidence = predictions[0][best_idx]
        
        if confidence > 0.7:  # Confidence threshold
            return self.class_names[best_idx], confidence
        else:
            return "Unknown", confidence

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
