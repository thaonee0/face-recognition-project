import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
from tensorflow.keras.preprocessing import image

class FaceEmbedding:
    def __init__(self, model_path="D:/FACENET/face-recognition-project/models/facenet_keras.h5"):
        # Load the FaceNet model
        self.model = tf.keras.models.load_model(model_path)
        
        # Configure GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    def preprocess_face(self, face):
        """Preprocess face with L2 normalization"""
        face = face.astype('float32')
        
        # Standardization (zero mean, unit variance)
        mean = face.mean()
        std = face.std()
        face = (face - mean) / std
        
        # L2 normalization
        norm = np.sqrt(np.sum(np.square(face)))
        normalized_face = face / norm
        
        return normalized_face

    def generate_embeddings(self, faces, batch_size=32):
        """Generate embeddings in batches"""
        print("\nGenerating embeddings...")
        embeddings = []
        
        n_batches = len(faces) // batch_size + (1 if len(faces) % batch_size != 0 else 0)
        
        for i in tqdm(range(0, len(faces), batch_size), total=n_batches):
            batch = faces[i:i+batch_size]
            batch_preprocessed = np.array([self.preprocess_face(face) for face in batch])
            batch_embeddings = self.model.predict(batch_preprocessed, verbose=0)
            
            # L2 normalize embeddings
            norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
            batch_embeddings = batch_embeddings / norms
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)

    def process_dataset(self, input_path, output_path):
        """Process entire dataset"""
        print("Loading face dataset...")
        
        # Initialize lists to store faces and corresponding labels
        faces = []
        labels = []
        
        # Iterate through the directories (each directory is a person)
        for label in os.listdir(input_path):
            label_folder = os.path.join(input_path, label)
            if os.path.isdir(label_folder):
                for img_name in tqdm(os.listdir(label_folder), desc=f"Processing {label}"):
                    img_path = os.path.join(label_folder, img_name)
                    img = image.load_img(img_path, target_size=(160, 160))
                    img_array = image.img_to_array(img)
                    faces.append(img_array)
                    labels.append(label)  # Use the folder name as the label
        
        # Convert lists to numpy arrays
        faces = np.array(faces)
        labels = np.array(labels)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(faces)
        
        # Save embeddings and labels (names of the directories) to output path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(output_path, 
                          embeddings=embeddings, 
                          labels=labels)
        
        print(f"\nEmbeddings processing completed:")
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # Define paths for input and output
    FACE_DATASET_PATH = "D:/FACENET/face-recognition-project/data/raw/train"  # Update this path to your data folder
    FACE_EMBEDDINGS_PATH = "D:/FACENET/face-recognition-project/data/processed/face_embedding.npz"  # Path to save embeddings
    
    # Process the dataset
    embedding_generator = FaceEmbedding()  # Initialize with the correct model path
    embedding_generator.process_dataset(FACE_DATASET_PATH, FACE_EMBEDDINGS_PATH)
