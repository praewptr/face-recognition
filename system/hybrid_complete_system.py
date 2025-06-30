import os
import numpy as np
import cv2
from tqdm import tqdm
import logging

# Setup logging: only show important logs
logging.basicConfig(
    level=logging.INFO,  # Change to WARNING to reduce logs even more
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Try import required libraries
try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError("Please install insightface")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLO not available.")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("DeepFace not available")


class HybridFaceDetector:
    def __init__(self, retina_model='buffalo_l', yolo_model_path='model/yolov11n-face.pt'):
        logging.info("Initializing HybridFaceDetector")
        self.retina = FaceAnalysis(name=retina_model)
        self.retina.prepare(ctx_id=0)

        if YOLO_AVAILABLE and os.path.exists(yolo_model_path):
            self.yolo = YOLO(yolo_model_path)
            self.yolo_available = True
            logging.info("YOLO model loaded successfully")
        else:
            self.yolo_available = False
            logging.warning("YOLO model not available or path does not exist")

    def detect_faces(self, image_path: str):
        logging.info(f"Detecting faces in image: {image_path}")
        try:
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Image not found or unreadable: {image_path}")
                return []

            faces_retina = self.retina.get(image)
            if faces_retina:
                logging.info(f"RetinaFace detected {len(faces_retina)} face(s)")
                return [face.bbox.astype(int) for face in faces_retina]

            if self.yolo_available:
                logging.info("RetinaFace failed. Trying YOLO fallback...")
                results = self.yolo(image)[0]
                boxes = []
                for box in results.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, conf, _ = box
                    if conf > 0.5:
                        boxes.append([int(x1), int(y1), int(x2), int(y2)])
                logging.info(f"YOLO detected {len(boxes)} face(s)")
                return boxes

            logging.warning("No detection method succeeded")
            return []

        except Exception:
            logging.exception(f"Detection error in image {image_path}")
            return []


class DeepFaceRecognitionSystem:
    def __init__(self):
        logging.info("Initializing DeepFaceRecognitionSystem")
        self.priority_models = ['ArcFace']

    def extract_deepface_embedding(self, image_path: str):
        for model_name in self.priority_models:
            try:
                embedding_objs = DeepFace.represent(
                    img_path=image_path,
                    model_name=model_name,
                    enforce_detection=False,
                    detector_backend='opencv'
                )
                if embedding_objs:
                    embedding = np.array(embedding_objs[0]["embedding"])
                    normalized_embedding = embedding / np.linalg.norm(embedding)
                    logging.info(f"Embedding extracted with model: {model_name}")
                    return normalized_embedding, model_name
            except Exception as e:
                logging.warning(f"{model_name} failed: {e}")
                continue

        logging.error("All models failed to extract embedding")
        return None, None

    def extract_face_embedding(self, image_path: str):
        logging.info(f"Processing image: {image_path}")
        return self.extract_deepface_embedding(image_path)

    def build_database(self, images_directory, max_images=1000):
        logging.info(f"Building database from: {images_directory}")
        if not os.path.exists(images_directory):
            logging.error(f"Directory not found: {images_directory}")
            return []

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
        image_files = [os.path.join(root, file)
                    for root, _, files in os.walk(images_directory)
                    for file in files if file.endswith(image_extensions)]

        if max_images:
            image_files = image_files[:max_images]

        database = []
        model_stats = {model: 0 for model in self.priority_models}

        for image_path in tqdm(image_files, desc="Processing"):
            embedding, model_used = self.extract_face_embedding(image_path)

            if embedding is not None:
                filename = os.path.basename(image_path)
                person_id = filename.split('(')[-1].split(')')[0]

                database.append({
                    'embedding': embedding,
                    'person_id': person_id,
                    'image_path': image_path,
                    'method': 'deepface',
                    'model': model_used,
                    'embedding_dim': embedding.shape[0]
                })

                model_stats[model_used] += 1
                logging.info(f"Found face for {person_id} using {model_used}")
            else:
                logging.warning(f"Failed to extract embedding for: {image_path}")

        logging.info("Database built successfully")
        logging.info(f"Images processed: {len(database)}")
        return database

    def recognize_face(self, query_image_path, database, top_k=5, threshold=0.6):
        logging.info(f"Recognizing face from: {query_image_path}")
        query_embedding, query_model = self.extract_face_embedding(query_image_path)
        if query_embedding is None:
            logging.error("Failed to extract query embedding")
            return []

        similarities = []
        for face_data in database:
            db_embedding = face_data['embedding']
            if query_embedding.shape[0] != db_embedding.shape[0]:
                continue

            similarity = np.dot(query_embedding, db_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding))
            similarities.append((similarity, face_data))

        if not similarities:
            logging.warning("No compatible embeddings found")
            return []

        similarities.sort(key=lambda x: x[0], reverse=True)

        logging.info(f"Top {min(top_k, len(similarities))} matches:")
        for i, (similarity, face_data) in enumerate(similarities[:top_k]):
            status = "STRONG MATCH" if similarity > threshold else "Weak match"
            confidence = "HIGH" if similarity > 0.7 else "MEDIUM" if similarity > 0.5 else "LOW"
            logging.info(f"{i+1}. {status} | Person: {face_data['person_id']} | Sim: {similarity:.4f} | Conf: {confidence}")

        if similarities[0][0] > threshold:
            best = similarities[0][1]
            logging.info(f"BEST MATCH: {best['person_id']} (similarity: {similarities[0][0]:.4f})")
        else:
            logging.info("No confident match found")

        return similarities[:top_k]
