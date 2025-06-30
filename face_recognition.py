
from system.hybrid_complete_system import DeepFaceRecognitionSystem
from system.utils.load_database import load_face_database
from core.config import DB_PATH

def face_recognition_system(database,image_path, top_k=5, threshold=0.6):

    face_recognition_system = DeepFaceRecognitionSystem()
    result = face_recognition_system.recognize_face(image_path, database, top_k, threshold)

    return result



if __name__ == "__main__":

    database = load_face_database(DB_PATH)

    image_path = "data/dataset/Humans/1 (3).jpeg"
    top_k = 5
    threshold = 0.6
    results = face_recognition_system(database, image_path, top_k, threshold)
    response = []
    for sim, face_data in results:
        confidence = "HIGH" if sim > 0.7 else "MEDIUM" if sim > 0.5 else "LOW"
        response.append({
            "person_id": face_data["person_id"],
            "similarity": float(sim),
            "confidence": confidence,
            "image_path": face_data["image_path"]
        })


