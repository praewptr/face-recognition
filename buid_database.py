
from system.hybrid_complete_system import DeepFaceRecognitionSystem
from core.config import DB_PATH
import pickle
#build the face database
def build_face_database(images_directory, max_images=1000):
    face_recognition_system = DeepFaceRecognitionSystem()
    database = face_recognition_system.build_database(images_directory, max_images)
    return database

if __name__ == "__main__":

    images_directory = "data/dataset/Humans"
    max_images = 1000
    database = build_face_database(images_directory, max_images)

    with open(DB_PATH, 'wb') as db_file:
        pickle.dump(database, db_file)

    print(f"Database built with {len(database)} entries.")