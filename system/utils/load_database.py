import pickle
from pathlib import Path

def load_face_database(path: str):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")

    with open(path, "rb") as f:
        database = pickle.load(f)

    return database