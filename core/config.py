from dotenv import load_dotenv
import os

load_dotenv()
DB_PATH = os.getenv("DB_PATH")
DATASET_PATH = os.getenv("DATASET_PATH", "data/dataset/Humans")
