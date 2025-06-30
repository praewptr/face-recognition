from fastapi import FastAPI, HTTPException, UploadFile, File
import numpy as np
import cv2
import os
import uvicorn
from system.hybrid_complete_system import DeepFaceRecognitionSystem
from system.utils.load_database import load_face_database
from core.config import DB_PATH


database = load_face_database(DB_PATH)


app = FastAPI()
system = DeepFaceRecognitionSystem()

@app.post("/api/recognize")
async def recognize(
    file: UploadFile = File(...),
    top_k: int = 5,
    threshold: float = 0.6
):

    try:
        contents = await file.read()
        image_np = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        temp_path = f"temp_{file.filename}"
        cv2.imwrite(temp_path, image)

        results = system.recognize_face(temp_path, database, top_k, threshold)

        response = []
        for sim, face_data in results:
            confidence = "HIGH" if sim > 0.7 else "MEDIUM" if sim > 0.5 else "LOW"
            response.append({
                "person_id": face_data["person_id"],
                "similarity": float(sim),
                "confidence": confidence,
                "image_path": face_data["image_path"]
            })

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return {
            "result": response,
            "query_info": {
                "filename": file.filename,
                "total_matches": len(response)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)