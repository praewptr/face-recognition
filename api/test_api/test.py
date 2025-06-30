import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from app.main import app

client = TestClient(app)


@pytest.fixture
def sample_image_path():
    return Path("test/test_img.png")


def test_recognition_success(sample_image_path):
    if not sample_image_path.exists():
        pytest.skip("Sample image not found.")

    with open(sample_image_path, "rb") as img:
        response = client.post(
            "/api/recognize",
            files={"file": ("test_img.jpg", img, "image/jpeg")},
            data={"top_k": 3, "threshold": 0.5}
        )

    assert response.status_code == 200
    json_data = response.json()
    assert "matches" in json_data
    assert "query_info" in json_data
    assert isinstance(json_data["matches"], list)



def test_invalid_file_upload():
    response = client.post(
        "/api/recognize",
        files={"file": ("test.txt", b"This is not an image", "text/plain")},
        data={"top_k": 3, "threshold": 0.5}
    )

    assert response.status_code == 500


def test_missing_file_field():
    response = client.post("/api/recognize", data={"top_k": 3, "threshold": 0.5})
    assert response.status_code == 422
