from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint_reports_status():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_requires_valid_age():
    response = client.post(
        "/predict",
        json={
            "age": 12,
            "job": "admin",
            "balance": 5000,
            "housing": "yes",
            "loan": "no",
        },
    )
    assert response.status_code == 422
