import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    # Инициализируем TestClient внутри контекстного менеджера lifespan
    with TestClient(app) as test_client:
        yield test_client


def test_predict_success(client):
    valid_payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post("/v1/predict", json=valid_payload)

    assert response.status_code == 200

    response_json = response.json()

    assert "predicted_class" in response_json
    assert "probabilities" in response_json
    assert "model_status" in response_json

    assert isinstance(response_json["predicted_class"], int)
    assert isinstance(response_json["probabilities"], list)
    assert len(response_json["probabilities"]) == 2


def test_predict_negative_value_error(client):
    invalid_payload = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post("/v1/predict", json=invalid_payload)

    assert response.status_code == 422
    assert "Input should be greater than 0" in response.text


def test_predict_type_error(client):
    invalid_payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": "string_data",
    }

    response = client.post("/v1/predict", json=invalid_payload)

    assert response.status_code == 422
    assert "unable to parse string as a number" in response.text


def test_predict_empty_payload_error(client):
    response = client.post("/v1/predict", json={})

    assert response.status_code == 422
    assert "Field required" in response.text