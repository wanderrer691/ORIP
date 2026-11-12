
import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    # Инициализация TestClient с запуском lifespan приложения
    with TestClient(app) as test_client:
        yield test_client


# Тест 1. Успешное предсказание качества вина
def test_predict_success(client):
    valid_payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.70,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }

    response = client.post(
        "/v1/predict",
        json=valid_payload
    )

    assert response.status_code == 200

    response_json = response.json()

    # Проверка наличия полей ответа
    assert "predicted_class" in response_json
    assert "probabilities" in response_json
    assert "model_status" in response_json

    # Проверка типов данных
    assert isinstance(response_json["predicted_class"], int)
    assert isinstance(response_json["probabilities"], list)
    assert isinstance(response_json["model_status"], str)

    # Проверка содержимого ответа
    assert len(response_json["probabilities"]) > 0
    assert response_json["model_status"] == "wine_quality_onnx_v1.0"

    # Проверка вероятностей
    for probability in response_json["probabilities"]:
        assert isinstance(probability, (int, float))
        assert 0 <= probability <= 1


# Тест 2. Ошибка при передаче отрицательного значения
def test_predict_negative_value_error(client):
    invalid_payload = {
        "fixed_acidity": -7.4,
        "volatile_acidity": 0.70,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }

    response = client.post(
        "/v1/predict",
        json=invalid_payload
    )

    assert response.status_code == 422
    assert "greater than 0" in response.text


# Тест 3. Ошибка при передаче строки вместо числа
def test_predict_type_error(client):
    invalid_payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.70,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": "string_data"
    }

    response = client.post(
        "/v1/predict",
        json=invalid_payload
    )

    assert response.status_code == 422
    assert "unable to parse string as a number" in response.text


# Тест 4. Ошибка при отправке пустого JSON
def test_predict_empty_payload_error(client):
    response = client.post(
        "/v1/predict",
        json={}
    )

    assert response.status_code == 422
    assert "Field required" in response.text
