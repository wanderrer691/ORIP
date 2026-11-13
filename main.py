
from typing import List
from contextlib import asynccontextmanager
import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Схема входящего запроса
class WineRequestData(BaseModel):

    fixed_acidity: float = Field(
        ..., gt=0, description="Фиксированная кислотность", example=7.4
    )

    volatile_acidity: float = Field(
        ..., gt=0, description="Летучая кислотность", example=0.70
    )

    citric_acid: float = Field(
        ..., ge=0, description="Лимонная кислота", example=0.0
    )

    residual_sugar: float = Field(
        ..., ge=0, description="Остаточный сахар", example=1.9
    )

    chlorides: float = Field(
        ..., ge=0, description="Хлориды", example=0.076
    )

    free_sulfur_dioxide: float = Field(
        ..., ge=0, description="Свободный диоксид серы", example=11.0
    )

    total_sulfur_dioxide: float = Field(
        ..., ge=0, description="Общий диоксид серы", example=34.0
    )

    density: float = Field(
        ..., gt=0, description="Плотность", example=0.9978
    )

    pH: float = Field(
        ..., gt=0, description="Уровень pH", example=3.51
    )

    sulphates: float = Field(
        ..., ge=0, description="Сульфаты", example=0.56
    )

    alcohol: float = Field(
        ..., gt=0, description="Содержание алкоголя", example=9.4
    )


# Схема исходящего ответа
class WineResponseData(BaseModel):

    predicted_class: int = Field(
        ..., description="Предсказанное качество вина"
    )

    probabilities: List[float] = Field(
        ..., description="Вероятности принадлежности к классам качества"
    )

    model_status: str = Field(
        ..., description="Статус и версия модели"
    )


# Единая сессия ONNX Runtime
onnx_session = None


# Управление жизненным циклом приложения
@asynccontextmanager
async def lifespan(app: FastAPI):

    global onnx_session

    print(
        "Инициализация сервера: "
        "загрузка графа ONNX в оперативную память..."
    )

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(
        BASE_DIR,
        "wine_quality_model.onnx"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ONNX-модель не найдена: {model_path}"
        )

    onnx_session = ort.InferenceSession(model_path)

    print("Модель WineQT успешно загружена!")

    yield

    print("Остановка сервера: освобождение ресурсов памяти...")

    onnx_session = None


# Создание приложения FastAPI
app = FastAPI(
    title="Wine Quality Prediction API",
    description="API для предсказания качества вина с использованием ONNX Runtime",
    version="1.0.0",
    lifespan=lifespan
)


# Эндпоинт предсказания
@app.post(
    "/v1/predict",
    response_model=WineResponseData
)
def predict(data: WineRequestData):

    if onnx_session is None:
        raise HTTPException(
            status_code=503,
            detail="Модель еще не загружена"
        )

    # Формируем входной массив из 11 признаков
    input_array = np.array([[
        data.fixed_acidity,
        data.volatile_acidity,
        data.citric_acid,
        data.residual_sugar,
        data.chlorides,
        data.free_sulfur_dioxide,
        data.total_sulfur_dioxide,
        data.density,
        data.pH,
        data.sulphates,
        data.alcohol
    ]], dtype=np.float32)

    # Получаем имя входного тензора
    input_name = onnx_session.get_inputs()[0].name

    # Выполняем инференс
    raw_predictions = onnx_session.run(
        None,
        {input_name: input_array}
    )

    # Получаем предсказанное качество вина
    predicted_class = int(raw_predictions[0][0])

    # Получаем вероятности классов
    probability_output = raw_predictions[1][0]

    if isinstance(probability_output, dict):
        probabilities = [
            float(value)
            for value in probability_output.values()
        ]
    else:
        probabilities = [
            float(value)
            for value in probability_output
        ]

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "model_status": "wine_quality_onnx_v1.1_updated"
    }
