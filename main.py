
from typing import List
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field


# Схема входящего запроса
class IrisRequestData(BaseModel):
    sepal_length: float = Field(
        ..., gt=0, description="Длина чашелистика в см", example=5.1
    )
    sepal_width: float = Field(
        ..., gt=0, description="Ширина чашелистика в см", example=3.5
    )
    petal_length: float = Field(
        ..., gt=0, description="Длина лепестка в см", example=1.4
    )
    petal_width: float = Field(
        ..., gt=0, description="Ширина лепестка в см", example=0.2
    )


# Схема исходящего ответа
class IrisResponseData(BaseModel):
    predicted_class: int = Field(
        ..., description="Индекс предсказанного класса (0, 1 или 2)"
    )
    probabilities: List[float] = Field(
        ..., description="Вектор вероятностей по классам"
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

    print("Инициализация сервера: загрузка графа ONNX в оперативную память...")
    onnx_session = ort.InferenceSession("iris_model.onnx")

    yield

    print("Остановка сервера: освобождение ресурсов памяти...")
    onnx_session = None


# Создание приложения FastAPI
app = FastAPI(lifespan=lifespan)


# Эндпоинт предсказания
@app.post("/v1/predict", response_model=IrisResponseData)
def predict(data: IrisRequestData):
    input_array = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]], dtype=np.float32)

    input_name = onnx_session.get_inputs()[0].name
    raw_predictions = onnx_session.run(None, {input_name: input_array})

    predicted_class = int(raw_predictions[0][0])
    probabilities = list(raw_predictions[1][0].values())

    return {
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "model_status": "iris_onnx_v1.0"
    }
