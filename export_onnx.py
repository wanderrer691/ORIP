
import mlflow
import mlflow.sklearn
import numpy as np
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType


# 1. Настройка подключения к серверу MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")


def main():

    # 2. URI модели из MLflow Model Registry
    # Укажи имя зарегистрированной модели и нужную стадию
    model_uri = "models:/wine_production_model/Staging"

    print(f"Загрузка модели из реестра: {model_uri}...")

    # 3. Загружаем обученную модель Random Forest
    loaded_model = mlflow.sklearn.load_model(model_uri)

    # 4. Определяем тип входных данных
    # WineQT содержит 11 признаков после удаления Id и quality
    initial_type = [
        ('float_input', FloatTensorType([None, 11]))
    ]

    print("Трансляция модели в формат ONNX...")

    # 5. Конвертируем модель Scikit-learn в ONNX
    onnx_model = to_onnx(
        loaded_model,
        initial_types=initial_type,
        target_opset=15
    )

    # 6. Сохраняем ONNX-файл
    onnx_filename = "wine_quality_model.onnx"

    with open(onnx_filename, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(
        f"Экспорт успешно завершен! "
        f"Файл сохранен как: '{onnx_filename}'"
    )


if __name__ == "__main__":
    main()
