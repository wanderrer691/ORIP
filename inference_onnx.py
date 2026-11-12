
import numpy as np
import onnxruntime as ort
import pandas as pd


def main():

    print("Инициализация высокопроизводительного движка ONNX Runtime...")

    # 1. Загружаем ONNX-модель
    session = ort.InferenceSession(
        "wine_quality_model.onnx"
    )

    # Получаем информацию о входных данных модели
    input_name = session.get_inputs()[0].name

    print(
        f"Модель ожидает входной тензор "
        f"с именем: '{input_name}'"
    )

    # 2. Загружаем тестовые данные из WineQT
    df = pd.read_csv("data/WineQT.csv")

    # Удаляем Id, если он присутствует
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # Отделяем признаки от целевой переменной
    X = df.drop(columns=["quality"])
    y = df["quality"]

    # Берем два тестовых образца
    dummy_input = X.iloc[:2].to_numpy(dtype=np.float32)

    print("\nВходные данные для предсказания:")
    print(dummy_input)

    # 3. Запускаем инференс через ONNX Runtime
    raw_predictions = session.run(
        None,
        {input_name: dummy_input}
    )

    # 4. Получаем предсказанные классы
    predicted_classes = raw_predictions[0]

    print("\nРезультаты инференса от ONNX Runtime:")

    print(
        f"Предсказанные классы качества вина: "
        f"{predicted_classes}"
    )

    # 5. Выводим реальные и предсказанные значения
    print("\nСравнение предсказаний с реальными значениями:")

    for i in range(len(dummy_input)):
        print(
            f"Образец {i + 1}: "
            f"Реальное качество = {y.iloc[i]}, "
            f"Предсказанное качество = {predicted_classes[i]}"
        )

    # 6. Получаем вероятности классов
    if len(raw_predictions) > 1:

        probabilities = raw_predictions[1]

        print("\nВероятности предсказания классов:")
        print(probabilities)


if __name__ == "__main__":
    main()
