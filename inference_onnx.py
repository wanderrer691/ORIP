import numpy as np
import onnxruntime as ort


def main():
    print("Инициализация высокопроизводительного движка ONNX Runtime...")

    # 1. Загружаем файл чертежа модели в C++ рантайм
    session = ort.InferenceSession("iris_model.onnx")

    # Извлекаем метаданные о том, каких входных данных ожидает модель
    input_name = session.get_inputs()[0].name
    print(f"Модель ожидает входной тензор с именем: '{input_name}'")

    # 2. Подготовим новые тестовые данные для предсказания (вектор из 4 признаков)
    # ВАЖНО: тип данных должен быть строго float32, так как граф ONNX жестко типизирован!
    dummy_input = np.array(
        [[0.5, 0.2, 0.8, 0.1], [0.1, 0.9, 0.4, 0.7]],
        dtype=np.float32
    )
    print(f"Входной вектор данных:\n{dummy_input}")

    # 3. Запускаем инференс (вычисление по графу DAG)
    # Передаем словарь, где ключ – имя входного узла, значение – массив данных
    raw_predictions = session.run(None, {input_name: dummy_input})

    # Разбираем выходные тензоры (первый элемент – предсказанные классы)
    predicted_classes = raw_predictions[0]
    probabilities = raw_predictions[1]  # Второй элемент (для классификаторов) – вероятности классов

    print("\nРезультаты инференса от ONNX Runtime:")
    print(f"Предсказанные классы для объектов: {predicted_classes}")
    print(f"Матрица вероятностей классов:\n{probabilities}")


if __name__ == "__main__":
    main()