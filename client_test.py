
import time
import requests

API_URL = "http://127.0.0.1:8000/v1/predict"

# Тестовый набор признаков WineQT
payload = {
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


def send_request():

    start_time = time.time()

    try:
        # Отправляем POST-запрос
        response = requests.post(
            API_URL,
            json=payload,
            timeout=10
        )

        # Вычисляем задержку в миллисекундах
        latency = (time.time() - start_time) * 1000

        if response.status_code == 200:

            data = response.json()

            print(
                f"[УСПЕХ] Статус: {response.status_code} | "
                f"Задержка (Latency): {latency:.2f} мс"
            )

            print(
                f"Предсказанное качество вина: "
                f"{data['predicted_class']} | "
                f"Статус модели: {data['model_status']}"
            )

            print(
                f"Вероятности классов: "
                f"{data['probabilities']}\n"
            )

        else:
            print(
                f"[ОШИБКА] Статус: {response.status_code} | "
                f"Ответ: {response.text}\n"
            )

    except requests.exceptions.RequestException as e:

        print(f"[ОШИБКА СОЕДИНЕНИЯ] {e}\n")


if __name__ == "__main__":

    print(
        "Старт автоматического тестирования "
        "ИИ-микросервиса WineQT...\n"
    )

    for i in range(5):

        print(f"Отправка запроса №{i + 1}...")

        send_request()

        time.sleep(0.5)

    print("Тестирование микросервиса завершено!")
