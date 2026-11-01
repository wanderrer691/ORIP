import time
import requests

API_URL = "http://127.0.0.1:8000/v1/predict"
# Тестовый набор признаков (эквивалентен схеме Pydantic)
payload = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
    }

def send_request():
    start_time = time.time()

# Отправляем POST-запрос, автоматически сериализуя словарь Python в JSON-текст

    response = requests.post(API_URL, json=payload)

    latency = (time.time() - start_time) * 1000 # Переводим секунды в миллисекунды
    if response.status_code == 200:
        data = response.json()
        print(f"[УСПЕХ] Статус: {response.status_code} | Задержка (Latency): {latency:.2f} мс")
        print(f" Предсказание модели: Класс {data['predicted_class']} | Статус: {data['model_status']}\n")
    else:
        print(f"[ОШИБКА] Статус: {response.status_code} | Ответ: {response.text}\n")


if __name__ == "__main__":
    print("Старт автоматического тестирования ИИ-микросервиса...\n")
    for i in range(5):
        print(f"Отправка запроса No{i+1}...") 
        send_request()
        time.sleep(0.5) # Небольшая пауза между запросами