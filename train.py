import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

# 1. Настройка подключения к серверу MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 2. Создание или выбор эксперимента
experiment_name = "iris_classification_experiment"
mlflow.set_experiment(experiment_name)


def main():
    # Задаем гиперпараметры
    n_estimators = 150
    max_depth = 15
    random_state = 42

    # Запуск эксперимента
    with mlflow.start_run(run_name="RandomForest_Run"):

        # --- Симуляция загрузки данных ---
        # В реальной работе данные читались бы из data/dataset.csv,
        # версия которого контролируется DVC

        np.random.seed(random_state)

        X = np.random.rand(200, 4)
        y = np.random.choice([0, 1], size=200)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=random_state
        )

        # --- Логирование параметров ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("dataset_version", "v2.0")

        # --- Обучение модели ---
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

        model.fit(X_train, y_train)

        # --- Оценка модели ---
        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="macro")

        # Логирование метрик
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Симуляция логирования по эпохам
        for epoch in range(1, 6):
            fake_loss = 0.5 / epoch
            mlflow.log_metric(
                "train_loss",
                fake_loss,
                step=epoch
            )

        print(
            f"Эксперимент завершен. Accuracy: {acc:.4f}, F1: {f1:.4f}"
        )

        # --- Создание графика ---
        os.makedirs("plots", exist_ok=True)

        plt.figure(figsize=(5, 5))
        plt.title("Feature Importances")

        plt.bar(
            range(X.shape[1]),
            model.feature_importances_
        )

        plt.savefig("plots/feature_importance.png")
        plt.close()

        # Логирование артефакта
        mlflow.log_artifact(
            "plots/feature_importance.png"
        )

        # --- Сохранение модели ---
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        


if __name__ == "__main__":
    main()