
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn


# 1. Настройка подключения к серверу MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")


# 2. Создание или выбор эксперимента
experiment_name = "wine_quality_classification_experiment"
mlflow.set_experiment(experiment_name)


def main():

    # Задаем гиперпараметры
    n_estimators = 150
    max_depth = 15
    random_state = 42

    # Путь к датасету
    dataset_path = "data/WineQT.csv"

    # Запуск эксперимента
    with mlflow.start_run(run_name="RandomForest_WineQT"):

        # --- Загрузка данных ---
        df = pd.read_csv(dataset_path)

        print("Первые строки датасета:")
        print(df.head())

        print("\nРазмер датасета:")
        print(df.shape)

        print("\nИнформация о датасете:")
        print(df.info())

        # --- Подготовка данных ---

        # Удаляем идентификатор, если он присутствует
        if "Id" in df.columns:
            df = df.drop(columns=["Id"])

        # Удаляем строки с пропущенными значениями
        df = df.dropna()

        # Целевая переменная
        X = df.drop(columns=["quality"])
        y = df["quality"]

        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=random_state,
            stratify=y
        )

        # --- Логирование параметров ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("dataset", "WineQT")
        mlflow.log_param("dataset_version", "v1.0")
        mlflow.log_param("number_of_features", X.shape[1])
        mlflow.log_param("number_of_samples", X.shape[0])

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
        f1_macro = f1_score(
            y_test,
            predictions,
            average="macro"
        )
        f1_weighted = f1_score(
            y_test,
            predictions,
            average="weighted"
        )

        # Логирование метрик
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("f1_weighted", f1_weighted)

        print("\nРезультаты классификации:")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, predictions))

        # --- Симуляция логирования по эпохам ---
        # Random Forest не обучается по эпохам.
        # Здесь сохраняется условная динамика для демонстрации MLflow.
        for epoch in range(1, 6):
            fake_loss = 0.5 / epoch

            mlflow.log_metric(
                "train_loss",
                fake_loss,
                step=epoch
            )

        # --- Создание папки для графиков ---
        os.makedirs("plots", exist_ok=True)

        # --- График важности признаков ---
        plt.figure(figsize=(10, 6))

        feature_importances = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=True)

        feature_importances.plot(
            kind="barh",
            title="WineQT Feature Importances"
        )

        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()

        plt.savefig("plots/feature_importance.png")
        plt.close()

        # Логирование графика в MLflow
        mlflow.log_artifact(
            "plots/feature_importance.png"
        )

        # --- Матрица ошибок ---
        plt.figure(figsize=(8, 6))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            cmap="Blues"
        )

        plt.title("WineQT Confusion Matrix")
        plt.tight_layout()

        plt.savefig("plots/confusion_matrix.png")
        plt.close()

        mlflow.log_artifact(
            "plots/confusion_matrix.png"
        )

        # --- Сохранение модели в MLflow ---
        mlflow.sklearn.log_model(
            model,
            name="random_forest_model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        # --- Итог ---
        print("\nЭксперимент успешно завершен!")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")


if __name__ == "__main__":
    main()
