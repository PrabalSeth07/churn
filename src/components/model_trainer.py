import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.neighbors import KNeighborsClassifier

# from sklearn.svm import SVC

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training input and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            RANDOM_STATE = 42

            models = {
                "Logistic Regression": LogisticRegression(
                    class_weight="balanced"
                ),

                "KNN": KNeighborsClassifier(),

                "Decision Tree": DecisionTreeClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced"
                ),

                "Random Forest": RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced"
                ),

                "AdaBoost": AdaBoostClassifier(
                    random_state=RANDOM_STATE
                ),

                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=RANDOM_STATE
                ),

                "XGBoost": XGBClassifier(
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    scale_pos_weight=2.77
                ),

                "CatBoost": CatBoostClassifier(
                    random_state=RANDOM_STATE,
                    verbose=False,
                    auto_class_weights="Balanced"
                )
            }

            params = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["liblinear", "lbfgs"]
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                },
                # "SVC": {
                #     "C": [0.1, 1, 10],
                #     "kernel": ["linear", "rbf"]
                # },
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20]
                },
                "AdaBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1, 1]
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5]
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 6]
                },
                # "XGBoost": {
                #     "n_estimators": [100, 200],
                #     "learning_rate": [0.05, 0.1],
                #     "max_depth": [3, 4, 6],
                #     "subsample": [0.8, 1.0],
                #     "colsample_bytree": [0.8, 1.0]
                # },
                "CatBoost": {
                    "iterations": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "depth": [4, 6]
                }
            }

            model_report,trained_models = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )
            best_model_name = max(
                model_report,
                key=lambda model: model_report[model]["f1"]
            )

            best_model = trained_models[best_model_name]
            best_model_score = model_report[best_model_name]["f1"]


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_prob)

            logging.info(
                f"Accuracy: {accuracy:.4f},"
                f"Precision: {precision:.4f},"
                f"Recall: {recall:.4f},"
                f"F1: {f1:.4f},"
                f"ROC-AUC: {roc_auc:.4f}"
            )

            return {
                    "best_model": best_model_name,
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "roc_auc": roc_auc
                }

            
        except Exception as e:
            raise CustomException(e,sys)


