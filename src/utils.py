import os
import sys
import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        joblib.dump(obj, file_path)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Returns:
        report (dict): Performance metrics of each model.
    """
    try:
        report = {}
        trained_models = {}

        for name, model in models.items():

            gs = GridSearchCV(
                estimator=model,
                param_grid=param[name],
                cv=5,
                scoring="f1",
                n_jobs=-1
            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            trained_models[name] = best_model

            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]

            report[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob)
            }

        return report,trained_models

    except Exception as e:
        raise CustomException(e, sys)