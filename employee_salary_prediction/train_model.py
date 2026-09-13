import json
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessing_pipeline,
    get_engineered_feature_names,
    prepare_features_and_target,
)

DATA_PATH = os.path.join("data", "employee_salary.csv")
MODEL_PATH = os.path.join("models", "salary_prediction_model.pkl")
METRICS_PATH = os.path.join("models", "model_metrics.json")
RANDOM_STATE = 42

def load_dataset():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run data/generate_dataset.py first.")
    return pd.read_csv(DATA_PATH)

def build_candidate_models():
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=14, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=250, max_depth=3, learning_rate=0.08, random_state=RANDOM_STATE),
    }

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {"mae": round(mae, 2), "mse": round(mse, 2), "rmse": round(rmse, 2), "r2": round(r2, 4)}

def resolve_output_feature_groups(preprocessor):
    output_names = preprocessor.get_feature_names_out()
    groups = []
    for name in output_names:
        if name.startswith("numeric__"):
            groups.append(name.replace("numeric__", ""))
        elif name.startswith("categorical__"):
            remainder = name.replace("categorical__", "")
            matched = next((column for column in CATEGORICAL_FEATURES if remainder.startswith(column)), remainder)
            groups.append(matched)
        elif name.startswith("engineered__"):
            groups.append("skills")
        else:
            groups.append(name)
    return groups

def compute_feature_importance(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    groups = resolve_output_feature_groups(preprocessor)

    if hasattr(model, "feature_importances_"):
        raw_importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw_importances = abs(model.coef_)
    else:
        return {}

    aggregated = {}
    for group_name, importance in zip(groups, raw_importances):
        aggregated[group_name] = aggregated.get(group_name, 0.0) + float(importance)

    total = sum(aggregated.values())
    if total <= 0:
        return {}

    normalized = {name: round(value / total, 4) for name, value in aggregated.items()}
    return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True))

def main():
    print("Loading dataset")
    raw_dataframe = load_dataset()
    print(f"Raw records: {len(raw_dataframe)}")

    features, target = prepare_features_and_target(raw_dataframe)
    print(f"Records after cleaning: {len(features)}")

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessing_pipeline()
    candidates = build_candidate_models()

    results = {}
    fitted_pipelines = {}

    for name, model in candidates.items():
        print(f"Training {name}")
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = evaluate_predictions(y_test, predictions)
        print(f"{name} -> {metrics}")
        results[name] = metrics
        fitted_pipelines[name] = pipeline

    best_model_name = max(results, key=lambda name: results[name]["r2"])
    best_pipeline = fitted_pipelines[best_model_name]
    print(f"Best model: {best_model_name}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    feature_importance = compute_feature_importance(best_pipeline)

    metrics_output = {
        "best_model": best_model_name,
        "models": results,
        "training_records": int(len(x_train)),
        "testing_records": int(len(x_test)),
        "feature_importance": feature_importance,
    }
    with open(METRICS_PATH, "w") as metrics_file:
        json.dump(metrics_output, metrics_file, indent=2)

    print(f"Saved pipeline to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")

if __name__ == "__main__":
    main()
