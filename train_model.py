from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

FEATURES = [
    "IQ",
    "Prev_Sem_Result",
    "CGPA",
    "Academic_Performance",
    "Internship_Experience",
    "Extra_Curricular_Score",
    "Communication_Skills",
    "Projects_Completed",
]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("dataset.csv was not found beside train_model.py")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    required = FEATURES + ["Placement"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    df["Placement"] = df["Placement"].astype(str).str.strip().str.title().map({"Yes": 1, "No": 0})
    df["Internship_Experience"] = (
        df["Internship_Experience"].astype(str).str.strip().str.title().map({"Yes": 1, "No": 0})
    )
    for column in FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["Placement"]).copy()
    if df["Placement"].nunique() != 2:
        raise ValueError("Placement must contain both Yes and No examples.")
    for column in FEATURES:
        df[column] = df[column].fillna(df[column].median())

    X = df[FEATURES]
    y = df["Placement"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Dataset rows used: {len(df):,}")
    print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"5-fold CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"Holdout accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Confusion matrix (rows=actual, columns=predicted; order No, Yes):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("Classification report:")
    print(classification_report(y_test, predictions, labels=[0, 1], target_names=["Preparation Recommended", "Positive Outlook"], zero_division=0))

    classifier = model.named_steps["classifier"]
    importances = sorted(zip(FEATURES, classifier.feature_importances_), key=lambda item: item[1], reverse=True)
    print("Feature importance (association, not causation):")
    for name, score in importances:
        print(f"  {name}: {score:.3f}")

    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)
    print(f"Saved model to: {MODEL_PATH}")
    print("Note: strong test scores on synthetic/rule-based data do not prove real-world placement accuracy.")


if __name__ == "__main__":
    main()
