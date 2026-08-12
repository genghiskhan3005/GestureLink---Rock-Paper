"""Evaluating GestureLink models on completely unseen users."""

from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Locating the root GestureLink directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Specifying the input path for the processed feature dataset
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
)


# Defining metadata column names that are excluded from model inputs
NON_FEATURE_COLUMNS = {
    "label",
    "session_id",
    "user_id",
    "handedness",
}


def load_dataset():
    """Loading the processed dataset and separating features from labels."""

    # Verifying that the dataset file exists on the filesystem
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {DATA_PATH}"
        )

    # Reading processed dataset samples into a pandas DataFrame
    data = pd.read_csv(DATA_PATH)

    # Filtering out metadata columns to isolate numerical features
    feature_columns = [
        column
        for column in data.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    # Extracting feature inputs for model training and testing
    X = data[feature_columns]

    # Extracting target pose labels for prediction
    y = data["label"]

    return data, X, y


def build_models():
    """Creating candidate classical ML models for evaluation."""

    return {
        "logistic_regression": Pipeline(
            [
                # Scaling features and configuring Logistic Regression
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),

        "knn": Pipeline(
            [
                # Scaling features and configuring K-Nearest Neighbors
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=7,
                    ),
                ),
            ]
        ),

        "svm": Pipeline(
            [
                # Scaling features and configuring Support Vector Machine with an RBF kernel
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),

        # Instantiating a Random Forest classifier directly without scaling
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_user_split(
    model_name: str,
    model,
    X,
    y,
    data,
    training_user: str,
    testing_user: str,
) -> dict:
    """Training on one user and evaluating on a completely unseen user."""

    # Filtering samples belonging to the training participant
    train_mask = (
        data["user_id"]
        == training_user
    )

    # Filtering samples belonging to the unseen test participant
    test_mask = (
        data["user_id"]
        == testing_user
    )

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    # Fitting the model pipeline using training participant data
    model.fit(
        X_train,
        y_train,
    )

    # Predicting target labels for the unseen test participant
    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    # Calculating accuracy and unweighted Macro F1 score across classes
    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    print()
    print("-" * 60)

    print(
        f"Model: {model_name}"
    )

    print(
        f"Train user: {training_user}"
    )

    print(
        f"Test user: {testing_user}"
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Macro F1: {macro_f1:.4f}"
    )

    print("\nClassification report:")

    # Outputting performance evaluation metrics and confusion matrix
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    return {
        "model": model_name,
        "training_user": training_user,
        "testing_user": testing_user,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }


def evaluate_users() -> None:
    """Running cross-user evaluation experiments for all candidate models."""

    data, X, y = load_dataset()

    # Extracting and sorting unique participant identifiers
    users = sorted(
        data["user_id"].unique()
    )

    # Validating that exactly two participant IDs exist in the dataset
    if len(users) != 2:
        raise ValueError(
            "This experiment currently expects exactly 2 users. "
            f"Found: {users}"
        )

    user_a = users[0]
    user_b = users[1]

    print()
    print("=" * 60)
    print("GESTURELINK UNSEEN-USER EVALUATION")
    print("=" * 60)

    print(
        f"\nUsers: {user_a}, {user_b}"
    )

    print("\nSamples per user:")

    print(
        data["user_id"]
        .value_counts()
    )

    results = []

    # Executing Experiment 1: training on User A and testing on User B
    print()
    print("=" * 60)
    print(
        f"EXPERIMENT 1: {user_a} -> {user_b}"
    )
    print("=" * 60)

    for model_name, model in build_models().items():

        result = evaluate_user_split(
            model_name=model_name,
            model=model,
            X=X,
            y=y,
            data=data,
            training_user=user_a,
            testing_user=user_b,
        )

        results.append(result)

    # Executing Experiment 2: reversing direction by training on User B and testing on User A
    print()
    print("=" * 60)
    print(
        f"EXPERIMENT 2: {user_b} -> {user_a}"
    )
    print("=" * 60)

    for model_name, model in build_models().items():

        result = evaluate_user_split(
            model_name=model_name,
            model=model,
            X=X,
            y=y,
            data=data,
            training_user=user_b,
            testing_user=user_a,
        )

        results.append(result)

    # Displaying cross-user evaluation performance summary
    print()
    print("=" * 60)
    print("UNSEEN-USER SUMMARY")
    print("=" * 60)

    for result in results:

        print(
            f"{result['model']:20s} | "
            f"{result['training_user']} "
            f"-> {result['testing_user']} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"Macro F1: {result['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    # Executing cross-user model evaluation pipeline
    evaluate_users()