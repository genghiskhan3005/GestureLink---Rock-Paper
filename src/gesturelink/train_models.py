from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Locating the project root path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Setting the input path for the processed feature dataset
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features.csv"
)

# Establishing the directory path for storing trained models
MODEL_DIR = PROJECT_ROOT / "models"

# Specifying the output path for the saved model file
MODEL_PATH = (
    MODEL_DIR
    / "pose_classifier.joblib"
)


# Defining metadata column names that are excluded from model inputs
NON_FEATURE_COLUMNS = {
    "label",
    "session_id",
    "user_id",
    "handedness",
}


def load_dataset():
    """Loading the processed GestureLink dataset."""

    # Verifying that the dataset file exists on the filesystem
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at: {DATA_PATH}"
        )

    # Reading the tabular CSV data into a pandas DataFrame
    data = pd.read_csv(DATA_PATH)

    # Filtering out metadata columns to isolate numerical features
    feature_columns = [
        column
        for column in data.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    # Extracting feature inputs and ground-truth target labels
    X = data[feature_columns]
    y = data["label"]

    return data, X, y, feature_columns


def build_models():
    """Creating candidate classical ML models."""

    # Constructing candidate classification pipelines and models
    return {
        # Scaling features and configuring Logistic Regression
        "logistic_regression": Pipeline(
            [
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

        # Scaling features and configuring K-Nearest Neighbors
        "knn": Pipeline(
            [
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

        # Scaling features and configuring Support Vector Machine with an RBF kernel
        "svm": Pipeline(
            [
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


def evaluate_predictions(
    y_true,
    predictions,
) -> tuple[float, float]:
    """Calculating accuracy and macro-F1."""

    # Calculating overall classification accuracy score
    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    # Calculating unweighted macro F1 score across all target classes
    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
    )

    return accuracy, macro_f1


def cross_validate_models(
    models,
    X_train,
    y_train,
    training_groups,
):
    """
    Comparing models using entire sessions as validation groups.

    LeaveOneGroupOut means one complete recording session
    becomes validation data while the remaining sessions
    are used for training.
    """

    # Initializing LeaveOneGroupOut cross-validation across sessions
    logo = LeaveOneGroupOut()

    results = []

    print()
    print("=" * 60)
    print("GROUPED CROSS-VALIDATION")
    print("=" * 60)

    # Iterating over each candidate model pipeline
    for model_name, model in models.items():

        fold_scores = []

        print(
            f"\nModel: {model_name}"
        )

        fold_number = 1

        # Iterating through grouped cross-validation folds
        for train_indices, validation_indices in logo.split(
            X_train,
            y_train,
            groups=training_groups,
        ):
            # Extracting feature training fold data
            X_fold_train = X_train.iloc[
                train_indices
            ]

            # Extracting target training fold data
            y_fold_train = y_train.iloc[
                train_indices
            ]

            # Extracting feature validation fold data
            X_fold_validation = X_train.iloc[
                validation_indices
            ]

            # Extracting target validation fold data
            y_fold_validation = y_train.iloc[
                validation_indices
            ]

            # Identifying current holdout validation session ID
            validation_session = (
                training_groups.iloc[
                    validation_indices
                ]
                .unique()
                .tolist()
            )

            # Fitting model pipeline on training fold data
            model.fit(
                X_fold_train,
                y_fold_train,
            )

            # Predicting target labels for validation fold data
            predictions = model.predict(
                X_fold_validation
            )

            # Computing evaluation metrics for current fold
            accuracy, macro_f1 = (
                evaluate_predictions(
                    y_fold_validation,
                    predictions,
                )
            )

            # Aggregating Macro F1 scores across folds
            fold_scores.append(
                macro_f1
            )

            print(
                f"  Fold {fold_number} | "
                f"Validation session: "
                f"{validation_session} | "
                f"Accuracy: {accuracy:.4f} | "
                f"Macro F1: {macro_f1:.4f}"
            )

            fold_number += 1

        # Calculating mean Macro F1 score across cross-validation folds
        mean_macro_f1 = float(
            np.mean(fold_scores)
        )

        # Calculating standard deviation of Macro F1 scores across folds
        std_macro_f1 = float(
            np.std(fold_scores)
        )

        print(
            f"  Mean Macro F1: "
            f"{mean_macro_f1:.4f}"
        )

        print(
            f"  Std Macro F1: "
            f"{std_macro_f1:.4f}"
        )

        # Storing aggregated cross-validation performance metrics
        results.append(
            {
                "name": model_name,
                "model": model,
                "mean_macro_f1": mean_macro_f1,
                "std_macro_f1": std_macro_f1,
            }
        )

    # Sorting validation results based on mean Macro F1 score in descending order
    results.sort(
        key=lambda result: result[
            "mean_macro_f1"
        ],
        reverse=True,
    )

    return results


def evaluate_final_model(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """Training the selected model and evaluate once on final test data."""

    print()
    print("=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    print(
        f"\nSelected model: {model_name}"
    )

    # Retraining selected model pipeline on full training set
    model.fit(
        X_train,
        y_train,
    )

    # Generating predictions for held-out final test set
    predictions = model.predict(
        X_test
    )

    # Computing accuracy and Macro F1 score on final test set
    accuracy, macro_f1 = evaluate_predictions(
        y_test,
        predictions,
    )

    print(
        f"\nAccuracy: {accuracy:.4f}"
    )

    print(
        f"Macro F1: {macro_f1:.4f}"
    )

    print("\nClassification report:")

    # Displaying classification metrics and confusion matrix
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
        )
    )

    print("Confusion matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    return accuracy, macro_f1


def train_models() -> None:
    """Running grouped model selection and final testing."""

    # Loading full dataset, feature columns, and target labels
    data, X, y, feature_columns = (
        load_dataset()
    )

    # Extracting and sorting unique session identifiers
    sessions = sorted(
        data["session_id"].unique()
    )

    # Validating minimum session count requirements
    if len(sessions) < 4:
        raise ValueError(
            "At least 4 sessions are required."
        )

    # Isolating the final session as an untouched test set
    final_test_session = sessions[-1]

    # Selecting remaining sessions for cross-validation training
    training_sessions = sessions[:-1]

    # Generating boolean mask for training sessions
    train_mask = data[
        "session_id"
    ].isin(training_sessions)

    # Generating boolean mask for final test session
    test_mask = (
        data["session_id"]
        == final_test_session
    )

    # Extracting training features and resetting index
    X_train = X.loc[
        train_mask
    ].reset_index(drop=True)

    # Extracting training target labels and resetting index
    y_train = y.loc[
        train_mask
    ].reset_index(drop=True)

    # Extracting training group session identifiers and resetting index
    training_groups = data.loc[
        train_mask,
        "session_id",
    ].reset_index(drop=True)

    # Extracting test features and resetting index
    X_test = X.loc[
        test_mask
    ].reset_index(drop=True)

    # Extracting test target labels and resetting index
    y_test = y.loc[
        test_mask
    ].reset_index(drop=True)

    print()
    print("=" * 60)
    print("GESTURELINK ML TRAINING")
    print("=" * 60)

    print(
        f"\nTotal samples: {len(data)}"
    )

    print(
        f"Features: {len(feature_columns)}"
    )

    print(
        f"Training sessions: "
        f"{training_sessions}"
    )

    print(
        f"Untouched test session: "
        f"{final_test_session}"
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Final test samples: "
        f"{len(X_test)}"
    )

    # Constructing candidate models
    models = build_models()

    # Executing grouped cross-validation on candidate models
    validation_results = (
        cross_validate_models(
            models=models,
            X_train=X_train,
            y_train=y_train,
            training_groups=training_groups,
        )
    )

    print()
    print("=" * 60)
    print("MODEL SELECTION")
    print("=" * 60)

    # Outputting model selection leaderboard
    for rank, result in enumerate(
        validation_results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{result['name']} | "
            f"Mean Macro F1: "
            f"{result['mean_macro_f1']:.4f} | "
            f"Std: "
            f"{result['std_macro_f1']:.4f}"
        )

    # Extracting top-performing model name from validation results
    best_result = validation_results[0]

    best_model_name = best_result["name"]

    # Instantiating fresh model instance for final training
    best_model = build_models()[
        best_model_name
    ]

    # Evaluating final model performance on untouched test set
    accuracy, macro_f1 = (
        evaluate_final_model(
            model_name=best_model_name,
            model=best_model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
    )

    # Creating target directory for saving model file if needed
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Constructing model bundle with model estimator, name, and column metadata
    model_bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "feature_columns": feature_columns,
        "validation_macro_f1": (
            best_result[
                "mean_macro_f1"
            ]
        ),
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
    }

    # Serializing model bundle to disk using joblib
    joblib.dump(
        model_bundle,
        MODEL_PATH,
    )

    print()
    print(
        f"Model saved to: "
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    # Executing the complete model training and evaluation workflow
    train_models()



"""
Why train_models.py was rewritten and why evaluate_users.py was created
======================================================================

We wrote train_models.py more than once because each version answered a
different machine-learning question and improved the reliability of our
evaluation.

1. First version of train_models.py
-----------------------------------

The first version was used to check whether our 73 engineered features
could successfully classify the three gesture classes:

    - fist
    - open_palm
    - other

We trained four classical machine-learning models:

    - Logistic Regression
    - K-Nearest Neighbors (KNN)
    - Support Vector Machine (SVM)
    - Random Forest

The first experiment used three recording sessions for training and one
recording session for testing.

This allowed us to answer:

    "Can our engineered features actually be used to recognize gestures?"

It also allowed us to compare the four models.

Random Forest performed best, with approximately:

    Accuracy: 0.7857
    Macro F1: 0.7846


2. Why train_models.py was improved
-----------------------------------

There was one important problem with the first experiment.

We used the same test session to:

    1. compare all four models
    2. choose the best model
    3. report the final performance

In proper machine learning, the final test set should ideally remain
untouched until after the model has already been selected.

Otherwise, information from the test set indirectly influences our model
choice.

To improve this, we rewrote train_models.py and introduced grouped
cross-validation.


3. Grouped cross-validation
---------------------------

Because our dataset was collected in separate recording sessions, we did
not want to randomly split individual frames.

Frames recorded next to each other are often very similar. Randomly
placing some of those frames in training and others in testing could make
the model appear more accurate than it really is.

Instead, we treated complete sessions as groups.

For example:

    Fold 1:
        Train on Sessions 2 and 3
        Validate on Session 1

    Fold 2:
        Train on Sessions 1 and 3
        Validate on Session 2

    Fold 3:
        Train on Sessions 1 and 2
        Validate on Session 3

The final fourth session remained completely untouched during model
selection.

This allowed us to answer:

    "Which model generalizes best to a new recording session?"

Random Forest again performed best.

Its grouped cross-validation result was approximately:

    Mean Macro F1: 0.7196

After Random Forest was selected using validation data, we trained it
again using all three training sessions and evaluated it once on the
untouched fourth session.

Final result:

    Accuracy: 0.7857
    Macro F1: 0.7846


4. Why evaluate_users.py was created
------------------------------------

After improving the session-based evaluation, we noticed another
limitation.

Our dataset contained two real users:

    labib_109
    rafsan_137

In the final session experiment, the training data contained:

    Labib Session 1
    Labib Session 2
    Rafsan Session 1

while the test data contained:

    Rafsan Session 2

Therefore, the test session was new, but the test person was not
completely new.

The model had already seen examples of Rafsan's hand during training.

This meant our experiment answered:

    "Can GestureLink recognize gestures in a new recording session?"

but it did not completely answer:

    "Can GestureLink recognize gestures from a person it has never seen?"


5. Purpose of evaluate_users.py
-------------------------------

We created evaluate_users.py to test true unseen-user generalization.

We performed two experiments.

Experiment 1:

    Train only on Labib
    Test only on Rafsan

Experiment 2:

    Train only on Rafsan
    Test only on Labib

In both cases, the test person's data was completely excluded from
training.

This is a much harder and more realistic test because the classifier must
recognize gestures from a new person's hand.


6. Unseen-user results
----------------------

Random Forest again performed best.

Labib -> Rafsan:

    Accuracy: 0.6993
    Macro F1: 0.6742

Rafsan -> Labib:

    Accuracy: 0.6920
    Macro F1: 0.6851

These results show that Random Forest does not only work on recording
sessions similar to the training data.

It can also recognize gestures from a person whose hand was never
included in the training dataset, although performance naturally drops
compared with the easier same-user/new-session experiment.


7. Overall reason for these experiments
---------------------------------------

The experiments became progressively more difficult:

    Stage 1:
        Can our features classify gestures at all?

        -> Initial train_models.py

    Stage 2:
        Can the models generalize to a new recording session without
        using the final test data for model selection?

        -> Improved train_models.py with grouped cross-validation

    Stage 3:
        Can the model recognize gestures from a completely unseen user?

        -> evaluate_users.py


This progression gives us a much more realistic evaluation of
GestureLink.

Instead of reporting only one accuracy score, we tested:

    - basic classification performance
    - session generalization
    - grouped validation
    - unseen-user generalization
    - multiple classical ML algorithms

The results consistently showed that Random Forest was the strongest
model among the four tested algorithms, so it became our first candidate
for live GestureLink gesture recognition.
"""