from pathlib import Path
import pandas as pd


# Finding the root GestureLink directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Location of the raw dataset created by dataset_collector.py.
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hand_landmarks.csv"
)


def inspect_dataset() -> None:

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found at: {DATA_PATH}"
        )

    # Loading the CSV into a Pandas DataFrame.
    data = pd.read_csv(DATA_PATH)

    # Beautification 
    print()
    print("=" * 60)
    print("GESTURELINK DATASET REPORT")
    print("=" * 60)

    # Number of rows = number of collected hand samples.
    print(f"\nTotal samples: {len(data)}")

    # Our current format should contain:
    # 5 metadata columns + (21 landmarks × 3 coordinates) = 68.
    print(f"Total columns: {len(data.columns)}")

    # Counting how many independent recording sessions exist.
    print(
        f"Recording sessions: "
        f"{data['session_id'].nunique()}"
    )

    # Counting how many different users contributed.
    print(
        f"Users: "
        f"{data['user_id'].nunique()}"
    )

    print("\n--- Samples by class ---")

    # Checking whether open_palm, fist, and other are balanced.
    print(
        data["label"]
        .value_counts()
        .sort_index()
    )

    print("\n--- Samples by session ---")

    # Shows how many examples came from each recording session.
    print(
        data["session_id"]
        .value_counts()
        .sort_index()
    )

    print("\n--- Class distribution inside each session ---")

    # Very important: each session should ideally contain examples of all classes.
    print(
        pd.crosstab(
            data["session_id"],
            data["label"],
        )
    )

    print("\n--- Handedness ---")

    print(
        data["handedness"]
        .value_counts(dropna=False)
    )

    print("\n--- Missing values ---")

    missing_values = data.isna().sum().sum()

    print(
        f"Total missing values: "
        f"{missing_values}"
    )

    print("\n--- Duplicate rows ---")

    # Exact duplicates are not necessarily disastrous, but lots of them can indicate unnecessarily repeated samples.
    duplicate_count = data.duplicated().sum()

    print(
        f"Exact duplicate rows: "
        f"{duplicate_count}"
    )

    print("\n--- Labels found ---")

    print(
        sorted(
            data["label"]
            .dropna()
            .unique()
        )
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    inspect_dataset()