import os
import pandas as pd

INPUT_CSV = "data/full_dataset.csv"
OUTPUT_CSV = "data/small_dataset.csv"
N = 5000
SEED = 42


def main() -> None:
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    n_sample = min(N, len(df))
    small = df.sample(n=n_sample, random_state=SEED)

    small.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {n_sample} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
