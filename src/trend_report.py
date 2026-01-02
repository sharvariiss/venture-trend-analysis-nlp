import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


IN_PATH = Path("data/processed/with_topics.csv")
OUT_CSV = Path("reports/topic_trends_by_year.csv")
OUT_FIG = Path("reports/figures/topic_trends.png")


def main():
    if not IN_PATH.exists():
        print(f"[ERROR] Missing file: {IN_PATH}")
        print("Run: python src/topic_model.py")
        sys.exit(1)

    df = pd.read_csv(IN_PATH)

    if "founded_year" not in df.columns:
        print("[WARN] founded_year not found in processed data. Trend report skipped.")
        return

    df["founded_year"] = pd.to_numeric(df["founded_year"], errors="coerce")
    df = df.dropna(subset=["founded_year", "topic_id"])
    df["founded_year"] = df["founded_year"].astype(int)

    # Keep a reasonable year range
    df = df[(df["founded_year"] >= 1995) & (df["founded_year"] <= 2026)]

    pivot = (
        df.pivot_table(index="founded_year", columns="topic_id", values="text", aggfunc="count")
        .fillna(0)
        .astype(int)
    )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(OUT_CSV)

    # Plot top 5 topics overall
    top_topics = pivot.sum(axis=0).sort_values(ascending=False).head(5).index.tolist()

    plt.figure()
    pivot[top_topics].plot()
    plt.xlabel("founded_year")
    plt.ylabel("count")
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=200)
    plt.close()

    print("[OK] Saved:", OUT_CSV)
    print("[OK] Saved figure:", OUT_FIG)


if __name__ == "__main__":
    main()
