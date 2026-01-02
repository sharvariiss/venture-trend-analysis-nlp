import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


IN_PATH = Path("data/processed/clean_text.csv")
RESULTS_MD = Path("reports/results.md")
FIG_PATH = Path("reports/figures/topic_distribution.png")
MODEL_PATH = Path("models/topic_model.joblib")

N_TOPICS = 10
TOP_WORDS = 12


def main():
    if not IN_PATH.exists():
        print(f"[ERROR] Missing processed file: {IN_PATH}")
        print("Run: python src/preprocess.py")
        sys.exit(1)

    df = pd.read_csv(IN_PATH)
    texts = df["text"].fillna("").astype(str)

    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=5
    )
    X = vectorizer.fit_transform(texts)

    # Topic model
    nmf = NMF(n_components=N_TOPICS, random_state=42, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)        # doc-topic weights
    H = nmf.components_             # topic-term weights

    # Topic labels for each doc
    topic_id = W.argmax(axis=1)
    df["topic_id"] = topic_id
    out_topics = df.groupby("topic_id").size().sort_values(ascending=False)

    # Extract top words per topic
    feature_names = vectorizer.get_feature_names_out()
    topic_words = []
    for t in range(N_TOPICS):
        top_idx = H[t].argsort()[::-1][:TOP_WORDS]
        words = [feature_names[i] for i in top_idx]
        topic_words.append(words)

    # Write report
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write("# Venture Trend Analysis (Topic Modeling)\n\n")
        f.write(f"**Model:** TF-IDF + NMF  \n")
        f.write(f"**Topics:** {N_TOPICS}  \n")
        f.write(f"**Documents:** {len(df)}  \n\n")

        f.write("## Topic summaries (top keywords)\n\n")
        for t, words in enumerate(topic_words):
            f.write(f"### Topic {t}\n")
            f.write(", ".join(words) + "\n\n")

        f.write("## Topic distribution\n\n")
        f.write(out_topics.to_frame("count").to_markdown() + "\n")

    # Plot distribution
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    out_topics.sort_index().plot(kind="bar")
    plt.xlabel("topic_id")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=200)
    plt.close()

    # Save model artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "nmf": nmf, "n_topics": N_TOPICS, "top_words": TOP_WORDS},
        MODEL_PATH
    )

    # Save enriched processed file (optional, useful)
    df.to_csv("data/processed/with_topics.csv", index=False)

    print("[OK] Wrote:", RESULTS_MD)
    print("[OK] Saved figure:", FIG_PATH)
    print("[OK] Saved model:", MODEL_PATH)
    print("[OK] Saved:", "data/processed/with_topics.csv")


if __name__ == "__main__":
    main()
