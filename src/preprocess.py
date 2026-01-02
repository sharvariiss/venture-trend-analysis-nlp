import re
import sys
from pathlib import Path

import pandas as pd


RAW_PATH = Path("data/raw/startup_investments.csv")
OUT_PATH = Path("data/processed/clean_text.csv")


def _safe_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name from candidates (case-insensitive)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def clean_text(s: str) -> str:
    s = s.lower()
    s = s.replace(",", " ")
    s = s.replace("/", " ")
    s = re.sub(r"[^a-z0-9\s\-\+]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    if not RAW_PATH.exists():
        print(f"[ERROR] Raw file not found: {RAW_PATH}")
        print("Place your Kaggle CSV at data/raw/startup_investments.csv")
        sys.exit(1)

    df = pd.read_csv(RAW_PATH, encoding="latin1", low_memory=False)


    cat_col = _safe_col(df, ["category_list", "categories", "category"])
    market_col = _safe_col(df, ["market", "markets", "sector"])
    year_col = _safe_col(df, ["founded_year", "year_founded", "founded"])

    # Build text field (this is what we run NLP on)
    parts = []
    if cat_col:
        parts.append(df[cat_col].fillna("").astype(str))
    if market_col:
        parts.append(df[market_col].fillna("").astype(str))

    if not parts:
        print("[ERROR] Could not find 'category_list' or 'market' columns.")
        print("Available columns:", list(df.columns))
        sys.exit(1)

    text = parts[0]
    for p in parts[1:]:
        text = text + " " + p

    df_out = pd.DataFrame()
    df_out["text"] = text.map(clean_text)

    # Optional fields for trend analysis
    if year_col:
        df_out["founded_year"] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")

    # Keep a few identifiers if present (useful for demos)
    name_col = _safe_col(df, ["name", "company_name"])
    status_col = _safe_col(df, ["status"])
    if name_col:
        df_out["name"] = df[name_col].astype(str)
    if status_col:
        df_out["status"] = df[status_col].astype(str)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)
    print(f"[OK] Saved processed text dataset: {OUT_PATH} (rows={len(df_out)})")


if __name__ == "__main__":
    main()
