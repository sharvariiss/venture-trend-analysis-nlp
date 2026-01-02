# Venture Trend Analysis using NLP (Crunchbase Data)

## Overview
This project applies **Natural Language Processing (NLP)** to Crunchbase-style startup investment data to identify **venture-relevant technology and sector trends**.  
The goal is not prediction, but **interpretable trend discovery** that supports **venture research and market intelligence**.

Using topic modeling, the project clusters startups into coherent sectors (e.g. Enterprise SaaS, FinTech, HealthTech, CleanTech) and analyzes their distribution and evolution.

---

## Why this matters for Venture Research
Early-stage venture decisions rely on understanding:
- where founder activity is concentrated
- which sectors show sustained momentum
- how technology themes evolve over time

This project demonstrates how **lightweight, interpretable NLP methods** can extract such insights from large-scale startup datasets.

---

## Dataset
- Source: Crunchbase-style startup investment dataset (Kaggle)
- Size: ~54,000 startups
- Key fields used:
  - `category_list`
  - `market`
  - `founded_year`
  - `status` (optional, for analysis)

> Raw data is kept out of version control. Only processed outputs and reports are tracked.

---

## Methodology
1. **Text construction**  
   Combined `category_list` and `market` into a single text field per startup.

2. **Preprocessing**  
   - Lowercasing  
   - Token normalization  
   - Noise and punctuation removal  

3. **Topic Modeling**  
   - TF-IDF vectorization  
   - Non-negative Matrix Factorization (NMF)  
   - 10 interpretable topic clusters  

4. **Analysis & Reporting**  
   - Topic keyword extraction  
   - Topic distribution analysis  
   - Trend analysis by founding year (when available)

---

## Key Outputs
- `reports/results.md`  
  → Topic keywords, VC-style sector interpretations, and venture insights

- `reports/figures/topic_distribution.png`  
  → Startup distribution across topics

- `reports/figures/topic_trends.png`  
  → Topic trends over time

---

## Example Topics Identified
- Enterprise Software & SaaS  
- Biotech & Pharmaceuticals  
- E-Commerce & Marketplaces  
- Mobile Payments & Apps  
- Digital Health & Wellness  
- CleanTech & Energy  
- AdTech & Social Platforms  

Each topic is interpreted from a **venture capital perspective**, including business models, investment relevance, and risk signals.

---

## Limitations
- Topics are derived from categorical labels, not full startup descriptions
- Dataset may overrepresent funded or visible startups
- Topic modeling reveals co-occurrence patterns, not causal outcomes

---

## How to Run

```bash
pip install -r requirements.txt
python src/preprocess.py
python src/topic_model.py
python src/trend_report.py
