# Spade — Spam Detection using ML & NLP

**Sathyabama Institute of Science and Technology, May 2022**  
Authors: Vathumalli Sri Ganesh · Vattikuti Manideep Sitaram  
Guide: Dr. S. Prince Mary M.E., Ph.D.

---

## Project structure

```
spade/
├── DP.py              # Data Processing module (NLP pipeline, entity extraction)
├── ML.py              # Machine Learning module (5 classifiers + ensemble)
├── UI.py              # Streamlit web interface
├── requirements.txt   # Python dependencies
├── Cleaned_Data.csv   # Pre-processed dataset (see Step 3 below)
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLP models

```bash
# NLTK data
python -c "
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
"

# spaCy English model
python -m spacy download en_core_web_sm
```

### 3. Prepare the dataset

Download the following datasets from Kaggle and place them in the `spade/` folder:

| Dataset | Kaggle slug | Key columns |
|---------|-------------|-------------|
| Enron Spam Subset | `wcukierski/enron-email-dataset` | `Body`, `Label` |
| Lingspam | `mandygu/lingspam-dataset` | `Body`, `Label` |
| Spam-6000 | any 6000-spam CSV | `Body`, `Label` |

Then run the one-time preprocessing script to generate `Cleaned_Data.csv`:

```python
# run_preprocessing.py
from DP import build_cleaned_csv

build_cleaned_csv(
    input_paths=["enronSpamSubset.csv", "lingspam.csv", "spam6000.csv"],
    text_col="Body",
    label_col="Label",
    output_path="Cleaned_Data.csv"
)
```

```bash
python run_preprocessing.py
```

> This takes a few minutes. Run it only once — the result is cached in `Cleaned_Data.csv`.

### 4. Run the app

```bash
streamlit run UI.py
```

Open your browser at **http://localhost:8501**

---

## How it works

```
User input
   │
   ▼
DP.clean()          — lowercase, strip tags, tokenise, remove stop-words, lemmatise
   │
   ▼
ML.get_vector()     — TF-IDF transform (same vocabulary as training)
   │
   ├──► Naive Bayes          ──┐
   ├──► Logistic Regression  ──┤
   ├──► Random Forest (n=19) ──┼──► Majority vote (≥3/5) ──► Spam / Ham
   ├──► KNN (k=9)            ──┤
   └──► SVM (RBF kernel)     ──┘
   │
   ▼
DP.ents()           — spaCy named-entity extraction on raw text
   │
   ▼
UI.py               — Display verdict + per-model confidence bars + entities
```

### Why TF-IDF over Bag-of-Words?

From the paper's experiments, TF-IDF outperformed BoW on every classifier:

| Model | BoW | TF-IDF |
|---|---|---|
| Naive Bayes | 98.04% | 96.05% |
| Logistic Regression | 98.53% | **98.80%** |
| KNN | 83.15% | **96.61%** |
| Random Forest | 96.84% | 96.80% |
| SVM | 59.41% | **98.82%** |
| **Ensemble** | — | **99.0%** |

TF-IDF is selected as the primary language model. The SVM BoW anomaly (59.41%) also highlights why TF-IDF is the safer choice across all five classifiers.

---

## Ensemble results (from paper)

| Metric | Value |
|--------|-------|
| Accuracy | **99.0%** |
| Precision | **98.5%** |
| F1 Score | **98.6%** |

---

## Limitations (as noted in the paper)

- Predicts and classifies spam — does not block it at the mail-server level.
- Entity detection may struggle with heavily obfuscated alphanumeric messages.
- Classification of a large message may take a few seconds on first run (model load).
- English-language only.
