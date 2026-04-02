"""
DP.py — Data Processing Module
Spade: Spam Detection using Machine Learning and Natural Language Processing


Authors: Vathumalli Sri Ganesh (38110623) & Vattikuti Manideep Sitaram (38110624)
Guide:   Dr. S. Prince Mary M.E., Ph.D.

Responsibilities:
  - Tag/character cleaning via regex
  - Sentence tokenisation and word tokenisation (NLTK)
  - Stop-word removal (NLTK)
  - Lemmatisation with POS-tagging (WordNet)
  - Named-entity extraction (spaCy en_core_web_sm)
  - Dataset merging and cleaning utilities
"""

import re
from collections import defaultdict

import pandas as pd
import spacy
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# ---------------------------------------------------------------------------
# POS tag → WordNet constant mapping
# ---------------------------------------------------------------------------
tag_map = defaultdict(lambda: wn.NOUN)
tag_map["J"] = wn.ADJ
tag_map["V"] = wn.VERB
tag_map["R"] = wn.ADV

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Load spaCy model once at import time
nlp = spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------

def process_sentence(sentence: str):
    """
    Clean, lemmatise, and extract nouns from a single sentence.

    Returns
    -------
    sent : str
        Lemmatised sentence with stop-words removed.
    nouns : list[str]
        Raw nouns (NN-tagged tokens, length > 1).
    """
    nouns = []
    base_words = []
    final_words = []

    # Keep original tokens for noun extraction (before regex strip)
    words_original = word_tokenize(sentence)

    # Strip punctuation and underscores, then re-tokenise
    sentence_clean = re.sub(r"[^ \w\s]", "", sentence)
    sentence_clean = re.sub(r"_", " ", sentence_clean)
    words = word_tokenize(sentence_clean)

    # Lemmatise using POS tags
    for token, tag in pos_tag(words):
        base_words.append(lemmatizer.lemmatize(token, tag_map[tag[0]]))

    # Remove stop-words
    for word in base_words:
        if word not in stop_words:
            final_words.append(word)

    sent = " ".join(final_words)

    # Extract plain nouns from original tokens
    for token, tag in pos_tag(words_original):
        if tag == "NN" and len(token) > 1:
            nouns.append(token)

    return sent, nouns


def clean(email: str):
    """
    Full pipeline: lowercase → remove HTML/XML tags → tokenise sentences →
    process each sentence → reassemble.

    Parameters
    ----------
    email : str
        Raw email or SMS body text.

    Returns
    -------
    cleaned_text : str
        Fully processed text ready for vectorisation.
    nouns : list[str]
        All nouns extracted across sentences.
    """
    email = email.lower()

    # Remove HTML/XML tags and non-ASCII characters
    email = re.sub(r"<[^>]+>", " ", email)
    email = re.sub(r"[^\x00-\x7F]+", " ", email)

    sentences = sent_tokenize(email)
    total_nouns = []
    string = ""

    for sent in sentences:
        sentence, nouns = process_sentence(sent)
        string += " " + sentence
        total_nouns += nouns

    return string.strip(), total_nouns


def ents(text: str):
    """
    Extract named entities from raw (un-cleaned) text using spaCy.

    Returns
    -------
    dict  : { entity_label: [entity_text, ...] }
            e.g. {"ORG": ["Gmail", "PayPal"], "GPE": ["Miami"]}
    "no"  : str literal when no entities are found.
    """
    doc = nlp(text)
    expls = {}

    if doc.ents:
        for ent in doc.ents:
            label = ent.label_
            word = ent.text
            if label in expls:
                expls[label].append(word)
            else:
                expls[label] = [word]
        return expls

    return "no"


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

RENAME_MAP = {
    "CARDINAL": "Numbers",
    "TIME":     "Time",
    "ORG":      "Companies / Organisations",
    "GPE":      "Locations",
    "PERSON":   "People",
    "MONEY":    "Money",
    "FAC":      "Facilities",
    "DATE":     "Dates",
    "EVENT":    "Events",
    "PRODUCT":  "Products",
}


def load_and_merge(paths: list[str], text_col: str, label_col: str) -> pd.DataFrame:
    """
    Load multiple CSV datasets, standardise column names, drop nulls/duplicates,
    and concatenate into a single DataFrame.

    Parameters
    ----------
    paths     : list of CSV file paths
    text_col  : name of the email-body column in source files
    label_col : name of the spam/ham label column (0 = ham, 1 = spam)

    Returns
    -------
    pd.DataFrame with columns ["Email", "Label"]
    """
    frames = []
    for path in paths:
        df = pd.read_csv(path, encoding="latin-1")
        df = df[[text_col, label_col]].copy()
        df.columns = ["Email", "Label"]
        df.dropna(inplace=True)
        df.drop_duplicates(subset="Email", inplace=True)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged.dropna(inplace=True)
    merged.drop_duplicates(subset="Email", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def build_cleaned_csv(input_paths: list[str], text_col: str, label_col: str,
                      output_path: str = "Cleaned_Data.csv"):
    """
    End-to-end: load → merge → NLP-clean each email → save.
    This is the offline preprocessing step; run once before training.
    """
    df = load_and_merge(input_paths, text_col, label_col)
    print(f"[DP] Merged dataset: {len(df)} records")

    cleaned_emails = []
    for i, row in df.iterrows():
        if i % 500 == 0:
            print(f"[DP] Cleaning record {i}/{len(df)} …")
        cleaned, _ = clean(str(row["Email"]))
        cleaned_emails.append(cleaned)

    df["Email"] = cleaned_emails
    df.to_csv(output_path, index=False)
    print(f"[DP] Saved cleaned dataset → {output_path}")
    return df
