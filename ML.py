"""
ML.py — Machine Learning Module
Spade: Spam Detection using Machine Learning and Natural Language Processing
Sathyabama Institute of Science and Technology, May 2022

Authors: Vathumalli Sri Ganesh (38110623) & Vattikuti Manideep Sitaram (38110624)
Guide:   Dr. S. Prince Mary M.E., Ph.D.

Responsibilities:
  - TF-IDF vectorisation (selected over BoW from experimentation)
  - Training five classifiers: Naive Bayes, Logistic Regression,
    Random Forest (n=19), KNN (k=9), SVM
  - Ensemble majority-vote prediction
  - Per-model probability output for UI display
  - Keyword/entity extraction helper
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Hyperparameters (determined through iterative experimentation in the report)
# ---------------------------------------------------------------------------
KNN_K         = 9     # K=9 gave best score for TF-IDF model
RF_ESTIMATORS = 19    # 19 trees gave best score for both BoW and TF-IDF
RANDOM_STATE  = 10    # fixed for reproducibility
TEST_SIZE     = 0.20  # 80/20 train-test split

MODEL_NAMES = [
    "Naive Bayes",
    "Logistic Regression",
    "Random Forest",
    "K-Nearest Neighbors",
    "Support Vector Machine",
]

# Published accuracy scores from the report (TF-IDF models)
MODEL_SCORES = {
    "Naive Bayes":            96.05,
    "Logistic Regression":    98.80,
    "Random Forest":          96.80,
    "K-Nearest Neighbors":    96.61,
    "Support Vector Machine": 98.82,
}


class SpadeModel:
    """
    Ensemble spam classifier.

    Usage
    -----
        model = SpadeModel()                 # loads data, trains all models
        vector  = model.get_vector(text)     # TF-IDF vector for user input
        verdict = model.get_prediction(vec)  # "Spam" or "Non-Spam"
        probs   = model.get_probabilities(vec)  # list of [ham%, spam%] per model
    """

    def __init__(self, data_path: str = "Cleaned_Data.csv"):
        print("[ML] Loading dataset …")
        df = pd.read_csv(data_path)
        df["Email"] = df["Email"].apply(lambda x: np.str_(x))

        data   = df["Email"]
        labels = df["Label"]

        # 80 / 20 split
        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(
            data, labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        # TF-IDF vectoriser — fit on training corpus only
        print("[ML] Fitting TF-IDF vectoriser …")
        self.vectorizer = TfidfVectorizer()
        self.train_vectors = self.vectorizer.fit_transform(
            self.X_train.tolist()
        )

        # Instantiate all five classifiers
        self.model_nb  = MultinomialNB()
        self.model_lr  = LogisticRegression(max_iter=1000)
        self.model_rf  = RandomForestClassifier(n_estimators=RF_ESTIMATORS,
                                                random_state=RANDOM_STATE)
        self.model_knn = KNeighborsClassifier(n_neighbors=KNN_K)
        self.model_svm = SVC(probability=True, random_state=RANDOM_STATE)

        # Train all classifiers on TF-IDF training vectors
        print("[ML] Training Naive Bayes …")
        self.model_nb.fit(self.train_vectors, self.y_train)

        print("[ML] Training Logistic Regression …")
        self.model_lr.fit(self.train_vectors, self.y_train)

        print("[ML] Training Random Forest …")
        self.model_rf.fit(self.train_vectors, self.y_train)

        print("[ML] Training KNN …")
        self.model_knn.fit(self.train_vectors, self.y_train)

        print("[ML] Training SVM …")
        self.model_svm.fit(self.train_vectors, self.y_train)

        print("[ML] All models ready.")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_vector(self, text: str):
        """
        Transform a single cleaned text string into a TF-IDF sparse vector.
        The same vectoriser that was fitted on training data is used so that
        vocabulary and IDF weights are consistent.
        """
        return self.vectorizer.transform([text])

    def get_prediction(self, vector) -> str:
        """
        Ensemble majority vote across all five classifiers.
        If >= 3 models predict spam (label 1) → "Spam", else "Non-Spam".
        """
        pred_nb  = self.model_nb.predict(vector)[0]
        pred_lr  = self.model_lr.predict(vector)[0]
        pred_rf  = self.model_rf.predict(vector)[0]
        pred_svm = self.model_svm.predict(vector)[0]
        pred_knn = self.model_knn.predict(vector)[0]

        preds = [pred_nb, pred_lr, pred_rf, pred_svm, pred_knn]
        spam_votes = preds.count(1)

        return "Spam" if spam_votes >= 3 else "Non-Spam"

    def get_probabilities(self, vector) -> list:
        """
        Return per-model probability arrays [P(ham), P(spam)] × 100.

        Returns
        -------
        list of 5 arrays, one per model in this order:
            [Naive Bayes, Logistic Regression, Random Forest, KNN, SVM]
        """
        prob_nb  = self.model_nb.predict_proba(vector)[0]  * 100
        prob_lr  = self.model_lr.predict_proba(vector)[0]  * 100
        prob_rf  = self.model_rf.predict_proba(vector)[0]  * 100
        prob_knn = self.model_knn.predict_proba(vector)[0] * 100
        prob_svm = self.model_svm.predict_proba(vector)[0] * 100

        return [prob_nb, prob_lr, prob_rf, prob_knn, prob_svm]

    def get_model_names(self) -> list:
        return MODEL_NAMES

    def get_model_scores(self) -> dict:
        """Return published accuracy scores from the paper for display."""
        return MODEL_SCORES

    def evaluate_test_set(self) -> dict:
        """
        Score all five models on the held-out test set.
        Returns accuracy (%) for each model and the ensemble.
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score

        test_vectors = self.vectorizer.transform(self.X_test.tolist())

        results = {}
        ensemble_preds = []

        for name, clf in [
            ("Naive Bayes",            self.model_nb),
            ("Logistic Regression",    self.model_lr),
            ("Random Forest",          self.model_rf),
            ("K-Nearest Neighbors",    self.model_knn),
            ("Support Vector Machine", self.model_svm),
        ]:
            preds = clf.predict(test_vectors)
            ensemble_preds.append(preds)
            results[name] = {
                "accuracy":  round(accuracy_score(self.y_test, preds) * 100, 2),
                "precision": round(precision_score(self.y_test, preds, zero_division=0) * 100, 2),
                "f1":        round(f1_score(self.y_test, preds, zero_division=0) * 100, 2),
            }

        # Ensemble via majority vote
        stacked = np.array(ensemble_preds)
        majority = (stacked.sum(axis=0) >= 3).astype(int)
        results["Ensemble (Proposed)"] = {
            "accuracy":  round(accuracy_score(self.y_test, majority) * 100, 2),
            "precision": round(precision_score(self.y_test, majority, zero_division=0) * 100, 2),
            "f1":        round(f1_score(self.y_test, majority, zero_division=0) * 100, 2),
        }

        return results
