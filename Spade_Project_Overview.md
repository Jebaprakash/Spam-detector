# Spade: Advanced Spam Detection using ML & NLP

Spade is a comprehensive machine learning application designed to classify emails and SMS messages as either **Spam** (unwanted/malicious) or **Ham** (legitimate). Developed at the Sathyabama Institute of Science and Technology, this project stands out by utilizing a robust **Ensemble Classifier** approach.

> [!NOTE]
> Instead of relying on a single algorithm, Spade acts as a "committee." Five distinct, highly-trained machine learning models independently vote on whether a message is spam. The majority verdict wins.

---

## 🏗️ Architecture & How It Works

The application operates in a straightforward, automated pipeline from the moment a user inputs a message to the final verdict.

### 1. Data Processing Pipeline (NLP)
Raw text is messy. Before it can be analyzed, it must pass through the `DP.py` (Data Processing) module. 
*   **Cleaning:** HTML tags, special characters, and non-ASCII elements are stripped away. Text is converted to lowercase.
*   **Tokenization & Stop-words:** NLTK (Natural Language Toolkit) breaks the text into individual words and removes "stop-words" (common words like "and", "the", "is" that carry no predictive weight).
*   **Lemmatization:** Words are reduced to their dictionary root form (e.g., "running" becomes "run") using POS-tagging to ensure grammatical accuracy.

### 2. Feature Extraction (TF-IDF)
The cleaned text must be translated into numbers for the ML models.
*   The system uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into mathematical arrays. 
*   TF-IDF measures how important a word is. If a word appears frequently in a single message but rarely across the English language, it scores highly. Experiments proved this was far more accurate than simple word counting (Bag-of-Words).

### 3. The Ensemble Models
The mathematical vector is simultaneously fed into five separate, pre-trained Classifiers (`ML.py`):
1.  **Naive Bayes:** Excellent at text probability modeling.
2.  **Logistic Regression:** Highly accurate binary classifier.
3.  **Random Forest:** A collection of 19 "decision trees."
4.  **K-Nearest Neighbors (KNN):** Looks for similarities with the 9 closest previously seen messages.
5.  **Support Vector Machine (SVM):** Draws complex mathematical boundaries to segregate spam data.

### 4. The Majority Verdict
Each model casts a "Spam" or "Ham" vote. If **3 or more models** classify the text as Spam, the final application verdict is **Spam**. This "wisdom of the crowd" approach achieves a documented **99.0% accuracy**.

### 5. Bonus: Named Entity Recognition
As an added feature, the UI runs the raw text through `spaCy`'s NLP engine to extract "Named Entities." This instantly highlights People, Organizations, Dates, and Monetary values mentioned in the text, allowing the user to scan the subject matter without reading the full message.

---

## 📁 File Structure Overview

| File / Folder | Purpose |
| :--- | :--- |
| `UI.py` | The main Streamlit web application. Handles the frontend interface, file uploads, and displaying the verdict. |
| `ML.py` | The Machine Learning core. Handles TF-IDF vectorization, trains the 5 classifiers, and manages the ensemble voting logic. |
| `DP.py` | Data Processing utilities. Contains Regex scrubbers, NLTK lemmatization logic, and the spaCy Entity extraction. |
| `Cleaned_Data.csv` | The pre-processed dataset used to train the models when the application starts. |
