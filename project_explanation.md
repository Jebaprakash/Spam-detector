# Spade: Spam Detection using Machine Learning & NLP

**Spade** is a Machine Learning and Natural Language Processing (NLP) project designed to detect whether an email or SMS message is **Spam** (junk/malicious) or **Ham** (legitimate). 

It was developed as a project at the Sathyabama Institute of Science and Technology and stands out because it doesn't just rely on one algorithm; instead, it uses a "team" of five different AI models to make a highly accurate final decision.

Here is a breakdown of what the project is and exactly how the pipeline works from start to finish:

---

## 1. The Input (User Interface)
The project features a sleek web-based interface built using **Streamlit** (`UI.py`). A user can either type/paste an email into a text box or upload a `.txt` file. 

## 2. Natural Language Processing Pipeline (`DP.py`)
Before a machine learning model can understand text, the text needs to be cleaned and standardized. When you click "Detect", the input goes through the Data Processing (DP) module:
*   **Cleaning:** It converts all text to lowercase and strips out any HTML tags or weird characters.
*   **Tokenization:** It breaks the message down into individual sentences and words using a library called **NLTK**.
*   **Stop-word Removal:** It removes common, low-value words (like "the", "is", "at", "which") that don't help determine if a message is spam.
*   **Lemmatization:** It figures out the grammatical context of a word (Noun, Verb, etc.) and reduces it to its dictionary root. (e.g., "running" becomes "run", "better" might become "good").

## 3. Feature Extraction - TF-IDF (`ML.py`)
Machine Learning models understand numbers, not words. The cleaned text is translated into a massive mathematical array (vector) using **TF-IDF** (Term Frequency-Inverse Document Frequency). 
*   Unlike simple "Bag-of-Words" (which just counts how many times a word appears), TF-IDF gives higher mathematical weight to words that are frequent in this specific message, but *rare* across the English language overall. 
*   According to the developers' research, TF-IDF was vastly superior for their spam detection.

## 4. The Ensemble Classifier (The Brains)
This is the core of the project. Instead of trusting one model, Spade feeds the mathematical vector into **five different machine learning classifiers** simultaneously:
1.  **Naive Bayes**: Great at understanding text probabilities.
2.  **Logistic Regression**: Excellent for binary (Spam vs. Not Spam) decisions.
3.  **Random Forest**: Uses 19 distinct "decision trees" to evaluate the data.
4.  **K-Nearest Neighbors (KNN)**: Looks at the 9 closest historical examples to the current message.
5.  **Support Vector Machine (SVM)**: Draws complex mathematical boundaries to separate spam from legitimate emails.

## 5. Majority Vote & Output
*   **The Voting System:** Each of the 5 models independently casts a vote: "Spam" or "Ham". 
*   If **3 or more** models vote Spam, the final system verdict is 🚫 **Spam**. If not, it's ✅ **Ham**. The authors note this ensemble approach achieves a **99.0% accuracy rate**.
*   The UI then renders the final verdict, along with animated progress bars showing exactly how confident *each* individual model was.

## 6. Bonus: Named-Entity Recognition (NER)
While the ML models run, a second system called **spaCy** reads the original, uncleaned text. It looks for "Named Entities"—things like Company names (Google, PayPal), People, Locations, Dates, or Money amounts ($1,000). The app displays these entities to give the user quick insights into what the email is discussing without having to read it all.
