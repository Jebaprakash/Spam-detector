"""
UI.py — User Interface Module (Streamlit)
Spade: Spam Detection using Machine Learning and Natural Language Processing
Sathyabama Institute of Science and Technology, May 2022

Authors: Vathumalli Sri Ganesh (38110623) & Vattikuti Manideep Sitaram (38110624)
Guide:   Dr. S. Prince Mary M.E., Ph.D.

Run:
    streamlit run UI.py

Requirements:
    pip install streamlit scikit-learn nltk spacy pandas numpy
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords');
               nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
    
    Place Cleaned_Data.csv in the same directory before running.
    To generate Cleaned_Data.csv from raw datasets, call DP.build_cleaned_csv().
"""
import time
import nltk

# Ensure NLTK data is ready BEFORE importing DP/ML which depend on it
for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)

import spacy
import streamlit as st

from DP import RENAME_MAP, clean, ents
from ML import SpadeModel

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spade — Spam Detector",
    page_icon="🂡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — keeps the look clean and consistent
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main .block-container { padding-top: 2rem; max-width: 900px; }
        .stTextArea textarea { font-size: 14px; }
        .verdict-spam {
            background-color: rgba(255, 75, 75, 0.15); 
            border: 1px solid rgba(255, 75, 75, 0.3);
            border-left: 5px solid #ff4b4b;
            padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 1rem;
        }
        .verdict-ham {
            background-color: rgba(33, 195, 84, 0.15); 
            border: 1px solid rgba(33, 195, 84, 0.3);
            border-left: 5px solid #21c354;
            padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: 1rem;
        }
        .verdict-title { font-size: 22px; font-weight: 600; }
        .ent-chip {
            display: inline-block; padding: 3px 10px;
            border-radius: 20px; font-size: 12px; margin: 2px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model (cached so it only trains once per session)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model() -> SpadeModel:
    return SpadeModel()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("## 🂡 Spade")
st.caption(
    "Spam Detection using Machine Learning & Natural Language Processing  "
    
)
st.divider()

# ---------------------------------------------------------------------------
# Input section
# ---------------------------------------------------------------------------
st.markdown("#### Paste your email or SMS")
user_text = st.text_area(
    label="email_body",
    label_visibility="collapsed",
    height=220,
    placeholder="Type or paste an email / SMS message here (minimum 50 characters) …",
)

# File uploader as alternative input
uploaded_file = st.file_uploader(
    "Or upload a plain-text file (.txt)",
    type=["txt"],
    label_visibility="visible",
)

# Resolve which input to use
given_text = ""
if user_text and len(user_text.strip()) > 20:
    given_text = user_text.strip()
if uploaded_file is not None:
    if given_text:
        st.error("Multiple inputs detected. Please use either the text area OR the file upload, not both.")
        given_text = ""
    else:
        given_text = uploaded_file.read().decode("utf-8", errors="ignore")

# ---------------------------------------------------------------------------
# Detect button
# ---------------------------------------------------------------------------
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    detect_clicked = st.button("🔍  Detect", use_container_width=True, type="primary")

st.caption("ℹ️ If you see a caching warning, click Detect once more.")

# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------
if detect_clicked:
    if not given_text or len(given_text) < 50:
        st.warning("Please provide at least 50 characters of text before running detection.")
    else:
        with st.spinner("Loading models and running detection — this may take a moment on first run …"):
            model = load_model()

        # Process input through NLP pipeline
        cleaned_text, _ = clean(given_text)
        vector           = model.get_vector(cleaned_text)

        # Ensemble prediction and per-model probabilities
        prediction  = model.get_prediction(vector)
        probability  = model.get_probabilities(vector)
        model_names = model.get_model_names()

        # ----------------------------------------------------------------
        # Verdict banner
        # ----------------------------------------------------------------
        is_spam = prediction == "Spam"

        if is_spam:
            st.markdown(
                '<div class="verdict-spam">'
                '<div class="verdict-title">🚫 Spam</div>'
                '<div>This message has been classified as <strong>SPAM</strong> '
                'by the ensemble classifier.</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="verdict-ham">'
                '<div class="verdict-title">✅ Not Spam (Ham)</div>'
                '<div>This message appears to be <strong>legitimate</strong> '
                'according to the ensemble classifier.</div>'
                "</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ----------------------------------------------------------------
        # Per-model confidence bars
        # ----------------------------------------------------------------
        st.markdown("#### Model-by-model predictions")
        st.caption("Each bar shows the model's confidence in the final verdict.")

        for i, name in enumerate(model_names):
            probs = probability[i]            # [P(ham)×100, P(spam)×100]
            confidence = int(probs[1]) if is_spam else int(probs[0])
            confidence = max(0, min(confidence, 100))

            col_name, col_bar = st.columns([1, 3])
            with col_name:
                st.markdown(f"**{name}**")
            with col_bar:
                st.write(f"{confidence}%")
                progress_bar = st.progress(0)
                for pct in range(confidence + 1):
                    time.sleep(0.005)
                    progress_bar.progress(pct)

        st.divider()

        # ----------------------------------------------------------------
        # Named-entity insights
        # ----------------------------------------------------------------
        st.markdown("#### Named-entity insights")
        st.caption(
            "Entities extracted from the original (un-cleaned) text using spaCy. "
            "Expand each category to view."
        )

        entities = ents(given_text)

        col_ent_desc, col_ent_vals = st.columns([1, 2])

        with col_ent_desc:
            st.markdown(
                "These are the **named entities** found in the text. "
                "Each category is described on the right."
            )

        with col_ent_vals:
            if entities == "no":
                st.info("No named entities were detected in this text.")
            else:
                for label, values in entities.items():
                    friendly = RENAME_MAP.get(label, label)
                    explanation = spacy.explain(label) or label
                    unique_vals = list(dict.fromkeys(values))          # deduplicate, preserve order

                    with st.expander(f"{friendly}  ({len(unique_vals)} found)"):
                        st.caption(explanation)
                        st.write(", ".join(unique_vals))

        st.divider()

        # ----------------------------------------------------------------
        # Technical details expander
        # ----------------------------------------------------------------
        with st.expander("🔬 Technical details"):
            st.markdown(
                f"""
| Detail | Value |
|---|---|
| Input length (raw) | {len(given_text)} characters |
| Cleaned text length | {len(cleaned_text)} characters |
| TF-IDF feature dimensions | {vector.shape[1]:,} |
| Spam votes | {sum(1 for p in probability if p[1] > p[0])} / 5 |
| Ham votes | {sum(1 for p in probability if p[0] >= p[1])} / 5 |
| Final verdict | **{prediction}** |
"""
            )
            st.markdown("**Published model accuracies (TF-IDF, from paper):**")
            scores = model.get_model_scores()
            for mname, acc in scores.items():
                st.write(f"- {mname}: {acc}%")
            st.write("- **Ensemble (proposed model): 99.0%**")
