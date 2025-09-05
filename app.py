from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import html, re
import pycld2 as cld2
import jieba
import spacy
from wordcloud import WordCloud
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Model Loading
CANTONESE_SEGMENTER_PATH = "trained_segmenter.pkl"
CANTONESE_MODEL_BUNDLE = "cantonese_model_bundle.pkl"
ENGLISH_MODEL_BUNDLE = "english_model_bundle.pkl"
BM_MODEL_BUNDLE = "bm_model_bundle.pkl"
MANDARIN_MODEL_BUNDLE = "mandarin_model_bundle.pkl"
font_path = BASE_DIR / "SimHei.ttf"
feedback_db= "feedback.db"

LANG_DISPLAY = {
    "en": "English",
    "ms": "Bahasa Melayu",
    "zh-cn": "Mandarin",
    "zh-tw": "Cantonese",
    "zh-hant": "Cantonese",
}

# CACHING HEAVY LOADS
@st.cache_resource(show_spinner=False)
def load_joblib(path):
    import joblib
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_spacy(name):
    import spacy
    return spacy.load(name, disable=["ner","parser","textcat"])

@st.cache_data(show_spinner=False)
def get_canto_stopwords():
    import pycantonese
    return pycantonese.stop_words()

@st.cache_resource
def get_malaya_tools():
    import malaya
    return {
        "rules": malaya.normalizer.rules.load(),
        "tokenizer": malaya.tokenizer.Tokenizer(),
        "stopwords": malaya.text.function.get_stopwords(),
        "lemmatizer": malaya.stem.sastrawi(),
    }

@st.cache_resource(show_spinner=False)
def get_db():
    conn = sqlite3.connect(feedback_db, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Timestamp TEXT NOT NULL,
            Text TEXT NOT NULL
        )
    """) 
    conn.commit()
    return conn

def add_feedback(conn, text: str):
    conn.execute(
        "INSERT INTO feedback (Timestamp, Text) VALUES (?, ?)",
        (datetime.now().isoformat(timespec="seconds"), text)
    )
    conn.commit()

@st.cache_data(show_spinner=False)
def fetch_feedback():
    conn = get_db()
    return pd.read_sql_query(
        "SELECT Timestamp, Text FROM feedback ORDER BY id DESC",
        conn
    )

@st.cache_resource(show_spinner=False)
def init_models():
    models = {
        "segmenter":load_joblib(CANTONESE_SEGMENTER_PATH),
        "canto_bundle":load_joblib(CANTONESE_MODEL_BUNDLE),
        "eng_bundle":load_joblib(ENGLISH_MODEL_BUNDLE),
        "bm_bundle":load_joblib(BM_MODEL_BUNDLE),
        "mandarin_bundle": load_joblib(MANDARIN_MODEL_BUNDLE),
        "nlp_en": load_spacy("en_core_web_sm"),
        "nlp_zh": load_spacy("zh_core_web_md"),
    }
    tools = get_malaya_tools()
    return models, tools

models, tools = init_models() #Initialize (cached)

# Cantonese
segmenter = models["segmenter"]
canto_bundle= models["canto_bundle"]
cantonese_clf = canto_bundle["stacking_model"]
cantonese_vectorizer = canto_bundle["vectorizer"]
canto_stopwords= get_canto_stopwords()   # cached stopwords
cantonese_selector= canto_bundle.get("selector", None)

# English
eng_bundle = models["eng_bundle"]
english_clf = eng_bundle["stacking_model"]
english_vectorizer= eng_bundle["vectorizer"]
english_selector= eng_bundle.get("selector", None)

# Bahasa Melayu
bm_bundle= models["bm_bundle"]
bm_clf = bm_bundle["stacking_model"]
bm_vectorizer= bm_bundle["vectorizer"]
bm_selector= bm_bundle.get("selector", None)

# SpaCy (cached)
nlp_en = models["nlp_en"]
nlp_zh = models["nlp_zh"]

# Malaya tools (cached)
rules = tools["rules"]
tokenizer  = tools["tokenizer"]
malay_stopwords = tools["stopwords"]
lemmatizer = tools["lemmatizer"]

# Mandarin
mandarin_bundle= models["mandarin_bundle"]
mandarin_clf= mandarin_bundle["stacking_model"]
mandarin_vectorizer= mandarin_bundle["vectorizer"]
mandarin_selector= mandarin_bundle.get("selector", None)

# Language Detection
def detect_language(text: str) -> str:
    try:
        _, _, details = cld2.detect(text)
        code = details[0][1].lower()  
        if code in ("yue", "zh-hant"):
            return "zh-tw"
        if code == "zh":
            return "zh-cn"
        if code in ("ms", "id"):
            return "ms"
        return code
    except Exception:
        return "Unknown"
    
# Text Preprocessing
def cantonese_preprocess_text(text: str):
    cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z\s']", "", text)
    pred = list(segmenter.predict(cleaned))
    if not pred or pred == [[]]:
        return cleaned, [], []
    tokens = [tok for sub in pred for tok in sub] if pred and isinstance(pred[0], list) else pred
    tokens_ns = [w for w in tokens if w not in canto_stopwords]
    
    return cleaned, tokens, tokens_ns

def english_preprocess_text(text: str):
    txt= re.sub(r'http\S+|www\S+|https\S+', '', text)  
    txt= re.sub(r"([!?.,])", r" \1 ", txt) 
    txt= re.sub('^ ', '', txt)
    txt= re.sub(' $', '', txt)
    txt= re.sub(r"\s+", " ", txt).strip().lower()
    doc= nlp_en(txt)
    tokens = [t.text for t in doc]
    tokens_ns = [t.text for t in doc if t.is_alpha and not t.is_stop]
    lemmas = [t.lemma_ for t in nlp_en(" ".join(tokens_ns))]
    return txt, tokens, tokens_ns, lemmas

def mandarin_preprocess_text(text: str):
    txt = re.sub(r"[^一-龥。！？]", "", text)
    tokens = [tok.text for tok in nlp_zh(txt) if not tok.is_space]
    doc = spacy.tokens.Doc(nlp_zh.vocab, words=tokens)
    tokens_ns = [tok.text for tok in doc if not tok.is_stop]
    return txt, tokens, tokens_ns

def melayu_preprocess_text(text: str):
    cleaned = re.sub(r"http\S+|www\S+", "", text.lower())
    cleaned = re.sub(r"[^a-zA-Z\u00C0-\u00FF\s]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    norm_result = rules.normalize(cleaned)
    normalized = norm_result[0] if isinstance(norm_result, list) and norm_result else cleaned
    tokens    = tokenizer.tokenize(normalized)
    tokens_ns = [w for w in tokens if w.lower() not in malay_stopwords]
    lemmas    = [lemmatizer.stem(w) for w in tokens_ns]
    return cleaned, normalized, tokens, tokens_ns, lemmas

# Prediction
def predict(text: str, lang: str) -> str:
    if lang in ("zh-tw", "zh-hant"):  # Cantonese
        _, _, tokens_ns = cantonese_preprocess_text(text)
        X = cantonese_vectorizer.transform([" ".join(tokens_ns)])
        return cantonese_clf.predict(X)[0]
    if lang == "zh-cn":  # Mandarin
        _, _, tokens_ns = mandarin_preprocess_text(text)
        X = mandarin_vectorizer.transform([" ".join(tokens_ns)])
        return mandarin_clf.predict(X)[0]
    if lang == "en":
        _, _, _, lemmas = english_preprocess_text(text)
        X = english_vectorizer.transform([" ".join(lemmas)])
        if english_selector:
            X = english_selector.transform(X)
        return english_clf.predict(X)[0]
    if lang == "ms":
        _, _, _, _, lemmas = melayu_preprocess_text(text)
        X = bm_vectorizer.transform([" ".join(lemmas)])
        return bm_clf.predict(X)[0]
    
    # fallback keyword rules
    fallback_rules = {
    "en": {
        "personal disorder": ["split", "multiple personality", "identity disorder"],
        "bipolar": ["manic", "mood swings", "highs and lows"],
        "depression": ["sad", "depress", "hopeless", "empty"],
        "anxiety": ["anxious", "panic", "worry", "nervous"],
        "stress": ["overwhelmed", "pressure", "burnout", "tension"]
    },
    "zh-cn": {
        "depression": ["抑郁", "难过", "悲伤", "情绪低落"],
        "normal": []
    },
    "zh-tw": {
        "anxiety": ["焦慮", "緊張", "擔心", "驚"],
        "loneliness": ["孤單", "寂寞", "冇人", "自己"]
    },
    "ms": {
        "depression": ["murung", "sedih", "tertekan"],
        "anxiety": ["cemas", "takut", "gelisah", "panik"],
        "stress": ["tekanan", "letih", "beban", "penat"]
    }
    }

    # Get fallback keywords for the language
    lc = text.lower()
    rules= fallback_rules.get(lang, {})
    for label, keywords in rules.items():
        if any(kw in lc for kw in keywords):
            return label
    # If no symptoms matched, return 'Unknown'
    return "Unknown"

def _pretty_cjk(tokens_ns, cleaned):
    if not tokens_ns:
        return cleaned
    # If most tokens are single-char, show as a continuous string.
    long_tokens = sum(1 for t in tokens_ns if len(t) > 1)
    if long_tokens >= max(1, int(0.3 * len(tokens_ns))):
        return " ".join(tokens_ns)          # looks like real words
    return "".join(tokens_ns)               # avoid spaces between every char

def to_preprocessed_text(lang, cleaned=None, tokens=None, ns=None, lemmas=None):
    """
    Return the string model uses as input, per language.
    - en/ms: space-joined lemmas
    - zh-*: space-joined stopword-removed tokens (ns)
    - fallback: cleaned
    """
    ns = ns or []
    lemmas = lemmas or []

    if lang == "en":
        return " ".join(lemmas)
    if lang == "ms":
        return " ".join(lemmas)
    if lang in ("zh-cn", "zh-tw", "zh-hant"):
        return _pretty_cjk(ns, cleaned or "")
               
    return cleaned or ""

def render_wordcloud_for_language(df: pd.DataFrame, lang: str):
    """Render a single word cloud for the selected language."""
    blob = " ".join(df.loc[df["Language"] == lang, "Cleaned_text"].astype(str)).strip()
    if not blob:
        st.info("No text available to render a word cloud for this language.")
        return
    blob = re.sub(r"http\S+|www\S+", "", blob)
    if lang in ("zh-cn", "zh-tw", "zh-hant"):
        # Segment Chinese text
        seg = " ".join(jieba.cut(blob, cut_all=False))
        wc = WordCloud(font_path=font_path, width=800, height=400, background_color="white").generate(seg)
    else:
        wc = WordCloud(width=800, height=400, background_color="white").generate(blob)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

def handle_upload_flow():
    files = st.file_uploader("Upload CSV(s)", type="csv", accept_multiple_files=True)
    if not files:
        return
    rows = []
    for f in files:
        df = pd.read_csv(f)
        col = st.selectbox(f"Text column for {f.name}", df.columns, key=f.name)
        for txt in df[col].astype(str):
            lang = detect_language(txt)
            if lang == "en":
                cleaned, tokens, ns, lemmas = english_preprocess_text(txt)
            elif lang == "zh-cn":
                cleaned, tokens, ns = mandarin_preprocess_text(txt)
                lemmas = ns
            elif lang in ("zh-tw", "zh-hant"):
                cleaned, tokens, ns = cantonese_preprocess_text(txt)
                lemmas = ns
            elif lang == "ms":
                cleaned, norm, tokens, ns, lemmas = melayu_preprocess_text(txt)
            else:
                cleaned, tokens, ns, lemmas = txt, [], [], []
            label = predict(cleaned, lang)
            rows.append((txt, cleaned, tokens, lemmas, label, lang))

    out_df = pd.DataFrame(
        rows, columns=["Text", "Cleaned_text", "Token", "Lemmas", "Predicted Label", "Language"]
    )

    st.dataframe(out_df[["Text", "Predicted Label"]], use_container_width=True)
    st.download_button("Download Predictions", out_df.to_csv(index=False), "Predictions.csv", "text/csv")

    # Pie chart
    st.markdown("### Distribution of Predicted Mental Health Categories")
    label_counts = out_df["Predicted Label"].value_counts().reset_index()
    label_counts.columns = ["Mental Health Category", "Count"]
    fig_pie = px.pie(label_counts, names="Mental Health Category", values="Count",
                     title="Mental Health Category Distribution")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Word cloud 
    st.markdown("### Word Cloud")
    grouped= out_df.groupby("Language", sort=False)

    for lang, group in grouped: 
        text_blob = " ".join(group["Cleaned_text"].astype(str)).strip()

        text_blob = re.sub(r"http\S+|www\S+", "", text_blob)

        if lang in ['zh-cn', 'zh-tw', 'zh-hant']:
            words= jieba.cut(text_blob, cut_all= False)
            segmented_text = " ".join(words)
            wordcloud= WordCloud(
                font_path= font_path, 
                width= 800, 
                height= 400, 
                background_color= 'white'
            ).generate(segmented_text)
        else:
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='white'
            ).generate(text_blob)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)
        break

def handle_enter_text_flow():
    user_input = st.text_area("Text:")
    if not (st.button("Predict") and user_input):
        return
    lang = detect_language(user_input)
    lang_name = LANG_DISPLAY.get(lang, lang)

    # preprocess
    if lang == "en":
        cleaned, tokens, ns, lemmas = english_preprocess_text(user_input)
    elif lang == "zh-cn":
        cleaned, tokens, ns = mandarin_preprocess_text(user_input)
        lemmas = ns
    elif lang in ("zh-tw", "zh-hant"):
        cleaned, tokens, ns = cantonese_preprocess_text(user_input)
        lemmas = ns
    elif lang == "ms":
        cleaned, _, tokens, ns, lemmas = melayu_preprocess_text(user_input)
    else:
        cleaned, tokens, ns, lemmas = user_input, [], [], []

    # label = predict(cleaned, lang)
    pre_text = to_preprocessed_text(lang, cleaned=cleaned, tokens=tokens, ns=ns, lemmas=lemmas)
    ex= predict_with_explanations(user_input, lang, top_k=8)

    st.write(f"**Detected Language:** {lang_name}")
    st.subheader("Prediction Result")
    st.write("**Original Text:**", user_input)
    st.write("**Text (After Preprocessed):**", pre_text)
    st.write("**Predicted Label:**", ex["label"])

    # Probabilities
    if ex["probs"]:
        st.caption("Prediction probabilities")

    # to DataFrame
    probs_df = (
        pd.Series(ex["probs"], name="prob")
        .rename_axis("label")
        .reset_index()
    )
    probs_df["is_pred"] = probs_df["label"].eq(ex["label"])
    probs_df = probs_df.sort_values("prob")  # smallest at top

    import plotly.express as px

    fig = px.bar(
        probs_df,
        x="prob",
        y="label",
        orientation="h",
        text="prob",
        color="is_pred",  # highlight predicted label
        color_discrete_map={False: "#b8daec", True: "#97d4fa"},
    )
    fig.update_traces(
        texttemplate="%{text:.2%}",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"
    )
    fig.update_layout(
        xaxis=dict(range=[0, 1], tickformat=".0%", title="Probability"),
        yaxis=dict(title=""),
        showlegend=False,
        height=200 + 24 * len(probs_df),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top pushing/pulling tokens
    c1, c2 = st.columns(2)

    with c1:
        st.caption(f"Tokens pushing **{ex['label']}**")
        if ex["pos_contrib"]:
            st.table(pd.DataFrame(ex["pos_contrib"], columns=["token", "weight"]))
        else:
            st.write("—")

    with c2:
        st.caption(f"Tokens pulling away from **{ex['label']}**")
        if ex["neg_contrib"]:
            st.table(pd.DataFrame(ex["neg_contrib"], columns=["token", "weight"]))
        else:
            st.write("—")

    # Highlighted text
    st.text("")
    st.markdown("**Text with highlighted tokens**")
    st.write(ex["html_highlight"], unsafe_allow_html=True)

def render_sidebar_feedback():
    st.sidebar.markdown("---")
    st.sidebar.subheader("Leave a Feedback")
    user_feedback = st.sidebar.text_input("Enter Feedback")

    if st.sidebar.button("Submit", type="primary"):
        if user_feedback.strip():
            try:
                conn= get_db()
                add_feedback(conn, user_feedback.strip())
                fetch_feedback.clear()  # refresh cached list
                st.sidebar.success("Thank you for your valuable feedback!")
            except Exception as e:
                st.sidebar.error(f"Couldn't save feedback: {e}")
        else:
            st.sidebar.warning("Please enter feedback before submitting.")
    
    if st.sidebar.checkbox("Show previous feedback"):
        try:
            df = fetch_feedback()
            if df.empty:
                st.sidebar.info("No feedback yet.")
            else:
                st.sidebar.dataframe(df, use_container_width=True, height=250)
        except Exception as e:
            st.sidebar.error(f"Error loading feedback: {e}")

def _transform_with_optional_selector(vectorizer, selector, feats_str):
    X_raw = vectorizer.transform([feats_str])
    if selector is None:
        # No selector (just use vectorizer output and its full vocab)
        return X_raw, vectorizer.get_feature_names_out()

    # Has selector → apply and adjust vocab to selected features if possible
    X = selector.transform(X_raw)
    if hasattr(selector, "get_support"):
        mask = selector.get_support()
        vocab = vectorizer.get_feature_names_out()[mask]
    else:
        # Fallback name if selector doesn’t expose get_support
        vocab = np.array([f"f{i}" for i in range(X.shape[1])])
    return X, vocab

# Use it for every language in _vectorize_for_lang
def _vectorize_for_lang(text, lang):
    if lang == "en":
        _, _, _, lemmas = english_preprocess_text(text)
        feats = " ".join(lemmas)
        X, vocab = _transform_with_optional_selector(english_vectorizer, english_selector, feats)
        return X, english_clf, vocab, feats

    if lang == "ms":
        _, _, _, _, lemmas = melayu_preprocess_text(text)
        feats = " ".join(lemmas)
        X, vocab = _transform_with_optional_selector(bm_vectorizer, bm_selector, feats)
        
        return X, bm_clf, vocab, feats

    if lang == "zh-cn":
        _, _, toks_ns = mandarin_preprocess_text(text)
        feats = " ".join(toks_ns)
        X, vocab = _transform_with_optional_selector(mandarin_vectorizer, mandarin_selector, feats)
        
        return X, mandarin_clf, vocab, feats
    
    if lang in ("zh-tw", "zh-hant"):
        cleaned, _, toks_ns = cantonese_preprocess_text(text)
        # char analyzers ignore spaces; word analyzers expect them → this works for both
        feats = " ".join(toks_ns) if toks_ns else cleaned
        X, vocab = _transform_with_optional_selector(cantonese_vectorizer, cantonese_selector, feats)
        return X, cantonese_clf, vocab, feats

    return None

def transform_feats(lang, feats_str):
    """Transform a *preprocessed* features string with the same pipeline used at train time."""
    if lang == "en":
        X_raw = english_vectorizer.transform([feats_str])
        return english_selector.transform(X_raw) if english_selector is not None else X_raw
    if lang == "ms":
        return bm_vectorizer.transform([feats_str])
    if lang == "zh-cn":
        return mandarin_vectorizer.transform([feats_str])
    if lang in ("zh-tw", "zh-hant"):
        X_raw = cantonese_vectorizer.transform([feats_str])
        return cantonese_selector.transform(X_raw) if cantonese_selector is not None else X_raw
    
    return None

def _fallback_highlight_when_empty(text, lang, feats_str, vocab, top_k=8):
    vocab_set = set(vocab.tolist() if hasattr(vocab, "tolist") else vocab)
    toks = []

    if lang in ("zh-cn", "zh-tw", "zh-hant"):
        s = re.sub(r"\s+", "", feats_str)   # remove any spaces
        for n in (4, 3, 2, 1):              # try longer n-grams first
            for i in range(len(s) - n + 1):
                ngram = s[i:i+n]
                if ngram in vocab_set:
                    toks.append(ngram)
    else:
        toks = [t for t in feats_str.split() if t in vocab_set]

    toks = list(dict.fromkeys(toks))        # unique, keep order
    if not toks:
        return None
    dummy_pos = [(t, 1.0) for t in toks[:top_k]]
    return highlight_text(text, lang, dummy_pos, [])

def _softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    e = np.exp(z)
    return e / e.sum()

def get_probs_dict(clf, X):
    """Return {class_label: prob} using predict_proba or a softmax over decision_function."""
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[0]
        return dict(zip(clf.classes_, probs))
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        s = s[0] if np.ndim(s) > 1 else s
        probs = _softmax(s)
        return dict(zip(clf.classes_, probs))
    return None

def token_contributions(clf, X, vocab, top_k=8):
    """
    For linear models: contribution = feature_value * coefficient_for_predicted_class.
    Returns (pos_list, neg_list) where each is [(token, weight), ...].
    """
    if not hasattr(clf, "coef_"):  # non-linear or no coefficients
        return None

    # predicted class index
    pred_label = clf.predict(X)[0]
    class_idx = int(np.where(clf.classes_ == pred_label)[0][0])

    # sparse doc vector (indices & values)
    x = X.tocoo()
    cols, vals = x.col, x.data
    coefs = clf.coef_[class_idx]  # shape (n_features,)

    contrib = [(vocab[j], float(coefs[j] * val)) for j, val in zip(cols, vals)]
    contrib.sort(key=lambda t: abs(t[1]), reverse=True)
    pos = [(w, c) for w, c in contrib if c > 0][:top_k]
    neg = [(w, c) for w, c in contrib if c < 0][:top_k]
    return pos, neg

def highlight_text(original_text, lang, pos_contrib, neg_contrib):
    """Return HTML with tokens highlighted (green = pushes to label, red = against)."""
    toks = (pos_contrib or []) + (neg_contrib or [])
    if not toks:
        return html.escape(original_text)

    max_w = max(abs(wt) for _, wt in toks) or 1.0

    def style(wt):
        alpha = 0.35 + 0.65 * (abs(wt) / max_w)
        color = "46, 204, 113" if wt > 0 else "231, 76, 60"  # green OR red
        return f"background-color: rgba({color}, {alpha}); padding:0 3px; border-radius:3px;"

    # Replace longest tokens first to reduce overlapping issues
    sorted_tokens = sorted(toks, key=lambda t: len(t[0]), reverse=True)
    out = html.escape(original_text)

    for tok, wt in sorted_tokens:
        safe = html.escape(tok)
        # word boundary for space-delimited langs, plain for CJK
        if lang in ("en", "ms"):
            pattern = rf"(?i)\b{re.escape(safe)}\b"
        else:
            pattern = re.escape(safe)

        out = re.sub(pattern, f'<span style="{style(wt)}">{safe}</span>', out)
    return out

def _candidates_for_lang(lang, feats_str, text, vocab):
    """Return (candidates, remove_token_fn) tailored to the language & vocab."""
    vocab_set = set(vocab.tolist() if hasattr(vocab, "tolist") else vocab)

    if lang in ("zh-cn", "zh-tw", "zh-hant"):
        # Prefer segmented words that are in the vocab
        if lang == "zh-cn":
            _, _, seg = mandarin_preprocess_text(text)
        else:
            _, _, seg = cantonese_preprocess_text(text)
        cand = [t for t in dict.fromkeys(seg) if t and t in vocab_set]

        # If none, fall back to character n-grams that exist in the vocab
        if not cand:
            s = re.sub(r"\s+", "", feats_str)
            grams = []
            max_n = min(4, max(1, len(s)))
            for n in range(max_n, 0, -1):
                grams.extend(s[i:i+n] for i in range(len(s) - n + 1))
            cand = [g for g in dict.fromkeys(grams) if g in vocab_set]

        remove_token = lambda s, tok: s.replace(tok, "")
    else:
        toks = feats_str.split()
        cand = [t for t in dict.fromkeys(toks) if t in vocab_set]
        remove_token = lambda s, tok: " ".join(w for w in s.split() if w != tok)

    return cand, remove_token

def get_scores_dict(clf, X):
    """
    Return raw decision scores per class if available; otherwise a log-odds
    surrogate so the deltas are not tiny like probabilities.
    """
    # Prefer true margins
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        # s -> shape (n_classes,) or scalar for binary
        if np.ndim(s) == 0:
            # binary scalar margin -> expand to 2 classes
            return {clf.classes_[0]: float(-s), clf.classes_[1]: float(s)}
        s = np.asarray(s).ravel()
        return {c: float(v) for c, v in zip(clf.classes_, s)}

    # Fallback: log-odds from probabilities
    if hasattr(clf, "predict_proba"):
        p = np.asarray(clf.predict_proba(X)[0], dtype=float)
        eps = 1e-12
        if p.size == 2:
            # symmetric binary margin
            m = np.log((p[1] + eps) / (1 - p[1] + eps))
            return {clf.classes_[0]: float(-m), clf.classes_[1]: float(m)}
        m = np.log((p + eps) / (1 - p + eps))
        return {c: float(v) for c, v in zip(clf.classes_, m)}

    return None

def predict_with_explanations(text, lang, top_k=8):
    """
    Returns dict with:
      label, probs (dict or None), pos_contrib, neg_contrib, html_highlight
    Uses coef_ if available; otherwise falls back to leave-one-token-out.
    """
    pack = _vectorize_for_lang(text, lang)
    if not pack:
        return {"label": predict(text, lang), "probs": None,
                "pos_contrib": None, "neg_contrib": None,
                "html_highlight": html.escape(text)}

    X, clf, vocab, feats_str = pack
    label = clf.predict(X)[0]
    probs = get_probs_dict(clf, X)

    # Linear, weight-based explanation if available
    if hasattr(clf, "coef_"):
        pos, neg = token_contributions(clf, X, vocab, top_k=top_k)
        html_text = highlight_text(text, lang, pos, neg) if (pos or neg) else html.escape(text)
        return {"label": label, "probs": probs, "pos_contrib": pos, "neg_contrib": neg,
                "html_highlight": html_text}
    
    # Fallback (leave-one-token-out on the *preprocessed* tokens)
    pos = neg = None
    html_text = html.escape(text)

    base_scores = get_scores_dict(clf, X)
    if base_scores is not None and label in base_scores:
        base = float(base_scores[label])

        cand, remove_token = _candidates_for_lang(lang, feats_str, text, vocab)

        contrib = []
        for t in cand:
            reduced = remove_token(feats_str, t)
            X2 = transform_feats(lang, reduced)
            if X2 is None:
                continue
            s2 = get_scores_dict(clf, X2)
            if not s2 or label not in s2:
                continue
            delta = base - float(s2[label])  # >0 (means pushes toward label)
            contrib.append((t, delta))

        contrib.sort(key=lambda x: abs(x[1]), reverse=True)
        top = contrib[:top_k]
        pos = [(w, c) for w, c in top if c >= 0]
        neg = [(w, c) for w, c in top if c < 0]

        if top:
            html_text = highlight_text(text, lang, pos, neg)
        else:
            fallback_html = _fallback_highlight_when_empty(text, lang, feats_str, vocab, top_k)
            if fallback_html:
                html_text = fallback_html

    return {"label": label, "probs": probs, "pos_contrib": pos, "neg_contrib": neg,
            "html_highlight": html_text}


def main():
    st.title("Multilingual Sentiment Analysis for Identifying Mental Health Trends on Social Media")
    st.write(':gray[This application analyzes social media posts to detect potential mental health signals using multilingual text processing (English, Bahasa Melayu, Mandarin, OR Cantonese).]')
    st.sidebar.title("Multilingual Sentiment Analysis")
    st.sidebar.markdown("Let's identify mental health illness from social media posts")
    st.sidebar.subheader("Input Settings")
    mode = st.sidebar.radio("Mode", ["Upload CSV", "Enter Text"])

    render_sidebar_feedback()

    if mode == "Upload CSV":
        handle_upload_flow()
    else:
        handle_enter_text_flow()

    _ = get_db()


if __name__ == "__main__":

    main()

