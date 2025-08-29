import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# -----------------------------
# Load model and vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    vectorizer, model = joblib.load("baseline_tfidf_logreg.joblib")
    return vectorizer, model

vectorizer, model = load_model()
LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

# -----------------------------
# Prediction function
# -----------------------------
def predict_toxicity(texts):
    X = vectorizer.transform(texts)
    probs = model.predict_proba(X)
    results = []
    for i, t in enumerate(texts):
        row = {"comment": t}
        for j, lbl in enumerate(LABELS):
            row[lbl] = float(probs[i][j])
        results.append(row)
    return pd.DataFrame(results)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Comment Toxicity Detector", layout="wide")
st.title("🧹 Comment Toxicity Detection Dashboard")

menu = ["Single Comment", "Batch Upload", "Data Insights", "Model Performance", "Sample Test Cases"]
choice = st.sidebar.radio("Choose Section", menu)

# -----------------------------
# 1. Single Comment
# -----------------------------
if choice == "Single Comment":
    st.subheader("🔹 Test a Single Comment")
    user_input = st.text_area("Enter a comment:")
    if st.button("Predict"):
        if user_input.strip():
            preds = predict_toxicity([user_input])
            st.dataframe(preds.set_index("comment").T.style.background_gradient(cmap="Reds"))
        else:
            st.warning("Please enter a comment!")

# -----------------------------
# 2. Batch Upload
# -----------------------------
elif choice == "Batch Upload":
    st.subheader("🔹 Upload CSV for Batch Predictions")
    uploaded = st.file_uploader("Upload CSV with a 'comment_text' column", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if "comment_text" in df.columns:
            preds = predict_toxicity(df["comment_text"].astype(str).tolist())
            st.dataframe(preds.head(20))
            csv_out = preds.to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions", csv_out, "toxicity_predictions.csv", "text/csv")
        else:
            st.error("CSV must have a column named 'comment_text'.")

# -----------------------------
# 3. Data Insights
# -----------------------------
elif choice == "Data Insights":
    st.subheader("📊 Dataset Insights")

    try:
        df = pd.read_csv("train_split.csv")
        st.write("### Label Distribution")
        label_counts = {c: int((df[c]==1).sum()) for c in LABELS}
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x=list(label_counts.keys()), y=list(label_counts.values()), ax=ax)
        ax.set_ylabel("Positive samples")
        st.pyplot(fig)

        st.write("### Comment Length Distribution")
        df["_len"] = df["comment_text"].astype(str).map(len)
        fig, ax = plt.subplots(figsize=(8,5))
        df["_len"].clip(upper=df["_len"].quantile(0.99)).hist(bins=50, ax=ax)
        ax.set_xlabel("Characters")
        st.pyplot(fig)
    except Exception as e:
        st.warning("Upload `train_split.csv` for insights. Error: " + str(e))

# -----------------------------
# 4. Model Performance
# -----------------------------
elif choice == "Model Performance":
    st.subheader("📈 Model Performance Metrics")
    try:
        valid_df = pd.read_csv("valid_split.csv")
        X_val = vectorizer.transform(valid_df["cleaned_text"].fillna(""))
        y_val = valid_df[LABELS].values
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)

        report = classification_report(y_val, y_pred, target_names=LABELS, output_dict=True, zero_division=0)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report.style.background_gradient(cmap="Blues"))

        st.write("### ROC-AUC Scores")
        aucs = {}
        for i, lbl in enumerate(LABELS):
            try:
                aucs[lbl] = roc_auc_score(y_val[:,i], y_pred_proba[:,i])
            except:
                aucs[lbl] = np.nan
        auc_df = pd.DataFrame.from_dict(aucs, orient="index", columns=["ROC-AUC"])
        st.dataframe(auc_df.style.background_gradient(cmap="Greens"))
    except Exception as e:
        st.warning("Upload `valid_split.csv` for performance metrics. Error: " + str(e))

# -----------------------------
# 5. Sample Test Cases
# -----------------------------
elif choice == "Sample Test Cases":
    st.subheader("📝 Predefined Sample Comments")
    examples = [
        "You are an idiot!",
        "I really love this project, great work!",
        "This is the worst thing ever",
        "Have a nice day friend",
        "I will kill you",
        "You're such a genius!"
    ]
    preds = predict_toxicity(examples)
    st.dataframe(preds.set_index("comment").style.background_gradient(cmap="Oranges"))
