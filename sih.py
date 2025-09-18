import streamlit as st
import numpy as np
import pandas as pd
import base64
from transformers import pipeline
import plotly.express as px

st.set_page_config(
    page_title="E-Consultation Reviews : Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
page_bg="""
<style>
    .stApp {
        background-color: #0A192F; /* Dark slate background */
    }
</style>"""
st.markdown(page_bg,unsafe_allow_html=True)

HEADER_HTML = """
<div style="background: linear-gradient(90deg,#3cc0be); padding: 18px; border-radius: 12px;">
  <h1 style="color: #fff; margin:0 0 6px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto;">
    E-Consultation Reviews — Sentiment Dashboard
  </h1>
  <p style="color: #FFFDFF7; margin:0; font-family: Inter, sans-serif;">
    Upload a CSV of patient / user reviews, run quick sentiment analysis, explore insights, and export results.
  </p>
</div>
"""
st.markdown(HEADER_HTML,unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.header("Upload and Settings")
    
    st.markdown("Don't have a CSV?")
    st.markdown("Generate a small sample to test:")

    if st.button("Generate sample CSV"):
        sample=pd.DataFrame({
            "review_id":[1,2,3,4,5,6],
            "review_text":[ "Doctor was kind and explained everything clearly.",
                "I waited for an hour, service was poor.",
                "Great consultation, helped me a lot!",
                "Terrible experience, rude staff and wrong prescription.",
                "Okay experience — nothing special.",
                "Excellent follow-up and clear instructions."
            ]
        })

        csv_bytes=sample.to_csv(index=False).encode("utf-8")
        b64 = base64.b64encode(csv_bytes).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="sample_reviews.csv">Download sample_reviews.csv</a>'
        st.markdown(href, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CSV file with reviews", type=["csv"])  # File uploader widget

#HUGGING FACE LOGIC
def load_model():
    return pipeline("sentiment-analysis",model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_analyser=load_model()

LABEL_MAP={
    "LABEL_0":"Negative",
    "LABEL_1":"Neutral",
    "LABEL_2":"Positive",
}

def analyze_sentence(sent:str):
    if not isinstance(sent,str):
        sent=str(sent)
    result=sentiment_analyser(sent,truncation=True)[0]
    return LABEL_MAP[result["label"]],round(result["score"],3)

#---

if uploaded_file is None:
    st.info("Upload a CSV file to begin the analysis. Please ensure that the file contains at least one column with review text.")
    st.stop()

try:
    df=pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Error reading the CSV file :{e}")
    st.stop()

st.write(f"Uploaded {uploaded_file.name}")

text_cols=[c for c in df.columns if df[c].dtype==object or df[c].dtype=="string"]

if not text_cols:
    text_cols=list(df.columns)

text_column = st.selectbox("Select the column that contains review text", text_cols,
                           index=text_cols.index(text_cols[0]) if text_cols else 0)
st.markdown("---")

if st.button("Run Sentiment Analysis ▶️"):
    with st.spinner("Analyzing..."):
        result_df = df.copy()
        labels = []
        scores = []

        texts = result_df[text_column].fillna("").astype(str).tolist()
        results = sentiment_analyser(texts, truncation=True)

        for r in results:
            labels.append(LABEL_MAP[r["label"]])
            scores.append(round(r["score"], 3))

        result_df["sentiment_label"] = labels
        result_df["sentiment_score"] = scores

    st.success("Sentiment analysis complete!")

    # --- SUMMARY (moved inside) ---
    counts = result_df["sentiment_label"].value_counts().reindex(
        ["Positive", "Neutral", "Negative"], fill_value=0
    )
    total = counts.sum()
    pct = (counts / total * 100).round(2)

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    col1.metric("Total reviews", int(total))
    col2.metric("Positive (%)", f"{pct['Positive']}%", int(counts['Positive']))
    col3.metric("Neutral (%)", f"{pct['Neutral']}%", int(counts['Neutral']))
    col4.metric("Negative (%)", f"{pct['Negative']}%", int(counts['Negative']))

    st.markdown("---")

    st.subheader("Sentiment distribution")
    pie_df = counts.reset_index()
    pie_df.columns = ["sentiment", "count"]

    c1, c2 = st.columns([1, 1])
    with c1:
        fig_pie = px.pie(pie_df, values="count", names="sentiment", hole=0.45, title="Sentiment share")
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        fig_bar = px.bar(pie_df, x="sentiment", y="count", text="count", title="Counts by sentiment")
        fig_bar.update_layout(yaxis_title="Number of reviews")
        st.plotly_chart(fig_bar, use_container_width=True)