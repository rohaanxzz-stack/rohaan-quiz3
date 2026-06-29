# app.py


import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="Semantic Similarity Analyzer",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------------------------
# TITLE
# -----------------------------------------------------

st.title("🧠 Semantic Similarity Analyzer")

st.markdown("""
Analyze semantic similarity between words, sentences,
and short text using a free pretrained NLP model.

**Model Used:** all-MiniLM-L6-v2
""")

# -----------------------------------------------------
# SIDEBAR
# -----------------------------------------------------

st.sidebar.header("Application Settings")

threshold = st.sidebar.slider(
    "Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.70,
    step=0.01
)

st.sidebar.info(f"""
Current Threshold: {threshold}

Similarity scores above this value
are classified as Similar.
""")

# -----------------------------------------------------
# MODEL
# -----------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------------------------------
# INPUT
# -----------------------------------------------------

default_text = """Artificial Intelligence is transforming education.
Machine Learning helps computers learn from data.
Deep Learning is a branch of AI.
Cats are domestic animals.
Dogs are loyal companions.
Education is becoming smarter with AI."""

user_input = st.text_area(
    "Enter one sentence per line",
    default_text,
    height=220
)

# -----------------------------------------------------
# BUTTON
# -----------------------------------------------------

if st.button("Analyze Similarity"):

    texts = [
        line.strip()
        for line in user_input.split("\n")
        if line.strip()
    ]

    if len(texts) < 2:
        st.warning("Please enter at least two sentences.")
        st.stop()

    with st.spinner("Generating embeddings..."):

        embeddings = model.encode(texts)

        similarity_matrix = cosine_similarity(
            embeddings
        )

    sim_df = pd.DataFrame(
        similarity_matrix,
        index=texts,
        columns=texts
    )

    # -----------------------------------------------------
    # BEST PAIR
    # -----------------------------------------------------

    max_score = -1
    best_pair = ("", "")

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):

            if similarity_matrix[i][j] > max_score:

                max_score = similarity_matrix[i][j]

                best_pair = (
                    texts[i],
                    texts[j]
                )

    # -----------------------------------------------------
    # DASHBOARD
    # -----------------------------------------------------

    similar_pairs = 0

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):

            if similarity_matrix[i][j] >= threshold:
                similar_pairs += 1

    st.subheader("📈 Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Sentences",
        len(texts)
    )

    col2.metric(
        "Embedding Size",
        embeddings.shape[1]
    )

    col3.metric(
        "Highest Similarity",
        f"{max_score:.4f}"
    )

    col4.metric(
        "Pairs Above Threshold",
        similar_pairs
    )

    st.success(
        f"""
Most Similar Pair:

{best_pair[0]}

AND

{best_pair[1]}

Similarity Score = {max_score:.4f}
"""
    )

    # -----------------------------------------------------
    # SIMILARITY MATRIX
    # -----------------------------------------------------

    st.subheader("📋 Similarity Matrix")

    st.dataframe(
        sim_df,
        use_container_width=True
    )

    # -----------------------------------------------------
    # THRESHOLD ANALYSIS
    # -----------------------------------------------------

    st.subheader("🎯 Threshold Analysis")

    threshold_results = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):

            score = similarity_matrix[i][j]

            threshold_results.append({
                "Sentence 1": texts[i],
                "Sentence 2": texts[j],
                "Similarity": round(score, 4),
                "Status":
                    "Similar"
                    if score >= threshold
                    else "Not Similar"
            })

    threshold_df = pd.DataFrame(
        threshold_results
    )

    st.dataframe(
        threshold_df,
        use_container_width=True
    )

    # -----------------------------------------------------
    # TOP MATCHES
    # -----------------------------------------------------

    st.subheader("🏆 Top Matching Pairs")

    top_df = threshold_df.sort_values(
        by="Similarity",
        ascending=False
    )

    st.dataframe(
        top_df.head(10),
        use_container_width=True
    )

    # -----------------------------------------------------
    # BAR CHART
    # -----------------------------------------------------

    st.subheader("📊 Graph 1: Similarity Scores")

    base_scores = pd.DataFrame({
        "Sentence": texts,
        "Similarity": similarity_matrix[0]
    })

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(
        base_scores["Sentence"],
        base_scores["Similarity"]
    )

    ax1.set_title(
        "Similarity Compared to First Sentence"
    )

    plt.xticks(rotation=45)

    st.pyplot(fig1)

    # -----------------------------------------------------
    # HEATMAP
    # -----------------------------------------------------

    st.subheader("🔥 Graph 2: Similarity Heatmap")

    fig2, ax2 = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        sim_df,
        annot=True,
        cmap="Blues",
        ax=ax2
    )

    st.pyplot(fig2)

    # -----------------------------------------------------
    # PCA
    # -----------------------------------------------------

    st.subheader("🧭 Graph 3: PCA Embedding Plot")

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(
        embeddings
    )

    fig3, ax3 = plt.subplots(
        figsize=(8, 6)
    )

    ax3.scatter(
        reduced[:, 0],
        reduced[:, 1]
    )

    for i, txt in enumerate(texts):

        ax3.annotate(
            txt,
            (
                reduced[i, 0],
                reduced[i, 1]
            )
        )

    ax3.set_title(
        "2D Semantic Embedding Space"
    )

    st.pyplot(fig3)

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    st.subheader("📊 Similarity Classification")

    similar_count = len(
        threshold_df[
            threshold_df["Status"] == "Similar"
        ]
    )

    not_similar_count = len(
        threshold_df[
            threshold_df["Status"] == "Not Similar"
        ]
    )

    pie_df = pd.DataFrame({
        "Category": [
            "Similar",
            "Not Similar"
        ],
        "Count": [
            similar_count,
            not_similar_count
        ]
    })

    fig4, ax4 = plt.subplots()

    ax4.pie(
        pie_df["Count"],
        labels=pie_df["Category"],
        autopct="%1.1f%%"
    )

    ax4.set_title(
        "Similarity Classification"
    )

    st.pyplot(fig4)

    # -----------------------------------------------------
    # THRESHOLD DECISION
    # -----------------------------------------------------

    st.subheader("🚦 Threshold Decision")

    if max_score >= threshold:

        st.success(
            f"""
Highest Similarity
({max_score:.4f})

EXCEEDS

Threshold
({threshold})
"""
        )

    else:

        st.error(
            f"""
Highest Similarity
({max_score:.4f})

IS BELOW

Threshold
({threshold})
"""
        )

    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    st.subheader("📥 Download Results")

    csv = sim_df.to_csv()

    st.download_button(
        label="Download Similarity Matrix",
        data=csv,
        file_name="similarity_matrix.csv",
        mime="text/csv"
    )

    # -----------------------------------------------------
    # PAUL'S CRITICAL THINKING
    # -----------------------------------------------------

    st.subheader("📖 Paul's Critical Thinking Standards")

    with st.expander("View Analysis"):

        st.markdown(f"""
### Clarity
The user entered {len(texts)} sentences and the system measured semantic similarity.

### Accuracy
The pretrained model used is all-MiniLM-L6-v2.

### Precision
The highest similarity score is {max_score:.4f}.

### Relevance
The dashboard, charts, and matrix directly support the similarity analysis.

### Logic
Sentences with related meanings receive higher similarity scores.

### Significance
The most important result is the strongest matching pair:

**{best_pair[0]}**

and

**{best_pair[1]}**

### Fairness
The model may not fully understand specialized domains and may inherit bias from training data.

### Threshold Used
The selected threshold is {threshold}.
Similarity scores above this value are considered semantically related.
""")

