import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Semantic Similarity Analyzer",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🧠 Semantic Similarity Analyzer")
st.markdown("""
Analyze semantic similarity between words, sentences, and short text using a
free pretrained NLP model.

**Model:** all-MiniLM-L6-v2
""")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("About")

st.sidebar.info("""
This application uses the pretrained
Sentence Transformer model:

all-MiniLM-L6-v2

No preprocessing.
No model training.
No paid API.
""")

# --------------------------------------------------
# INPUT
# --------------------------------------------------

default_text = """Artificial Intelligence is transforming education.
Machine Learning helps computers learn from data.
Deep Learning is a branch of AI.
Cats are domestic animals.
Dogs are loyal companions."""

user_input = st.text_area(
    "Enter one sentence per line",
    default_text,
    height=220
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# --------------------------------------------------
# BUTTON
# --------------------------------------------------

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

    # --------------------------------------------------
    # SIMILARITY DATAFRAME
    # --------------------------------------------------

    sim_df = pd.DataFrame(
        similarity_matrix,
        index=texts,
        columns=texts
    )

    # --------------------------------------------------
    # FIND BEST PAIR
    # --------------------------------------------------

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

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------

    col1, col2, col3 = st.columns(3)

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

    st.success(
        f"""
Most Similar Pair:

'{best_pair[0]}'

and

'{best_pair[1]}'

Similarity Score: {max_score:.4f}
"""
    )

    # --------------------------------------------------
    # TABLE
    # --------------------------------------------------

    st.subheader("Similarity Matrix")

    st.dataframe(
        sim_df,
        use_container_width=True
    )

    # --------------------------------------------------
    # GRAPH 1
    # BAR CHART
    # --------------------------------------------------

    st.subheader("📊 Graph 1: Similarity Scores")

    base_text = texts[0]

    bar_df = pd.DataFrame({
        "Text": texts,
        "Similarity": similarity_matrix[0]
    })

    fig_bar = px.bar(
        bar_df,
        x="Text",
        y="Similarity",
        title=f"Similarity Compared to First Sentence"
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

    # --------------------------------------------------
    # GRAPH 2
    # HEATMAP
    # --------------------------------------------------

    st.subheader("🔥 Graph 2: Similarity Heatmap")

    fig_heatmap = px.imshow(
        similarity_matrix,
        x=texts,
        y=texts,
        text_auto=".2f",
        aspect="auto",
        title="Pairwise Semantic Similarity"
    )

    st.plotly_chart(
        fig_heatmap,
        use_container_width=True
    )

    # --------------------------------------------------
    # GRAPH 3
    # PCA EMBEDDING
    # --------------------------------------------------

    st.subheader("🧭 Graph 3: 2D Embedding Visualization")

    pca = PCA(
        n_components=2
    )

    reduced = pca.fit_transform(
        embeddings
    )

    pca_df = pd.DataFrame({
        "PCA1": reduced[:, 0],
        "PCA2": reduced[:, 1],
        "Text": texts
    })

    fig_pca = px.scatter(
        pca_df,
        x="PCA1",
        y="PCA2",
        text="Text",
        title="2D Semantic Embedding Space"
    )

    fig_pca.update_traces(
        textposition="top center"
    )

    st.plotly_chart(
        fig_pca,
        use_container_width=True
    )

    # --------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------

    st.subheader("Download Results")

    csv = sim_df.to_csv()

    st.download_button(
        label="📥 Download Similarity Matrix CSV",
        data=csv,
        file_name="similarity_matrix.csv",
        mime="text/csv"
    )

    # --------------------------------------------------
    # CRITICAL THINKING
    # --------------------------------------------------

    st.subheader("Paul's Critical Thinking Standards")

    with st.expander("View Analysis"):

        st.markdown(f"""
### Clarity
The user entered **{len(texts)}** sentences and the model calculated semantic similarity between them.

### Accuracy
The analysis uses the pretrained NLP model **all-MiniLM-L6-v2**.

### Precision
The highest similarity score obtained was **{max_score:.4f}**.

### Relevance
The heatmap, bar chart, and embedding plot directly support the similarity results.

### Logic
Sentences with related meanings receive higher similarity scores and appear closer in the embedding space.

### Significance
The most significant relationship was found between:

**{best_pair[0]}**

and

**{best_pair[1]}**

### Fairness
The pretrained model may not fully understand domain-specific knowledge and may inherit biases from training data.
""")
