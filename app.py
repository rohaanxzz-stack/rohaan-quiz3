import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("NLP Text Similarity Analyzer")

model = SentenceTransformer('all-MiniLM-L6-v2')

st.write("Enter multiple words/sentences (one per line):")

user_input = st.text_area(
    "Input Text",
    "cat\nkitten\ndog\nanimal\ncar"
)

if st.button("Analyze Similarity"):

    texts = [x.strip() for x in user_input.split("\n") if x.strip()]

    embeddings = model.encode(texts)

    similarity_matrix = cosine_similarity(embeddings)

    st.subheader("Similarity Matrix")

    sim_df = pd.DataFrame(
        similarity_matrix,
        index=texts,
        columns=texts
    )

    st.dataframe(sim_df)

    # -------------------------
    # Graph 1: Bar Chart
    # -------------------------

    st.subheader("Graph 1: Top Similarities")

    base_text = texts[0]

    scores = similarity_matrix[0]

    bar_df = pd.DataFrame({
        "Text": texts,
        "Similarity": scores
    })

    fig1, ax1 = plt.subplots()
    ax1.bar(bar_df["Text"], bar_df["Similarity"])
    ax1.set_title("Similarity with First Input")
    plt.xticks(rotation=45)

    st.pyplot(fig1)

    # -------------------------
    # Graph 2: Heatmap
    # -------------------------

    st.subheader("Graph 2: Heatmap")

    fig2, ax2 = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        sim_df,
        annot=True,
        cmap="Blues",
        ax=ax2
    )

    st.pyplot(fig2)

    # -------------------------
    # Graph 3: PCA Plot
    # -------------------------

    st.subheader("Graph 3: 2D Embedding Plot")

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(embeddings)

    fig3, ax3 = plt.subplots()

    ax3.scatter(
        reduced[:, 0],
        reduced[:, 1]
    )

    for i, txt in enumerate(texts):
        ax3.annotate(
            txt,
            (reduced[i, 0], reduced[i, 1])
        )

    st.pyplot(fig3)

    # -------------------------
    # Critical Thinking
    # -------------------------

    st.subheader("Paul's Critical Thinking Standards")

    highest_score = sim_df.iloc[0, 1:].max()

    st.markdown(f"""
### Clarity
Input texts were analyzed using a pretrained NLP model.

### Accuracy
Model Used: all-MiniLM-L6-v2.

### Precision
Highest similarity score: **{highest_score:.4f}**

### Relevance
Graphs directly visualize similarity relationships.

### Logic
Texts with similar meanings appear closer and receive higher scores.

### Significance
The most important result is the highest similarity pair.

### Fairness
The model may not fully understand domain-specific or cultural context.
""")
