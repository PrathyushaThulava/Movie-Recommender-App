import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page config
st.set_page_config(
    page_title="🎬 Tollywood Movie Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- 💅 Custom CSS for Style ----
def load_css():
    st.markdown("""
        <style>
        /* Gradient Background */
        body, .stApp {
            background: linear-gradient(120deg, #fbc2eb 0%, #a6c1ee 100%) !important;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Centered title */
        h1 {
            text-align: center;
            color: #2b2d42;
        }

        /* Sidebar style */
        .st-emotion-cache-1cypcdb {
            background-color: #ffffff30 !important;
            padding: 1rem;
            border-radius: 10px;
        }

        /* Card style for recommendations */
        .movie-card {
            background-color: #ffffffcc;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        .movie-card:hover {
            background-color: #ffffffee;
        }

        /* Markdown font styling */
        .stMarkdown p {
            font-size: 16px;
            color: #333;
        }
        </style>
    """, unsafe_allow_html=True)

# Call the CSS
load_css()

# ---- Load Data ----
@st.cache_data
def load_data():
    df = pd.read_csv("movies_data.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# ---- Similarity Function ----
def create_similarity_matrix(data):
    tfidf = TfidfVectorizer(stop_words='english')
    data['Overview'] = data['Overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(data['Overview'])
    return cosine_similarity(tfidf_matrix)

similarity_matrix = create_similarity_matrix(df)

# ---- Sidebar ----
st.sidebar.title("🎥 Select a Movie")
selected_movie = st.sidebar.selectbox("Choose a movie to get recommendations", sorted(df['Movie'].dropna().unique()))

# ---- Recommendation Logic ----
def recommend_movies(movie_title, data, similarity_matrix, top_n=5):
    if movie_title not in data['Movie'].values:
        return ["Movie not found in dataset."]
    
    idx = data[data['Movie'] == movie_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in similarity_scores[1:top_n+1]]
    return data['Movie'].iloc[top_indices].tolist()

# ---- Main App Layout ----
st.markdown("<h1>🎬 Tollywood Movie Recommender</h1>", unsafe_allow_html=True)
st.markdown("---")

if selected_movie:
    recommendations = recommend_movies(selected_movie, df, similarity_matrix)

    st.success(f"🎯 Top Recommendations for **'{selected_movie}'**:")

    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
            <div class="movie-card">
                <strong>{i}. {rec}</strong>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    selected_info = df[df['Movie'] == selected_movie].iloc[0]
    st.markdown("### 📌 Overview")
    st.markdown(f"{selected_info['Overview']}")

    st.markdown("### 🎞 Movie Details")
    st.markdown(f"""
        - 🎭 **Genre**: {selected_info['Genre']}
        - 🎓 **Certificate**: {selected_info['Certificate']}
        - 📆 **Year**: {selected_info['Year']}
        - ⏰ **Runtime**: {selected_info['Runtime']}
        - ⭐ **Rating**: {selected_info['Rating']} ({selected_info['No.of.Ratings']} ratings)
    """)
