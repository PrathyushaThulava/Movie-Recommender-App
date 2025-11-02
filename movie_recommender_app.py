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

# ---- CSS for Style ----
def load_css():
    st.markdown("""
        <style>
        /* Background Gradient */
        body, .stApp {
            background: linear-gradient(120deg, #f9e1f2 0%, #c6dbf7 100%) !important;
            font-family: 'Poppins', 'Segoe UI', sans-serif;
            color: #222;
        }

        /* Title styling */
        h1 {
            text-align: center;
            color: #2b2d42;
            font-size: 2.2rem;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
            font-weight: 600;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.4);
            backdrop-filter: blur(8px);
            border-radius: 12px;
            padding: 1.2rem;
        }

        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label {
            color: #2b2d42 !important;
            font-weight: 600;
        }

        /* Movie cards */
        .movie-card {
            background: rgba(255, 255, 255, 0.85);
            border-radius: 14px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }

        .movie-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.15);
        }

        /* Recommendation section */
        .stSuccess {
            background: linear-gradient(90deg, #9be15d 0%, #00e3ae 100%) !important;
            border-radius: 10px;
            color: #00332e !important;
            font-weight: 600;
        }

        /* Markdown content */
        .stMarkdown p {
            font-size: 1rem;
            line-height: 1.6;
            color: #2b2d42;
        }

        /* Section headings */
        h3 {
            color: #3b3b98;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }

        /* Details bullets */
        ul {
            list-style-type: none;
            padding-left: 0;
        }

        ul li::before {
            content: "🎬 ";
            color: #ff6f91;
        }

        /* Divider line */
        hr {
            border: none;
            border-top: 2px solid rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)


# Call the CSS
load_css()

# ---- Load Data ----
@st.cache_data
@st.cache_data
def load_data():
    df = pd.read_csv("movies_data.csv", on_bad_lines='skip', encoding='utf-8')
    df.columns = df.columns.str.strip()  # Clean spaces
    # Drop unwanted index column if exists
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
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
st.markdown("<h1> Tollywood Movie Recommender</h1>", unsafe_allow_html=True)
st.markdown("---")

if selected_movie:
    recommendations = recommend_movies(selected_movie, df, similarity_matrix)

    st.success(f" Top Recommendations for **'{selected_movie}'**:")

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
        -  **Genre**: {selected_info['Genre']}
        -  **Certificate**: {selected_info['Certificate']}
        -  **Year**: {selected_info['Year']}
        -  **Runtime**: {selected_info['Runtime']}
        -  **Rating**: {selected_info['Rating']} ({selected_info['No.of.Ratings']} ratings)
    """)
