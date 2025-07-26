# 🎬 Movie Recommender System

A content-based movie recommender web app built using **Streamlit**. This project uses a custom movie dataset and provides recommendations based on movie **overview similarity** using **TF-IDF vectorization** and **cosine similarity**.

![Banner](https://cdn-icons-png.flaticon.com/512/2844/2844205.png)

---

## 🔍 Features

- 🍿 Recommend top 5 similar movies based on a selected movie
- ✍️ Uses **TF-IDF** to understand movie overview content
- 🔗 Calculates **cosine similarity** between movies
- 📈 Beautiful, interactive Streamlit UI with CSS styling
- 🧾 Takes CSV input (Movie Dataset)

---

## 📂 Project Structure

movie-recommender-app/
├── app.py # Streamlit application
├── movies.csv # Movie metadata (input dataset)
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---

## 🛠️ Technologies Used

- Python 🐍
- pandas, NumPy 📊
- scikit-learn ⚙️
- Streamlit 🌐
- TF-IDF, Cosine Similarity 📐

---

## 🚀 How to Run Locally

1. **Clone the repository**

git clone https://github.com/PrathyushaThulava/movie-recommender-app.git
cd movie-recommender-app

2. **Install dependencies**

pip install -r requirements.txt

3. **Run the app**

streamlit run app.py

4. Open your browser and go to `http://localhost:8501/`

---

## 💡 Sample Output

If you select `Inception (2010)`
📽️ Top 5 Recommended Movies:

- Interstellar
- The Prestige
- Shutter Island
- The Matrix
- Memento

---

## 🌐 Live Demo

[👉 Try the App (Streamlit Cloud)](https://prathyushathulava-movie-recommender-app.streamlit.app)
_Note: Link will work after deployment_

---

## 🧑‍💻 Author

**Prathyusha Thulava**
[GitHub](https://github.com/PrathyushaThulava) • [LinkedIn](https://linkedin.com/in/prathyusha-thulava-514215254)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

⭐ If you like this project, consider giving it a **star** on GitHub!

---
