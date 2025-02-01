import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import nltk
from nltk.corpus import stopwords
import os

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load preprocessed dataset
df = pd.read_csv('uipath_products.csv')

# Load saved vectorizer and TF-IDF matrix
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

# Preprocessing function
def preprocess(text):
    text = text.lower().strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Recommendation function
def recommend_products(user_input, top_n=4):
    processed_input = preprocess(user_input)
    input_vector = vectorizer.transform([processed_input])
    cosine_similarities = cosine_similarity(input_vector, tfidf_matrix).flatten()
    df['Similarity'] = cosine_similarities
    recommended_products = df.sort_values(by='Similarity', ascending=False).head(top_n)
    return recommended_products[['Product Name', 'Description', 'Category', 'ImagePath']]

# Streamlit UI
st.title('UiPath Product Recommendation System')
user_input = st.text_area("Describe your business case:")

if st.button('Recommend'):
    if user_input:
        recommendations = recommend_products(user_input)

        # Display results as cards in columns
        st.write("### Recommended Products:")
        cols = st.columns(len(recommendations))  # Create dynamic columns

        for col, (_, row) in zip(cols, recommendations.iterrows()):
            with col:
                st.subheader(row['Product Name'])
                st.write(f"**Category:** {row['Category']}")
                st.write(f"**Description:** {row['Description']}")

                # Display Image (if exists)
                if os.path.exists(row['ImagePath']):
                    st.image(row['ImagePath'], caption=row['Product Name'], use_container_width=True)
                else:
                    st.write("*Image not available.*")

    else:
        st.write("Please enter a business case description.")
