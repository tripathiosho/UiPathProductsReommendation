import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import pickle
import os

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load dataset from Excel
df_file = "uipath_products.xlsx"
if os.path.exists(df_file):
    df = pd.read_excel(df_file)
else:
    raise FileNotFoundError(f"{df_file} not found. Please ensure the file exists.")

# Handle missing values
df = df.fillna("")

# Preprocessing function
def preprocess(text):
    text = text.lower().strip()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Apply preprocessing to descriptions
df['Processed_Description'] = df['Description'].apply(preprocess)

# Generate Image Paths
def generate_image_path(product_name):
    file_name = product_name.replace(" ", "_") + ".png"
    return f"images/{file_name}"

df['ImagePath'] = df['Product Name'].apply(generate_image_path)

# Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['Processed_Description'])

# Save Preprocessed Data, Vectorizer, and TF-IDF Matrix
df.to_csv("uipath_products.csv", index=False)  # Save CSV for Streamlit

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

print("Data preprocessing complete. CSV and pickle files saved.")
