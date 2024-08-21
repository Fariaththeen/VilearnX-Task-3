# clean_data.py

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK resources (run this once)
#nltk.download('punkt')
#nltk.download('stopwords')

# Load dataset
data = pd.read_csv('movie_reviews.csv')  # Adjust the path as needed

def clean_text(text):
    """Clean and preprocess the text data."""
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(tokens)

# Apply cleaning
data['cleaned_reviews'] = data['reviews'].apply(clean_text)

# Display original and cleaned reviews
print(data[['reviews', 'cleaned_reviews']].head())  # Display first few rows

# Save the cleaned dataset to a new CSV file
cleaned_file_path = 'cleaned_movie_reviews.csv'  # Specify the path for the new CSV file
data.to_csv(cleaned_file_path, index=False)  # Save DataFrame to CSV without the index
print(f'Cleaned dataset saved to {cleaned_file_path}')