# sentiment_analysis.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the cleaned dataset
cleaned_data = pd.read_csv('cleaned_movie_reviews.csv')  # Adjust the path if necessary

# Prepare features and labels
X = cleaned_data['cleaned_reviews']
y = cleaned_data['sentiment']  # Assuming sentiment is already encoded (0 for negative, 1 for positive)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Visualization

# 1. Distribution of Sentiment
plt.figure(figsize=(8, 5))
sns.countplot(data=cleaned_data, x='sentiment', palette='viridis')
plt.title('Distribution of Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'])
plt.show()

# 2. Word Cloud of Cleaned Reviews
all_reviews = ' '.join(cleaned_data['cleaned_reviews'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # Hide axes
plt.title('Word Cloud of Cleaned Reviews')
plt.show()

# 3. Sentiment Over Time (if you have a date column)
# Uncomment and modify the following lines if your dataset has a date column
# cleaned_data['date'] = pd.to_datetime(cleaned_data['date'])  # Convert to datetime if necessary
# cleaned_data.set_index('date', inplace=True)
# daily_sentiment = cleaned_data.resample('D').sentiment.value_counts().unstack().fillna(0)
# plt.figure(figsize=(12, 6))
# daily_sentiment.plot(kind='line', marker='o')
# plt.title('Sentiment Over Time')
# plt.xlabel('Date')
# plt.ylabel('Count')
# plt.legend(['Negative', 'Positive'])
# plt.show()