# -Sentiment-Analysis-in-Natural-Language-Processing
The sentiment analysis project leverages Natural Language Processing (NLP) and data analytics to determine the sentiment expressed in textual data, classifying it as positive, negative, or neutral. The project involved sourcing a comprehensive dataset from various platforms, followed by meticulous data cleaning and preprocessing techniques, such as stemming, lemmatization, and tokenization, to prepare the text for analysis. I have used the Amazon food Reviews dataset for the training. I developed and trained the sentiment analysis model using several machine learning algorithms, including Naive Bayes, Random Forest, and Support Vector Machines (SVM), ensuring a robust comparison of their performance. The model's effectiveness was rigorously evaluated using accuracy, precision, and recall metrics, and the results were visualized with data visualization libraries like Matplotlib and Seaborn to identify trends and insights. It also includes the collection, preprocessing, and curation of data, as well as the development, tuning, and optimization of the sentiment analysis model. Many big companies use this type of analytical system for easy differentiation of reviews and understanding the quality of products.

## Overview

This repository provides a complete Flask web application for sentiment analysis using multiple machine learning models. The app allows users to input any text and receive sentiment predictions (positive, neutral, negative) with confidence scores. The backend leverages scikit-learn, spaCy, NLTK, and Hugging Face Transformers, while the frontend offers a clean, user-friendly interface. The project also demonstrates how to visualize and interpret sentiment analysis results using various graphs and charts.

---

## Features

- **Web Interface:** Simple HTML forms for text input and result display.
- **Multiple ML Models:** Naive Bayes, Random Forest, and SVM for robust sentiment prediction.
- **Text Preprocessing:** Lemmatization, stopword removal, and negation handling using spaCy and NLTK.
- **TF-IDF Vectorization:** Converts text into numerical features for machine learning.
- **Model Persistence:** Trained models and vectorizer are saved and loaded via pickle.
- **Sentiment Visualization:** Guidance on using bar charts, pie charts, word clouds, sentiment arcs, and time series for deeper insights[2][3][5].
- **Extensible:** Easily add more models or advanced NLP pipelines.

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Visualization and Graphs](#visualization-and-graphs)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-flask-app.git
   cd sentiment-analysis-flask-app
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is missing, install manually:*
   ```bash
   pip install flask scikit-learn spacy nltk transformers
   python -m spacy download en_core_web_sm
   ```

4. **Download NLTK data:**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

---

## Project Structure

```
sentiment-analysis-flask-app/
│
├── app.py                  # Main Flask application
├── trained_models.pkl      # Serialized vectorizer and trained models (auto-created)
├── templates/
│   ├── index.html          # Home page template
│   └── result.html         # Results page template
├── static/                 # (Optional) For CSS, images, JS
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Usage

1. **Run the Flask app:**
   ```bash
   python app.py
   ```

2. **Access the web interface:**
   - Open your browser and go to `http://127.0.0.1:5000/`

3. **Analyze sentiment:**
   - Enter your text in the provided box and submit.
   - View the predicted sentiment scores for positive, neutral, and negative classes.

---

## How It Works

**1. Preprocessing:**
   - Text is lowercased, lemmatized, and stopwords are removed using spaCy.
   - Negations (e.g., "not happy") are handled to improve model accuracy.

**2. Feature Extraction:**
   - Text is converted to TF-IDF vectors, capturing important words and their weights.

**3. Model Training & Persistence:**
   - On first run, a sample dataset is used to train Naive Bayes, Random Forest, and SVM models.
   - Models and vectorizer are saved to `trained_models.pkl` for reuse.

**4. Prediction:**
   - User input is preprocessed and vectorized.
   - Each model predicts sentiment probabilities (positive, neutral, negative).
   - Scores are normalized and displayed.

**5. Web Interface:**
   - The app uses Flask routes and HTML templates for input and results.

---

## Visualization and Graphs

Visualizing sentiment analysis results is essential for interpretation and actionable insights. Here are common methods and their explanations:

### **1. Bar Charts**

- **Purpose:** Compare the frequency of positive, negative, and neutral sentiments.
- **How to Use:** Plot the count of each sentiment class for a dataset or over time.
- **Example:** 
  - X-axis: Sentiment classes (Positive, Neutral, Negative)
  - Y-axis: Number of reviews
- **Interpretation:** Quickly see which sentiment dominates your data[2].

### **2. Pie Charts / Sentiment Distribution Charts**

- **Purpose:** Show the proportion of each sentiment in your dataset.
- **How to Use:** Plot the percentage of positive, neutral, and negative sentiments.
- **Interpretation:** Understand the overall sentiment landscape at a glance[2].

### **3. Word Clouds**

- **Purpose:** Visualize the most frequent words in positive, neutral, or negative texts.
- **How to Use:** Generate a word cloud for each sentiment class; larger words appear more often.
- **Interpretation:** Identify key terms associated with each sentiment[2][5].

### **4. Sentiment Arcs**

- **Purpose:** Show how sentiment changes throughout a single text (e.g., a review).
- **How to Use:** Plot sentiment polarity for each sentence as a function of position in the text.
- **Interpretation:** Reveal emotional shifts and narrative flow within a review[3].

### **5. Time Series Analysis**

- **Purpose:** Track sentiment trends over time (e.g., before/after a product launch).
- **How to Use:** Plot average sentiment scores by date or time period.
- **Interpretation:** Detect shifts in customer mood or reactions to events[2][3].

### **6. Sentiment Polarity Histograms**

- **Purpose:** Show the distribution of sentiment scores for specific keywords or topics.
- **How to Use:** Plot histograms of sentiment polarity for texts containing certain terms.
- **Interpretation:** Compare how different topics are perceived[3].

---

### **Sample Visualization Code**

Below are code snippets for common sentiment analysis visualizations (using matplotlib and wordcloud):

#### **Bar Chart**
```python
import matplotlib.pyplot as plt

sentiments = ['Positive', 'Neutral', 'Negative']
counts = [120, 45, 35]
plt.bar(sentiments, counts, color=['green', 'grey', 'red'])
plt.xlabel('Sentiment')
plt.ylabel('Number of Reviews')
plt.title('Sentiment Distribution')
plt.show()
```

#### **Pie Chart**
```python
plt.pie(counts, labels=sentiments, autopct='%1.1f%%', colors=['green', 'grey', 'red'])
plt.title('Sentiment Proportion')
plt.show()
```

#### **Word Cloud**
```python
from wordcloud import WordCloud, STOPWORDS
text = " ".join(review for review in positive_reviews)
wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white").generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Reviews Word Cloud')
plt.show()
```

#### **Sentiment Arc**
```python
from textblob import TextBlob
import numpy as np

def get_sentiment_arc(text):
    sentences = TextBlob(text).sentences
    polarities = [s.sentiment.polarity for s in sentences]
    return polarities

arcs = [get_sentiment_arc(review) for review in reviews[:100]]
plt.figure(figsize=(10, 6))
for arc in arcs:
    plt.plot(np.linspace(0, 1, len(arc)), arc, alpha=0.1, color='blue')
plt.xlabel('Position in Review')
plt.ylabel('Sentiment Polarity')
plt.title('Sentiment Arcs')
plt.show()
```


---

## Customization

- **Expand Dataset:** Replace the sample data in `app.py` with a larger, real-world dataset for better results.
- **Add More Models:** Integrate deep learning models (e.g., LSTM, BERT) for advanced analysis[1].
- **Improve UI:** Edit HTML templates for branding or additional features.
- **API Integration:** Expose prediction as a REST API for programmatic access[4].

---

## Troubleshooting

- **spaCy Model Not Found:**  
  Run `python -m spacy download en_core_web_sm`
- **NLTK Data Not Found:**  
  Ensure `nltk.download('punkt')` and `nltk.download('stopwords')` have been executed.
- **Port Already in Use:**  
  Change the port in `app.run(debug=True, port=XXXX)` if needed.

---

## License

This project is open-source and free to use under the MIT License.

---

## Acknowledgments

- Inspired by [GeeksforGeeks Sentiment Analysis Flask Tutorial].
- Uses [scikit-learn], [spaCy], [NLTK], [Hugging Face Transformers].
- Visualization ideas from [V7 Labs][2], [Hex][3], and [Hugging Face][5].
- Sample project structures referenced from [GitHub sentiment analysis Flask apps][4].

---

**Happy Analyzing!**

[1] https://research.aimultiple.com/sentiment-analysis-machine-learning/
[2] https://www.v7labs.com/blog/ai-sentiment-analysis-definition-examples-tools
[3] https://hex.tech/templates/sentiment-analysis/sentiment-analysis/
[4] https://github.com/MLH-Fellowship/0.1.2-sentiment-analysis-visualization
[5] https://huggingface.co/blog/sentiment-analysis-python
[6] https://www.projectpro.io/article/sentiment-analysis-project-ideas-with-source-code/518
[7] https://getthematic.com/sentiment-analysis
[8] https://guides.lib.purdue.edu/d-velop/data-viz/r5
