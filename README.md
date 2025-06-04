# -Sentiment-Analysis-in-Natural-Language-Processing
The sentiment analysis project leverages Natural Language Processing (NLP) and data analytics to determine the sentiment expressed in textual data, classifying it as positive, negative, or neutral. The project involved sourcing a comprehensive dataset from various platforms, followed by meticulous data cleaning and preprocessing techniques, such as stemming, lemmatization, and tokenization, to prepare the text for analysis. I have used the Amazon food Reviews dataset for the training. I developed and trained the sentiment analysis model using several machine learning algorithms, including Naive Bayes, Random Forest, and Support Vector Machines (SVM), ensuring a robust comparison of their performance. The model's effectiveness was rigorously evaluated using accuracy, precision, and recall metrics, and the results were visualized with data visualization libraries like Matplotlib and Seaborn to identify trends and insights. It also includes the collection, preprocessing, and curation of data, as well as the development, tuning, and optimization of the sentiment analysis model. Many big companies use this type of analytical system for easy differentiation of reviews and understanding the quality of products.

---

# 🧠 Sentiment Analysis Web App with NLTK, VADER, RoBERTa & Flask

This project is a comprehensive **Sentiment Analysis** tool that combines classical and modern Natural Language Processing (NLP) techniques to analyze product reviews from Amazon. The app uses:

* **VADER (Valence Aware Dictionary and sEntiment Reasoner)** from NLTK
* **RoBERTa (Robustly Optimized BERT)** from HuggingFace Transformers
* **Visualization** with `Matplotlib` and `Seaborn`
* **Web Interface** using `Flask`

---

## 📁 Project Structure

```
├── app.py                  # Flask application
├── templates/
│   ├── index.html          # Input form page
│   └── result.html         # Sentiment output page
├── static/
│   └── styles.css          # Styling for HTML pages
├── Reviews.csv             # Amazon reviews dataset
├── sentiment_analysis.ipynb# Core NLP + sentiment logic
├── README.md               # Project documentation
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the application**

```bash
python app.py
```

4. **Open in your browser**

```
http://127.0.0.1:5000/
```

---

## 📊 Dataset

We use the first 500 rows from a subset of the [Amazon Fine Food Reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). Each review has the following fields:

* `Id`: Unique ID
* `Score`: Star rating (1–5)
* `Text`: Review text

---

## 📈 Data Visualizations & Analysis

### 1. **Count of Reviews by Stars**

```python
df['Score'].value_counts().sort_index().plot(kind='bar', title='Count of Reviews by Stars')
```

#### 🔍 Interpretation:

* X-axis: Star ratings from 1 to 5.
* Y-axis: Number of reviews.
* Helps identify class imbalance in the dataset. For instance, more 5-star reviews may indicate customer bias.

---

### 2. **VADER Sentiment Scoring**

Each review is passed through VADER's `SentimentIntensityAnalyzer` to get:

* `compound`: Normalized sentiment score between -1 (negative) and +1 (positive).
* `pos`, `neu`, `neg`: Proportions of positive, neutral, and negative sentiments.

```python
sns.barplot(data=vaders, x='Score', y='compound')
```

#### 🔍 Interpretation:

* Positive correlation expected between high star ratings and compound scores.
* Reviews with 1–2 stars often show negative compound scores.

---

### 3. **VADER Pos/Neu/Neg Score by Star**

```python
fig, axs = plt.subplots(1, 3)
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
```

#### 🔍 Interpretation:

* **Positive sentiment** increases with star rating.
* **Negative sentiment** spikes at 1-2 star reviews.
* Neutral sentiment stays relatively flat but higher for mid-range scores.

---

### 4. **RoBERTa Sentiment Analysis**

Used pre-trained `cardiffnlp/twitter-roberta-base-sentiment` model from HuggingFace.

```python
scores_dict = {
    'roberta_neg': scores[0],
    'roberta_neu': scores[1],
    'roberta_pos': scores[2]
}
```

#### 🔍 How it helps:

Provides a transformer-based understanding of sentiment, more robust to complex sentences and sarcasm compared to rule-based systems like VADER.

---

### 5. **Pairplot: VADER vs RoBERTa**

```python
sns.pairplot(data=results_df, vars=[...], hue='Score', palette='tab10')
```

#### 🔍 Interpretation:

* Compares all six sentiment scores (three from each model).
* Highlights clustering patterns (e.g., 5-star reviews tend to have high `vader_pos` and `roberta_pos`).

---

## 💬 Web Interface (Flask App)

### Input Form (index.html)

* Simple text area for user to enter a review.
* Submits to `/analyze`.

### Output Page (result.html)

* Displays:

  * The original sentence
  * Predicted sentiment label (`POSITIVE`, `NEGATIVE`, etc.)
  * Confidence score

### Styling (styles.css)

* Gradient background
* Styled text area and button
* Responsive layout for better UX

---

## 🧪 Example Sentences & Outputs

| Sentence                                                      | Label    | Score |
| ------------------------------------------------------------- | -------- | ----- |
| "I love sentiment analysis!"                                  | POSITIVE | 0.99  |
| "This is the worst thing ever."                               | NEGATIVE | 0.99  |
| "I did not like the vehicle's color I ordered"                | NEGATIVE | 0.97  |
| "I liked the shoes I ordered. The leather is of good quality" | POSITIVE | 0.98  |

---

## 🧠 Key Learnings

* **VADER** is fast and useful for short, social-media-style text.
* **RoBERTa** provides deeper linguistic understanding.
* **Combining models** allows for more robust sentiment classification.
* **Visualizations** help validate and explain model behavior.
* **Flask integration** makes the NLP logic accessible to end-users.

---

## ✅ Future Improvements

* Fine-tune a custom transformer model for product reviews.
* Add multi-language support using multilingual transformers.
* Use Bootstrap or Tailwind CSS for enhanced UI.
* Store analysis results in a database.

---

## 📜 License

MIT License. Feel free to use and modify with attribution.

---

## 🙌 Acknowledgements

* [NLTK](https://www.nltk.org/)
* [HuggingFace Transformers](https://huggingface.co/transformers/)
* [Matplotlib & Seaborn](https://seaborn.pydata.org/)
* [Flask](https://flask.palletsprojects.com/)
* [Amazon Review Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)

---

##Requirements.txt 


