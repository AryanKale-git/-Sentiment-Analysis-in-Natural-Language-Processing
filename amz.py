import pandas as pd
import numpy as np
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
from tqdm import tqdm  # Use the standard tqdm import
import pandas as pd
from tqdm import tqdm
import pandas as pd

df = pd.DataFrame({'col1': range(10)})
tqdm.pandas(desc="Processing")  # Enable progress_apply
df['new_col'] = df['col1'].progress_apply(lambda x: x * 2)
df = pd.DataFrame({'col1': range(10)})
for i, row in tqdm(df.iterrows(), total=len(df)):

    df = pd.read_csv('C:\\Users\\kalea\\Downloads\\Reviews.csv.csv')
    print(df.shape)
    df = df.head(500)
    print(df.shape)
df.head()

#1. Count of Reviews by Stars
#X-axis: The review star ratings (e.g., 1, 2, 3, 4, 5).
#Y-axis: The count of reviews for each star rating.
ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()

example = df['Text'][50]
print(example)

tokens = nltk.word_tokenize(example)
tokens[:10]

tagged = nltk.pos_tag(tokens)
tagged[:10]

entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
sia = SentimentIntensityAnalyzer()
sia.polarity_scores('I am so happy!')
sia.polarity_scores('This is the worst thing ever.')
sia.polarity_scores(example)

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
vaders.head()

#2. Compound Score by Amazon Star Review
#X-axis: Amazon star ratings (1 to 5).
#Y-axis: The compound sentiment score (ranges from -1 to 1)
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()

#3. Positive, Neutral, and Negative Sentiment by Star Rating
#X-axis: Amazon star ratings (1 to 5).
#Y-axis: Sentiment score (calculated by Vader).
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

print(example)
sia.polarity_scores(example)

encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')
results_df.columns

#4. Pair Plot of Sentiment Scores

#Purpose: This pair plot compares the sentiment scores from Vader and RoBERTa models for each star rating.

#Axes:Pairwise relationships between vader_neg, vader_neu, vader_pos (Vader scores) and roberta_neg, roberta_neu, roberta_pos (RoBERTa scores).
#The color (hue) represents the star ratings.

sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()
#High vader_pos and roberta_pos scores may correspond to 5-star ratings.
#High vader_neg and roberta_neg scores may correspond to 1-star ratings.

results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]
results_df.query('Score == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0]
results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]
results_df.query('Score == 5') \
    .sort_values('vader_neg', ascending=False)['Text'].values[0]

from transformers import pipeline
from flask import Flask, render_template, request

sent_pipeline = pipeline("sentiment-analysis")
result1 = sent_pipeline('I love sentiment analysis!')
result2 = sent_pipeline('Make sure to like and subscribe!')
result3 = sent_pipeline('booo')
result4 = sent_pipeline('I hate to do projects')
result5 = sent_pipeline('the headphones that I ordered were worst')
result6 = sent_pipeline('I did not like the vehicles colour I ordered')
result7 = sent_pipeline('i liked the shoes i ordered from amazon.The leather is of good quality')

print("Sentiment Analysis Results:")
print(f"Sentence: 'I love sentiment analysis!' => Sentiment: {result1}")
print(f"Sentence: 'Make sure to like and subscribe!' => Sentiment: {result2}")
print(f"Sentence: 'booo' => Sentiment: {result3}")
print(f"sentence: 'I hate to do projects' => Sentiment: {result4}")
print(f"sentence: 'the headphones that I ordered were worst' => Sentiment: {result5}")
print(f"sentence: 'I did not like the vehicles colour I ordered' => Sentiment: {result6}")
print(f"sentence: 'i liked the shoes i ordered from amazon.The leather is of good quality' => Sentiment: {result7}")

app = Flask(__name__)
# Load the sentiment analysis pipeline once when the app starts
sent_pipeline = pipeline("sentiment-analysis")
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/analyze', methods=['POST'])
def analyze():
    user_review = request.form['review']  # Get the user input from the form
    sentiment = sent_pipeline(user_review)  # Analyze the sentiment of the user input
    results = [{
        'sentence': user_review,
        'sentiment': sentiment[0]  # Get the first result
    }]
    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

