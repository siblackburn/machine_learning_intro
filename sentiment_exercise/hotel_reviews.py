import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.datasets import load_digits
import seaborn as sns
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk

#https://www.kaggle.com/datafiniti/hotel-reviews#7282_1.csv
raw_data = pd.read_csv('sentiment_exercise/data/7282_1.csv')


# transform data. Drop na's in ratings, round the rating to an int, get rid of anything less than 0 etc, then make 1's and 2's bad, and 4's and 5's good
df_new = raw_data.dropna(subset=['reviews.rating'])
df_new['reviews.rating'] = df_new['reviews.rating'].apply(lambda x: int(round(x)))
df_new = df_new[(df_new['reviews.rating'] > 0) & (df_new['reviews.rating'] <= 5) & (df_new['reviews.rating'] != 3)]
df_new['reviews.binary'] = df_new['reviews.rating'].apply(lambda x: 'good' if x > 3 else 'bad')
print(df_new['reviews.binary'].value_counts())

#explore data:
# print(raw_data.head())
# print(raw_data.dtypes)
# print(raw_data.columns)
print("Raw data has ", len(raw_data))
print("Cleansed data has", len(df_new))


#pick features and target. 
features = df_new['reviews.text']
target = df_new['reviews.binary']

# Manipulate data into right format. STILL NEED TO APPLY STEMMER!!!!
for cells in features:
    cells = str(cells)
    cells = re.sub(r'\W', ' ', cells)
    cells = cells.lower()
    cells = re.sub(r'\s+[a-zA-Z]\s+', ' ', cells)
    

# Convert features to a string for the classifier to learn from
features = features.apply(str)

#convert ratings into integers (to round) and then convert to string so the classifier treats each rating as a class
sns.countplot(target)
plt.show()
plt.close()

print(target.value_counts())

target = target.astype(str)

X, y = features, target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
def evaluate_classifier(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)

    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))
    # print(accuracy_score(y_test, y_pred))



from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords


#inner_classifier = RandomForestClassifier(n_estimators=400)
inner_classifier = SGDClassifier(
    loss='hinge', 
    penalty='l2',
    alpha=1e-3, 
    random_state=42,
    max_iter=10,
    tol=None
)

#count vectorizer converts document into numbers
# 
#max df = ignore terms that appear in x% of documents
# min df = Ignore terms that appear in less than x documents
#Pipeline convert the processes into a superclassifier. step 1) countvectoriser to convert to numbers, step 2 tfidf counts the number of words into a massive grid for then step 3 to run a classifier
classifier = Pipeline([
    ('vect', CountVectorizer(
        max_features=5000,
        min_df=2,
        max_df=0.8, 
        stop_words=stopwords.words('english'),
        analyzer='word',
        strip_accents='unicode',
        ngram_range=(1, 3)
    )),
    ('tfidf', TfidfTransformer()),
    ('classifier', inner_classifier)
])

#run the classifier
classifier = classifier.fit(X_train, y_train)

evaluate_classifier(classifier, X_test, y_test)


# discover what the score is!!!!
train_score=classifier.score(X_train, y_train)
print('training score is: ', train_score)

test_score=classifier.score(X_test, y_test)
print('test score is: ', test_score)