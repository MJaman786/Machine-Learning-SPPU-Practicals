
import pandas as pd

df = pd.read_csv('/content/SMSSpamCollection', sep='\t',
                 names=['label','text'])

df.head()

df.shape

!pip install nltk

import nltk

nltk.download('stopwords')
nltk.download('punkt')

sentence = "Hello frineds!, how are you."

# tokenize the content
# it is nothing but seperating the whole line into parts
from nltk.tokenize import word_tokenize
word_tokenize(sentence)

# Now,
# Removing all the stopwords
from nltk.corpus import stopwords

swords = stopwords.words('english')

clean = [word for word in word_tokenize(sentence) if word not  in swords ]

print(clean)

from nltk.stem import PorterStemmer
ps = PorterStemmer()
clean = [ps.stem(word) for word in word_tokenize(sentence)
            if word not in swords]

print(clean)

sent = "Hello friends! How are you? We will be learning Python today"

"""# We can also make a function to clean the data"""

# here we can also make a function to clean the data
def clean_text(sentence):
  # This will tokenize the sentence
  token = word_tokenize(sentence)
  # This will remove stopwords from the sentence
  clean = [word for word in token
           if word.isdigit or word.isalpha]
  # This will remove the stem words(postfix from the word)
  clean = [ps.stem(word) for word in token
           if word not in swords]
  return clean

clean_text(sent)

"""# Preprocessing
# Coverting Charecter to Numeric value using TF-IDF Vectorize
"""

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(analyzer=clean_text)

x = df['text']
y = df['label']

x_new = tfidf.fit_transform(x)

x.shape

x_new.shape

y.value_counts()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x_new, y, random_state = 0, test_size = 0.25)

x_train.shape

x_test.shape

"""# Naive_Bayes Classifier

"""

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train.toarray(),y_train)

y_pred = nb.predict(x_test.toarray())

y_test.value_counts()

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test,y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

"""#         RandomForest Classifier"""

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0)

rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

print(classification_report(y_test, y_pred))

accuracy_score(y_test,y_pred)

"""# Logistic Regression"""

from sklearn.linear_model import LogisticRegression
log = LogisticRegression()

log.fit(x_train, y_train)

y_pred = log.predict(x_test)

print(classification_report(y_test, y_pred))

accuracy_score(y_test, y_pred)

"""# Hyper Parameter Tuning"""

from sklearn.model_selection import GridSearchCV

params = {
    'criterion':['gini','entropy'],
    'max_features':['sqrt','log2'],
    'random_state':[0,1,2,3,4],
    'class_weight':['balanced','balanced_subsample']
}

grid = GridSearchCV(rf, param_grid=params, cv=5, scoring='accuracy')

grid.fit(x_train, y_train)

rf = grid.best_estimator_

y_pred = rf.predict = (x_test)

accuracy_score(y_test,y_pred)