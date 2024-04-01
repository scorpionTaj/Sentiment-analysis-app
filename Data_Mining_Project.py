import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Load necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Load the data
df = pd.read_csv(r"amazon_cells_labelled.txt",sep="\t",header = None)
df.columns = ["Comment","label"]
# df.shape

# df.info()


# df.describe()


# df.label.value_counts()


# df.label.value_counts(normalize=True)

# Function to remove punctuation from a comment
def remove_punctiation(comment):
  comment_nopunct = "".join([char for char in comment if char not in string.punctuation])
  return comment_nopunct

# Apply remove_punctuation function to each comment and convert to lowercase
df["Comment_nopunct"] = df["Comment"].apply(lambda x : remove_punctiation(x.lower()))

# Tokenize each comment
def tokenize(comment):
    tokens = word_tokenize(comment)
    return tokens

df['Comment_tokenized'] = df['Comment_nopunct'].apply(lambda x : tokenize(x))

# Define and remove English stopwords
stopwords_english = nltk.corpus.stopwords.words('english')

def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopwords_english]
    return text

df['Comment_nostopwords'] = df['Comment_tokenized'].apply(lambda x : remove_stopwords(x))

# Lemmatize the words
wn = nltk.WordNetLemmatizer()
def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

df['Comment_lemmatized'] = df['Comment_nostopwords'].apply(lambda x : lemmatizing(x))

df1=df[["label","Comment","Comment_lemmatized"]]

df2=df1[["label","Comment"]]
df2['Comment_sentences'] = df1['Comment_lemmatized'].apply(lambda x: ' '.join(x))

vectorizer = CountVectorizer()
features_cv = vectorizer.fit_transform(df2['Comment_sentences'])

#create a DataFrame with the counts
features_cv_df = pd.DataFrame(features_cv.toarray(), columns=vectorizer.get_feature_names_out())

# Feature extraction
df2['Comment_length'] = df1['Comment'].apply(lambda x : len(x))  # Length of the comment
def count_punct(text):
    if len(text) == 0:
        return 0
    else:
        count = sum([1 for char in text if char in string.punctuation])
        return count / len(text)

df2['puncts'] = df2['Comment'].apply(count_punct)
def calculate_word_count(text):
    # Split the text into words and count them
    words = text.split()
    return len(words)

def count_unique_words(text):
    words = text.split()
    return len(set(words))

# Add Word_count column
df2['Word_count'] = df2['Comment'].apply(calculate_word_count)

scaler = StandardScaler()
num_vars = ["Comment_length","Word_count","puncts"]

df2[num_vars] = scaler.fit_transform(df2[num_vars])

target=df2['label']
final_df=df2.loc[:,['Comment_length','Word_count','puncts']]
final_df = pd.concat([final_df, pd.DataFrame(features_cv_df)], axis=1)
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(final_df, target, test_size=0.20)

# Creating a RandomForestClassifier object with n_jobs=-1 (using all processors)
rf_model = RandomForestClassifier(n_jobs=-1)

# Fitting the model with training data
rf_model.fit(X_train, y_train)

# Scoring the model on the test data
rf_model.score(X_test, y_test)

# Creating a RandomForestClassifier object with n_jobs=-1 and n_estimators=200
rf_model = RandomForestClassifier(n_jobs=-1, n_estimators=200)

# Fitting the model with training data
rf_model.fit(X_train, y_train)

# Scoring the model on the test data
rf_model.score(X_test, y_test)

# Constructing the confusion matrix.
from sklearn.metrics import confusion_matrix
y_pred = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)


TP = cm[1, 1]  # true positives
TN = cm[0, 0]  # true negatives
FP = cm[0, 1]  # false positives
FN = cm[1, 0]  # false negatives

# print('True Positives : ', TP)
# print('True Negatives : ', TN)
# print('False Positives : ', FP)
# print('False Negatives : ', FN)

# print('Number of test Comments : ', TP + TN + FP + FN)
# print('Number of actual Negative Comments : ', TP + FN)
# print('Number of actual Positive Comments : ', TN + FP)
# print('Number of predicted Comments as Negative Ones: ', TP + FP)
# print('Number of predicted Comments as Positive Ones : ', FN + TN)


Accuracy = (TP + TN) / (TP + TN + FP + FN)
# print('Accuracy : ', round(Accuracy, 3))

Precision = TP / (TP + FP)
# print('Precision : ', round(Precision, 3))

Recall = TP / (TP + FN)
# print('Recall : ', round(Recall, 3))

F1Score = (2* Precision*Recall)/(Precision+Recall)
# print ("F1 Score : ",round(F1Score,3))