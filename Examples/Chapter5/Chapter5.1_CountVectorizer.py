#Chapter5.1_CountVectorizer.py

from sklearn.feature_extraction.text import CountVectorizer

# Dataset
list_message = ["Call me soon", "CALL to win", "Pick me up soon"]
# Create instance of vectorizer
vectorizer = CountVectorizer(decode_error="ignore")
# Perform word counts
Xfit = vectorizer.fit_transform(list_message)
# Generate feature matrix (transform so sample axis is along columns)
X = Xfit.toarray().T
print("X: {}".format(X))
# list words in vocabulary
print("Vocabulary: {}".format(vectorizer.get_feature_names()))