from sklearn.feature_extraction.text import CountVectorizer
corpus = [
             'This is the first document.',
             'This document is the second document.',
             'And this is the third one.',
             'Is this the first document?',

        ]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())

words = "first second third"
x_test = vectorizer.transform([words])
print(x_test)
x_test= x_test.todense()
print(x_test[0])
