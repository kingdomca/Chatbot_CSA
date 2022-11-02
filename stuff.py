from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ["hi",'a','b']
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(corpus)