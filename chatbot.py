from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess

user_message = "I am a fan of anime"
response_a = "I hate anime"

response_b = "Hi are you an avid game fan who plays games"
response_c = "no i went to tokyo"
response_d = "im bored"


documents = [response_a,response_b,response_c,user_message]

# create tfidf vectorizer, preprocceses 
vectorizer = TfidfVectorizer(preprocessor=preprocess)

# fit and transform vectorizer on processed docs
tfidf_vectors = vectorizer.fit_transform(documents)


# compute cosine similarity betweeen the user message tf-idf vector and the different response tf-idf vectors
cosine_similarities = cosine_similarity(tfidf_vectors[-1], tfidf_vectors)
print(cosine_similarities)
# get the index of the most similar response to the user message
similar_response_index = cosine_similarities.argsort()[0][-2]

best_response = documents[similar_response_index]
print(best_response)