from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer('bert-base-nli-mean-tokens')

sentences = [
    "I like anime",
    "I hate anime",
    "I'm a fan of anime",
    "I like eating food"
]

sentence_embeddings = model.encode(sentences)

li = cosine_similarity(
    [sentence_embeddings[0]],
    sentence_embeddings[1:]
)
print(li)