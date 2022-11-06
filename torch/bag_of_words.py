import nltk
from nltk.stem import *
from nltk import word_tokenize
import numpy as np
#from model import NeuralNet
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
def bagofwords(ts, words): 
    sentence_words = [lemmatizer.lemmatize(stemmer.stem((word))) for word in ts]
    #print(sentence_words)
    # initialize bag with 0 for each word
    vector = np.zeros(len(words), dtype=np.float32)
    #print(vector)
    for i, word in enumerate(ts):
        if word in words:
            vector[words.index(word)] += 1

    return vector
if __name__ == '__main__':
    print(bagofwords("hi i eat",["hi","i","eat"]))