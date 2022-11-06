import numpy as np
import random
import json

import torch
import torch.nn as nn
from chatdataset import ChatbotDataset
from model import NeuralNet
from torch.utils.data import Dataset, DataLoader

from nltk.tokenize import word_tokenize as tokenize
from nltk.corpus import stopwords
from nltk.stem import *
from bag_of_words import bagofwords
#from model import NeuralNet
stop_words = set(stopwords.words("english"))
stop_words.update("!","-",",",".","?")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
with open('intents.json', 'r') as f:
    intents = json.load(f)

words = []
tags = []
#list of tuples
tokenwords_tag = []
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        #lowercase and alpha
        w = tokenize(pattern)
        w = [word.lower() for word in w if word.isalpha()]
        # add to our words list
        words.extend(w)
        # add to xy pair
        tokenwords_tag.append((w, tag))


words = [lemmatizer.lemmatize(stemmer.stem(w)) for w in words if w not in stop_words]
print(words)
# remove duplicates and sort
words = sorted(set(words))
tags = sorted(set(tags))

x_train = []
y_train = []

for (tp_sentence,tag) in tokenwords_tag:
    bag = bagofwords(tp_sentence, words)
    #list of tensors
    x_train.append(bag)

    label = tags.index(tag)
    #list of index (numbers)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)


batch_size = 6
hidden_size = 6
output_size = len(tags)
input_size = len(words)
learning_rate = 0.001
epochs = 1000
#print(input_size,output_size)

dataset = ChatbotDataset(x_train,y_train,len(x_train))
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size, output_size)
model.to(device)
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#print(train_loader)
for epoch in range(epochs):
    for (ws, labels) in train_loader:
        #ws: batches
        #print(ws)
        #model and data both set to same (either cpu or gpu)
        ws = ws.to(device)
        labels = labels.to(device)
        #forward
        outputs = model(ws)
        loss = criteria(outputs, labels)
        #backward and optimizer step 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch + 2) % 100 == 0:
        print(f'epoch {epoch +2}/{epochs}, loss={loss.item()}')
print(f'final loss {loss.item()}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": words,
"tags": tags
}
f = "data.pth"
torch.save(data, f)

