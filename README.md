# Chatbot_CSA

main code in chat.py of torch directory

- based on intents.json as training data, builds a neural network finding the weights for the words within intents.json.
- trains 1000 epochs, adam optimizer, learning rate 0.001, based on the x_train as vector repr of a sentence using bag of words, y_train as the index of the tag intent
- will check user input, give the sentence a vector representation with bag of words, plug it into the neural network and find the highest probability intent
