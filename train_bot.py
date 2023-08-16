#Biblioteca de pré-processamento de dados de texto
import nltk

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

import json
import pickle
import numpy as np

words=[]
classes = []
word_tags_list = []
ignore_words = ['?', '!',',','.', "'s", "'m"]
train_data_file = open('./intents.json').read()
intents = json.loads(train_data_file)

#função para anexar palavras-tronco
def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)  
    return stem_words

for intent in intents['intents']:
    
        # Adicione todas as palavras dos padrões à lista
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                      
            word_tags_list.append((pattern_word, intent['tag']))
        # Adicione todas as tags à lista classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            stem_words = get_stem_words(words, ignore_words)

print(stem_words)
print(word_tags_list[0]) 
print(classes)   

#Crie o corpus de palavras para o chatbot
def create_bot_corpus(stem_words, classes):

    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    pickle.dump(stem_words, open('words.pkl','wb'))
    pickle.dump(classes, open('classes.pkl','wb'))

    return stem_words, classes

stem_words, classes = create_bot_corpus(stem_words,classes)  

print(stem_words)
print(classes)

# Crie um saco de palavras 
trainingData = []
numberOfTags = len(classes)
labels = [0]*numberOfTags
for wordTags in word_tags_list:
    bagOfWords = []
    pattern_word = wordTags[0]
    for word in pattern_word:
        index = pattern_word.index(word)
        word = stemmer.stem(word.lower())
        pattern_word[index] = word
    for stemWord in stem_words:
        if stemWord in pattern_word:
            bagOfWords.append(1)
        else: 
            bagOfWords.append(0)
    print(bagOfWords)
    labelCode = list(labels)
    tag = wordTags[1]
    tagIndex = classes.index[tag]
    labelCode[tagIndex] = 1
    trainingData.append([bagOfWords,labelCode])
print(trainingData[0])
# Crie os dados de treinamento
