# -*- coding: utf-8 -*-
"""
@author: Soham Shah
"""

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import wmi
import speech_recognition as sr
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import os
from playsound import playsound
from gtts import gTTS


def bulkfetch():
    language='en'
    arr = [
     ["Hello!", "Good to see you again!", "Hi there, how can I help?"], 
     ["Nothing much. I'm good, you say?", "Happier than a seagull with a french fry!", "Hi there, how can I help?"],
     ["Sad to see you go :(", "Talk to you later", "Goodbye!"], ["I am 19 years old!", "19 years young!", "I'm Immortal"],
     ["You can call me ChatX.", "I'm ChatX!", "I'm ChatX, created by Master Soham"], 
     ["I'm available all day everyday; until you shut down ur pc ;)!"], 
     ["I’m humbled and grateful.", "Thank you!", "You’re a blessing to me.", "I’m touched beyond words.", "My heart just keeps thanking you and thanking you.", "All I can say is wow! (Except, of course, I’m grateful.)"],
     ["I can do anything for you ;)"],
     ["I love you", "Okay here's what i did yesterday. I ate a clock yesterday, it was very time-consuming.", "Have you played the updated kids’ game? I Spy With My Little Eye . . . Phone.", "Q. What is the biggest lie in the entire universe?\nA. “I have read and agree to the Terms & Conditions.", "Person 1: Do you know how to use Outlook?\nPerson 2: As a matter of fact, I Excel at it.\nPerson 1: Was that a Microsoft Office pun?\nPerson 2: Word",
      "Q: Why did the computer show up at work late?\nA: It had a hard drive."],
      ["Done!"]
      ]
    low_confidence = ["Oops, Didn't get it", "Umm...seems you are drunk, ask proper questions!", 
                              "Whhoooops, can't understand what that means. Maybe try another question?"]
     
    k=1
    """for i in range(0, len(arr)):
        for j in arr[i]:
            myob=gTTS(text=j,lang=language,slow=False)
            y = "Voice" + str(k) + ".mp3"
            myob.save(y)
            k = k+1
        """
    for i in low_confidence:
        myob = gTTS(text=i, lang = language, slow = False)
        y = "Voice" + str(k) + ".mp3"
        myob.save(y)
        k = k+1
bulkfetch()           
def play(key):
   
   main_path = 'your-path-here'
   if key == 1:
       y = '\\age'
   elif key == 2:
       y = '\\do'
   elif key == 3:
       y = '\\done'
   elif key == 4:
       y = '\\greeting'
   elif key == 5:
       y = '\\howareyou'
   elif key == 6:
       y = '\\joke'
   elif key == 7:
       y = '\\name'
   elif key == 8:
       y = '\\praising'
   elif key == 9:
       y = '\\time'
   elif key == 10:
       y = '\\Bye'
   elif key == 11:
       y = '\\low_confidence'
       
   z = random.choice(os.listdir(main_path + y))   
   playsound(main_path + y + "\\" + z)

       
def audiototext():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak now...")
        audio = r.listen(source)
        print("...")
    
    try:
        print("You said: " + r.recognize_google(audio) + "\n");
        return r.recognize_google(audio)
    except:
        pass;


with open("C:\\Users\\Soham Shah\\Desktop\\chatbot\\Bot\\intents.json") as file:
    data = json.load(file)

try:
    with open("C:\\Users\\Soham Shah\\Desktop\\chatbot\\Bot\\data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("C:\\Users\\Soham Shah\\Desktop\\chatbot\\Bot\\data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])

net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=270, batch_size=8, show_metric=True)
model.save("model2.tflearn")
    
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def talk():
    print("Start chatting with the bot (type quit to stop)!")
    while True:
        inp = audiototext()
        
        try:
                
            if inp.lower() == "quit":
                break
    
            results = model.predict([bag_of_words(inp, words)])
            results_index = numpy.argmax(results)
            if results[0][results_index] > 0.4:
                
                tag = labels[results_index]
                
                if tag == 'greeting':
                    play(4)
                elif tag == 'questions':
                    play(5)
                elif tag == 'goodbye':
                    play(10)
                elif tag == 'age':
                    play(1)
                elif tag == 'name':
                    play(7)
                elif tag == 'hours':
                    play(9)
                elif tag == 'praising':
                    play(8)
                elif tag == 'do':
                    play(2)
                elif tag == 'bored':
                    play(6)
                elif tag == 'VOLUME UP':
                    play(3)
                elif tag == 'VOLUME DOWN':
                    play(3)
                elif tag == 'VOLUME OFF':
                    play(3)
            
            else:
                play(11)
        except:
            print("Didn't listen properly. Say it again!")

def chat():
    print("Start chatting with the ChatX (say quit to stop)!")
    while True:
        
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        if results[0][results_index] > 0.4:
            tag = labels[results_index]
    
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
    
            print("\nChatX: " + random.choice(responses))
        else:
            low_confidence = ["Oops, Didn't get it", "Umm...seems you are drunk, ask proper questions!", 
                              "Whhoooops, can't understand what that means. Maybe try another question?"]
            print(random.choice(low_confidence))
            
    

def chatx():
    dec = input("Hi there! To interract--- Type 'CHAT' or 'TALK' as per your convinience\n")
    if dec.lower() == 'chat':
        chat()
    elif dec.lower() == 'talk':
        talk()
    else:
        print("Oops, I didnt get it. Try again")
        chatx()
  
chatx()
