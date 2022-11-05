#!/usr/bin/env python
# coding: utf-8



import speech_recognition as sr


r = sr.Recognizer()


# In[2]:


with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)
    data = r.record(source, duration=5)
    print("Sesinizi Tanımlıyor…")
    print(2)
    text = r.recognize_google(data,language='ru')
    print(1)
    print(text)


# In[3]:


import io
import random
import string
import warnings
import numpy as np
import warnings
from gtts import gTTS
import os
warnings.filterwarnings('ignore')
import speech_recognition as sr 
import nltk
from nltk.stem import WordNetLemmatizer
#for downloading package files can be commented after First run
nltk.download('popular', quiet=True)
nltk.download('nps_chat',quiet=True)
nltk.download('punkt') 
nltk.download('wordnet')


# In[4]:


posts = nltk.corpus.nps_chat.xml_posts()[:10000]
# To Recognise input type as QUES. 
def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features
featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[5]:


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# In[6]:


#Reading in the input_corpus
#with open('intro_join','r', encoding='utf8', errors ='ignore') as fin:
#     raw = fin.read().lower()
raw = text
#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words
# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens): 
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# In[7]:


#colour palet
def prRed(skk): print("\033[91m {}\033[00m" .format(skk)) 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk)) 
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk)) 
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk)) 
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk)) 
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk)) 
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk)) 
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))


# In[ ]:





# In[8]:


#Recording voice input using microphone 
file = "file.mp3"
flag=True
fst="My name is Kondo. I will do what you want. If you want to exit, say Bye"
tts = gTTS(fst, lang="en",tld="com")
tts.save(file)
os.system("mpg123 " + file )
r = sr.Recognizer()
prYellow(fst)


# In[4]:


# Yandex music API
get_ipython().system('pip install yandex-music --upgrade')


# In[9]:


from yandex_music import Client

client = Client("y0_AgAAAAAqvlDlAAiJcgAAAADSicCfF-W6WmPOSTaafBFgACMr9Eq5wU4").init()


# In[10]:


client.tracks(['10994777:1193829', '40133452:5206873', '48966383:6693286', '51385674:7163467'])[0]
#client.tracks_download_info(38318)[0].download(filename="1.mpg")
client.search("Birds")["best"]["result"]["id"]#.download(filename="1.mpg")


# In[11]:


from playsound import playsound
playsound("1.mpg")

#!pip install playsound
def play_music_by_id(id):
    client.tracks_download_info(38318)[0].download(filename="1.mpg") # add id
    playsound("1.mpg")
def play_music_by_name(name):
    client.tracks_download_info(client.search(name)["best"]["result"]["id"])[0].download(filename="1.mpg") # add id
    import time
    time.sleep(3)
    playsound("1.mpg")

play_music_by_name("Closed")


# In[8]:


client.search("closed")["best"]["result"]["id"]


# In[12]:


nothing  = lambda x:None
commands = {"Play music":lambda id:play_music_by_name(id), "Move Head":nothing , "Go forward":nothing, "Go backward":nothing, "Turn right":nothing, "Turn left":nothing}
key_words_command = {"music":commands["Play music"]}
key_words_command["music"](1)


# 

# In[13]:


# "music Tokyo Drift".find("music")
mp3_nameold='111'
mp3_name = "txt.mp3"
def say_and_wait(txt):
            tts=gTTS(text=txt, lang='ru')
            # Получаем от гугла озвученное предложение в виде mp3 файла           
            tts.save(mp3_name)
            # Проигрываем полученный mp3 файл
            playsound(mp3_name)
            # Если предыдущий mp3 файл существует удаляем его
            # чтобы не захламлять папку с приложением кучей mp3 файлов
            if(os.path.exists(mp3_nameold) and (mp3_nameold!="1.mp3")):
                os.remove(mp3_nameold)
            mp3_nameold=mp3_name



# In[15]:


while(flag==True):
    with sr.Microphone() as source:
        audio= r.listen(source, phrase_time_limit=3)
    try:
        user_response = format(r.recognize_google(audio))
        print("\033[91m {}\033[00m" .format("YOU SAID : "+user_response))
    except sr.UnknownValueError:
        prYellow("Oops! Didn't catch that")
    
    #user_response = input()
    #user_response=user_response.lower()
    if ("play" in user_response):
        print("Start play music")
        key_words_command["music"](user_response.find("music") + len("music"))
    clas=classifier.classify(dialogue_act_features(user_response))
    print(clas)
    if(clas!='Bye'):
        if(clas=='Emotion'):
            flag=False
            text = "Kondo: You are welcome.."
            prYellow(text)
            say_and_wait(text)
        
                
        else:
            if(greeting(user_response)!=None):

                text = greeting(user_response) 
                
                say_and_wait(text)
                
                print("\033[93m {}\033[00m" .format("Kondo: "+ text))
            else:
                print("\033[93m {}\033[00m" .format("Kondo: ",end=""))
                # res=(response(user_response))
                say_and_wait("Goodbue")
                sent_tokens.remove(user_response)
                tts = gTTS("Goodbye", 'en')
                tts.save(file)
                os.system("mpg123 " + file)
    else:
        flag=False
        text = "Kondo: Bye! take care.."
        prYellow(text)
        say_and_wait(text)
        # play the speech

