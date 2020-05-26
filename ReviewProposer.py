import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#From the text data we have, read it to form the corpus of the chatbox
with open('MegaData.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

alive = True
tonken_1 = nltk.sent_tokenize(raw)

# Preprocessing using Lemmatization
lemmatizer = WordNetLemmatizer()

def call():

    print("Hello I am Jeff Bezos and here to directly help you write the best product descriptions and reviews ")
    print("based on successful ones. We can have a chat, or if you want to evaluate a rating or a description")
    print("just say \"I want a review\"")


    while (alive == True):
        user_response = input()
        user_response = user_response.lower()
        #print("User_reponse = ", user_response)
        if review_request(user_response):

            print("Great to hear! Please, enter what you would like to be reviewed: ")
            user_response = input()
            print("Jeff Bezos: Ok let me think about it...")
            return user_response


            # Here call to the model and print the results

        elif (user_response != 'bye' or user_response != 'ciao' or user_response != 'seeya'):
            if (user_response == 'thanks' or user_response == 'thank you'):
                print("Jeff Bezos: ", thank())
            else:

                if (welcome(user_response) != None):
                    print("Jeff Bezos: " + welcome(user_response))
                else:

                    print("Jeff Bezos: ", end="")
                    print(answer(user_response))
                    tonken_1.remove(user_response)
        else:
            flag = False
            print("Jeff Bezos: Thank you for choosing us! See you later")


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(corpus):
    return lemmatizetokens(nltk.word_tokenize(corpus.lower().translate(remove_punct_dict)))


def lemmatizetokens(tokens):
    return [lemmatizer.lemmatize(token) for token in tokens]


WELCOME = ("hello", "hi", "greetings", "sup", "what's up", "hey", "bonjour", "salut", "slt")
WELCOME_ANSWERS = ["Hello there!", "Hi there", "Greetings!", "Welcome here", "Welcome my friend"]
REVIEW_CHECK_REQUEST = "i want a review"
THANK_YOU = ["thanks", "you're welcome", "my plesure", "no problems"]

def answer(user_response):
    back = ''
    tonken_1.append(user_response)
    #vectorize the strings to feed the model
    tfidfit = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english').fit_transform(tonken_1)
    idx = cosine_similarity(tfidfit[-1], tfidfit).argsort()[0][-2]

    # Compute the cosine similarities
    #cosine_score = cosine_similarity(tfidfit[-1], tfidfit).flatten()
    # Selects the best
    #cosine_best = cosine_score.sort()
    #final_out = cosine_best[-2]

    # req_tfidf = cosine_similarity(tfidfit[-1], tfidfit)

    cosine_score = cosine_similarity(tfidfit[-1], tfidfit).flatten()
    cosine_score.sort()
    final_out = cosine_score[-2]

    #req_tfidf = cosine_similarity(tfidfit[-1], tfidfit).flatten().sort()[-2]

    if (final_out != 0):
        back = back + tonken_1[idx]
        return back
    else:
        #if nothing matching is found
        back = back + "I can not find a satisfiable answer for you"
        return back

def thank():
    return random.choice(THANK_YOU)


def welcome(sentence):
    for word in sentence.split():
        if word.lower() in WELCOME:
            return random.choice(WELCOME_ANSWERS)


"""Returns true if the user wants to evaluate a request from the model"""


def review_request(sentence):
    if sentence in REVIEW_CHECK_REQUEST:
        return True
    else:
        return False



