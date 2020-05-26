# -*- coding: UTF-8 -*-
import argparse
import random

import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neural_network import MLPClassifier
import numpy as np

import ReviewProposer
from ReviewProposer import call
GOOD_FEEDBACK = ["Wow! Looks like this is a pretty good review",
                 "That is a great review you got here :)",
                 "Hehe nice, just need to put it online then",
                 "Another great review",
                 "Seems like you have some sucess"]
BAD_FEEDBACK = ["Wow! That is a bad one",
                "Seems like someone is not a fan",
                "You better avoid that",
                "Should avoid such comments on your products!"]

def test_review(review):
    review_vect = vectorizer.transform(review)
    print(review_vect.toarray())
    if clr.predict(review_vect) >= 1:
        print("Jeff Bezos: ", random.choice(GOOD_FEEDBACK))
        print("That is 4 or 5 stars!")
    else:
        print("Jeff Bezos: ", random.choice(BAD_FEEDBACK))
        print("That is like 1 or 2 stars...")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        default='output.csv',
                        help='reviews_Clothing_Shoes_and_Jewelry_5.json, XXXX is the file')
    parser.add_argument('--dataset3',
                        default='output.csv',
                        help='Software.json, XXXX is the file')
    args = parser.parse_args()


    print('start processing data')

    option = 2

    if option == 1: #Claqué
        #Accuracy is 100%
        param1 = "average_review_rating"
        param2 = "customer_reviews"
        perso_dataset = "amazon_co-ecommerce_sample.csv"
        input_data = pd.read_csv(perso_dataset)
    elif option ==2:
        param1 = "overall"
        param2 = "reviewText"
        input_data = pd.read_csv(args.dataset)
    elif option == 3:
        param1 = "overall"
        param2 = "reviewText"
        input_data = pd.read_csv(args.dataset3)




    #print(input_data.describe())
    #print(input_data.head(10))

    input_data[param1] = input_data[param1].astype(object)  # fix datatype error
    input_data[param2] = input_data[param2].astype(object)  # fix datatype error

    input_data[param2][:100].to_csv('data_as_text.txt', header=None, index=None, sep=' ', mode='a')

    """print("Describe the dataset")
    print(input_data.describe())
    print(input_data.head(10))"""

    working_data = {param2: input_data[param2][:100000], param1: input_data[param1][:100000]}
    working_data = pd.DataFrame(data=working_data)
    working_data = working_data.dropna()  # ignore if any row contained NaN
    working_data = working_data[working_data[param1] != '3']

    #Here we relabel our data such that we work only on good and bad reviews and discard medium ones
    if option==3 or option==2:
        working_data['label'] = working_data[param1].apply(lambda rating: +1 if str(rating).split(" ", 1)[0] > '4' else 0)
    elif option==1:
        working_data['label'] = working_data[param1].apply(lambda rating: +1 if str(rating).split(" ", 1)[0] > '4' else 0)

    # Splitting data into training set and testing set
    X = pd.DataFrame(working_data, columns=[param2])
    y = pd.DataFrame(working_data, columns=['label'])

    print("split the dataset...")
    X_training, test_X, Y_training, test_y = train_test_split(X, y)


    print("vectorize the corpus...")
    train_vector = vectorizer.fit_transform(X_training[param2])
    test_vector = vectorizer.transform(test_X[param2])
    print('processing ... ok')
    print('start training model')


    #In case you want to try something else
    #clr = LogisticRegression()
    #clr = LinearRegression()
    #clr = Ridge(alpha=.5)
    #clr = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (5, 2), random_state = 1)



    clr.fit(train_vector, Y_training.values.ravel())
    scores = clr.score(test_vector, test_y)
    print('training ...ok')
    print('accuracy: {}%'.format(scores * 100))

    #In case you want a quick check by yourself
    """ review1_1 = "That is so bad, the worst ever, not recommend"
    review1 = [review1_1]
    test_review(review1)
   
    review2 = ["This is the greatest, I love it so much. Highly recommend it to everyone. It is the best ever"]
    test_review(review2)"""


def display_header():
    print()
    print()
    print()
    print("___________________________________________________________________________________________")
    print("*******************************************************************************************")
    print("___________________________________________________________________________________________")
    print()
    print("                                Welcome to the NLP Project!")
    print()
    print("You will soon be greeted by Jeff Bezos who will guide you through the wonders of Amazon...")
    print("         Do not hesitate to ask him his opinion about the description of your product")
    print("         or certain reviews...       ")
    print()
    print("___________________________________________________________________________________________")
    print("*******************************************************************************************")
    print("___________________________________________________________________________________________")
    print()

if __name__ == '__main__':
    NGram = 2
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b', analyzer='word', ngram_range=(NGram, NGram))
    clr = LogisticRegression()
    main()
    display_header()
    chatbox = ReviewProposer#.call()
    flag = True

    while(flag):
        user_response = [chatbox.call()]
        print("call done")
        test_review(user_response)
        print("test review done")
