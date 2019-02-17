import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
#import sklearn.linear_model as lm
from sklearn.svm import SVC, LinearSVC #, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.classify import ClassifierI
import nltk.classify.scikitlearn as sk
from statistics import mode
from nltk.tokenize import word_tokenize



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            #v = c.prob_classify(features)
            votes.append(v)
        #return mode(votes)
        return most_element(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(most_element(votes))
        conf = choice_votes / len(votes)
        return conf


def most_element(liste):
    numeral=[[liste.count(nb), nb] for nb in liste]
    numeral.sort(key=lambda x:x[0], reverse=True)
    return(numeral[0][1])

word_features5k_f = open("~/word_features.pickle", "rb")
word_features = pickle.load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features



open_file = open(r"~\processed data\originalnaivebayes.pickle", "rb")
classifier = pickle.load(open_file)
open_file.close()


open_file = open(r"~\processed data\MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(open_file)
open_file.close()


open_file = open(r"~\processed data\LogisticRegression_classifier.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()


open_file = open(r"~\processed data\LinearSVC_classifier.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


open_file = open(r"~\processed data\svcclassifier.pickle", "rb")
SVC_classifier = pickle.load(open_file)
open_file.close()


open_file = open(r"~\processed data\Randomforest classifier.pickle", "rb")
Random_forest_classifier = pickle.load(open_file)
open_file.close()


open_file = open(r"~\processed data\DecisionTreeClassifier.pickle", "rb")
Decision_tree_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(
                                  classifier,
                                  MNB_classifier,
                                  LogisticRegression_classifier,
                                  LinearSVC_classifier,
                                  Random_forest_classifier,
                                  Decision_tree_classifier
                                  )


#LogisticRegression_classifier,SVC_classifier,Random_forest_classifier,Decision_tree_classifier

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)