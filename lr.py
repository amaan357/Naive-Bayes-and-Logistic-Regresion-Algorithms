import re
import os
import numpy as np
import sys


def get_vocab(stop=0):
    V = {}
    for c in C:
        path = './train/' + c
        for filename in os.listdir(path):
            document_text = open(path + "/" + filename, 'r', encoding="latin1")
            text_string = document_text.read().lower()
            match_pattern = re.findall(r'\b[a-z]{1,15}\b', text_string)
            for word in match_pattern:
                if stop == 1:
                    if word not in stop_words:
                        count = V.get(word, 0)
                        V[word] = count + 1
                else:
                    count = V.get(word, 0)
                    V[word] = count + 1
    return V


def get_features(v, input='train'):
    y = []
    features = []
    for c in C:
        path = './' + input + '/' + c
        for filename in os.listdir(path):
            x = {}
            b = []
            document_text = open(path + "/" + filename, 'r', encoding="latin1")
            text_string = document_text.read().lower()
            match_pattern = re.findall(r'\b[a-z]{1,15}\b', text_string)
            for word in match_pattern:
                count = x.get(word, 0)
                x[word] = count + 1
            for a in sorted(v):
                b.append(x.get(a, 0))
            features.append(b)
            if c == 'ham':
                y.append(0)
            elif c == 'spam':
                y.append(1)

    feat = np.asarray(features)
    b0 = np.ones((feat.shape[0], 1))
    X = np.hstack((b0, feat))
    Y = np.asarray(y)
    return X, Y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def trainlr(x, y, l, a, n):
    w = np.zeros(x.shape[1])
    for m in range(n):
        z = np.dot(x, w)
        p = sigmoid(z)
        dz = y - p
        dw = np.dot(x.T, dz) - l*w
        w += a*dw
    return w

def testlr(x, y, w):
    final = np.dot(x, w)
    pred = np.round(sigmoid(final))
    print(format(float((pred == y).sum())*100 / len(pred), ".3f"))

def model(v, l=1, a=0.05, n=50):
    x, y = get_features(v)
    w = trainlr(x, y, l, a, n)

    a, b = get_features(v, 'test')
    testlr(a, b, w)



stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours"]
C = ['ham', 'spam']

V = get_vocab()
V1 = get_vocab(1)


intarg = sys.argv
if len(intarg) <= 4:
    if len(intarg) <= 3:
        if len(intarg) <= 2:
            if len(intarg) <= 1:
                print("Accuracy with stop words : ", end="")
                model(V)
                print("Accuracy without stop words : ", end="")
                model(V1)
            else:
                l = int(intarg[1])
                print("Accuracy with stop words : ", end="")
                model(V, l)
                print("Accuracy without stop words : ", end="")
                model(V1, l)
        else:
            l = int(intarg[1])
            a = float(intarg[2])
            print("Accuracy with stop words : ", end="")
            model(V, l, a)
            print("Accuracy without stop words : ", end="")
            model(V1, l, a)
    else:
        l = int(intarg[1])
        a = float(intarg[2])
        n = int(intarg[3])
        print("Accuracy with stop words : ", end="")
        model(V, l, a, n)
        print("Accuracy without stop words : ", end="")
        model(V1, l, a, n)
else:
    print("Too many arguments")
