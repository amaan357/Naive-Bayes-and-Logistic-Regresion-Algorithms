import re
import os
import math
import pandas as pd

def trainnb(stop=0):
    V = {}
    N = 0
    text = {}
    prior = {}
    for c in C:
        path = './train/' + c
        x = {}
        for filename in os.listdir(path):
            document_text = open(path + "/" + filename, 'r', encoding="latin1")
            text_string = document_text.read().lower()
            match_pattern = re.findall(r'\b[a-z]{1,15}\b', text_string)
            N += 1
            for word in match_pattern:
                if stop == 1:
                    if word not in stop_words:
                        count = V.get(word, 0)
                        V[word] = count + 1
                        ct = x.get(word, 0)
                        x[word] = ct + 1
                else:
                    count = V.get(word, 0)
                    V[word] = count + 1
                    ct = x.get(word, 0)
                    x[word] = ct + 1
        text[c] = x
    condprob = pd.DataFrame(index=V.keys(), columns=C)

    for c in C:
        path = './train/' + c
        nc = len(os.listdir(path))
        prior[c] = nc / N
        sumc = sum(text[c].values())
        ext = len(text[c])
        for t in V.keys():
            condprob.at[t, c] = (text[c].get(t, 0) + 1) / (sumc + ext)

    return V, prior, condprob, N


def applynb(v, p, cond, d, e):
    W = []
    score = {}
    pathd = './test/' + e
    dtext = open(pathd + "/" + d, 'r', encoding="latin1")
    tstring = dtext.read().lower()
    mpattern = re.findall(r'\b[a-z]{1,15}\b', tstring)
    for w in mpattern:
        for q in v:
            if w == q:
                W.append(w)
                break

    for c in C:
        score[c] = math.log(p[c])
        for t in W:
            score[c] += math.log(cond.at[t, c])

    max_key = max(score, key=lambda k: score[k])
    return max_key

def testnb(V, prior, condprob, N):
    s = 0
    for c in C:
        path = './test/' + c
        for filename in os.listdir(path):
            e = applynb(V, prior, condprob, filename, c)
            if e == c:
                s += 1
    return format(s*100/N, ".3f")


stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours"]
C = ['ham', 'spam']
a, b, c, d = trainnb()
print("accuracy with stop words : " + str(testnb(a, b, c, d)))
e, f, g, h = trainnb(1)
print("accuracy without stop words : " + str(testnb(e, f, g, h)))
