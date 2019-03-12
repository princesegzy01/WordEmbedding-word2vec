import sys
from gensim.models import Word2Vec
from gensim.corpora import WikiCorpus
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt

# stop_words = stopwords()

# print(stop_words)

dataset = open("dataset.txt", "r")
dataset = str(dataset.read())

# remove citation
dataset = re.sub('[0-9]', ' ', dataset)
dataset = re.sub('[^0-9a-zA-Z]', ' ', dataset)
dataset = re.sub('[\n]', ' ', dataset)

sentences = []

dataset = dataset.split(".")

for line in dataset:
    line = word_tokenize(line)

    # remove stop words
    words = [word.lower() for word in line if not word in set(
        stopwords.words("english"))]

    sentences.append(words)

# build vocabulary
model = Word2Vec(sentences, min_count=1, window=10)

# train model
model.train(sentences, total_examples=len(sentences), epochs=10000)

# sys.exit(0)

vocabulary = list(model.wv.vocab)

# Access vector for one word
# print(model['impeachment'])

# Get all the vocabulary
# print(list(model.wv.vocab))


# Get token details, count & index
# print(model.wv.vocab['impeachment'])


# get most similar tokens
print(model.most_similar(['business'], topn=4))

# get positive and negative filtered data
# print(model.most_similar(
#    positive=['company', 'finance'], negative=['president'], topn=2))

x = model[model.wv.vocab]

PCA = PCA(n_components=2)
result = PCA.fit_transform(x)


plt.scatter(result[:, 0], result[:, 1])

for i, word in enumerate(vocabulary):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
