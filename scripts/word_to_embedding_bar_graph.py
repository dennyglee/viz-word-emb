#
# word_to_embedding_bar_graph.py
# Description: Script to calculate word embeddings and plot them
# Author: Denny Lee
#

from gensim.models import KeyedVectors
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing


# Arguement parsing: specify word to calculate embeddings and plot
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("word", help="Word you want to calculate embeddings and plot")
args = parser.parse_args()

#
# Calculate word embedding
#

# Load a pre-trained Word2Vec model (GloVe wiki gigaword
#import gensim.downloader as api
#word_vectors = api.load("glove-wiki-gigaword-300")
#word_vectors = api.load("glove-wiki-gigaword-100")

# Load Glove 6B 50d model converted to word2vec format
# Use glove_to_word2vec.py to convert Glove 6B 50d model to word2vec format
w2v_file = '../glove/glove.6B.50d.word2vec.txt'
word_vectors = KeyedVectors.load_word2vec_format(w2v_file)

# Generate word_embedding for the word "..."
word_text = args.word

if word_text is None:
    print("Please provide a word to calculate embeddings and plot")
    exit()

# Scale word embedding
word_embedding = sklearn.preprocessing.minmax_scale(word_vectors[word_text])


#
# Plot word embedding by bar graph
#

# Reshape word_embedding
r2 = np.array([word_embedding])
r2 = r2.T

# Configure colormap
# https://matplotlib.org/stable/users/explain/colors/colormaps.html
my_cmap = plt.get_cmap("bwr")
rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

# Configure plot size
plt.figure(figsize=(8, 0.8))
plt.bar(np.arange(len(r2)), height=0.05, width=0.95, bottom=0, color=my_cmap(rescale(r2[:,0])))
plt.axis('off')

# Save figure
savefigPath = '../images/glove.6B.50d/' + word_text + '.png' 
plt.savefig(savefigPath, bbox_inches='tight', pad_inches=0)
