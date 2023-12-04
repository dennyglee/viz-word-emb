#
# Purpose: Convert GloVe vectors into word2vec format
#
# Author: Denny Lee
#
# Download GloVe vectors from https://nlp.stanford.edu/projects/glove/
# The used in this sample is the glove.6B.zip file
#

# Attribution: https://radimrehurek.com/gensim/scripts/glove2word2vec.html

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('../glove/glove.6B.50d.txt')
w2v_file = get_tmpfile('../glove/glove.6B.50d.word2vec.txt')

_ = glove2word2vec(glove_file, w2v_file)

