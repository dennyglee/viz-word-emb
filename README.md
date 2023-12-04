# Visualize your word embeddings (viz-word-emb)

This repository contain various scripts, references, and datasets to help visualize your word embeddings.  These scripts are inspired by the [Generative AI exists because of the transformer](https://ig.ft.com/generative-ai/).

* glove
  * `glove.6B.50d.word2vec.txt`: The original Glove 6B 50d dataset can be downloaded from [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/).  This is a copy of the converted `glove.6B.50d.txt` to `word2vec` format to make it easier to work with `gensim`.
* images
  * Contains word embedding visualizations in bar graph format
* scripts
  * `glove_to_word2vec.py`: Converts Glove format to word2vec format; script is based on [glove2word2vec page](https://radimrehurek.com/gensim/scripts/glove2word2vec.html.
  * `word_to_embedding_bar_graph.py`: Finds word embedding of a given word from Glove 6B 50d, scales the values, and plots it as a  1 row 50D (cell) bar graph
    *  Usage: `python3 word_to_embedding_bar_graph.py cycling`  


