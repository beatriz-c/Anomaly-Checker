""" import numpy as np
import gensim.models.word2vec as w2v
from sklearn.manifold import TSNE

# Open the trained model
model = w2v.Word2Vec.load('Trained.w2v')

# Vocabulary length
vocab_len = len(model.wv.index_to_key)
print('Vocabulary length is ', vocab_len)

# Define Matix
word_vectors_matrix = np.ndarray(shape=(vocab_len, 10), dtype = 'float64')
word_list = []
i = 0

# Fill the Matix
for word in model.wv.index_to_key:
    word_vectors_matrix[i] = model.wv[word]                   
    word_list.append(word)
    i += 1
    if i == vocab_len:
        break

# Compress the word vectors into 2D space
tsne = TSNE(n_components = 2, random_state = 0, metric="cosine")
word_vectors_matrix_2d = tsne.fit_transform(word_vectors_matrix)

# Save Matrix
print(word_vectors_matrix.shape, word_vectors_matrix_2d.shape)
#print(word_vectors_matrix)
print()
print()
#print(word_vectors_matrix_2d) """

#---------------------------------------------------------------------------------------------------------------#

import numpy as np
import gensim.models.word2vec as w2v
from sklearn.manifold import TSNE

# Open the trained model
model = w2v.Word2Vec.load('Trained.w2v')

count = 2000
word_vectors_matrix = np.ndarray(shape=(count, 10), dtype='float64')
word_list = []
i = 0
for word in model.wv.index_to_key:
    word_vectors_matrix[i] = model.wv[word]
    word_list.append(word)
    i += 1
    if i == count:
        break

print("word_vectors_matrix shape is ", word_vectors_matrix.shape)

# Compress the word vectors into 2D space
tsne = TSNE(n_components = 2, random_state = 0, metric="cosine")
word_vectors_matrix_2d = tsne.fit_transform(word_vectors_matrix)
print("word_vectors_matrix_2d shape is ", word_vectors_matrix_2d.shape)