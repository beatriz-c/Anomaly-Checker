import gensim.models.word2vec as w2v
import numpy as np

new_model = w2v.load('/tmp/mymodel')

M1 = new_model.wv.syn0