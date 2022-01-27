#converter de w2v para numpy array
#falta: métricas de qualidade de algoritmos de machine learning - crossvalidation; 
#probs condicionadas - produto das matrizes - função softmax; 
#matrizes densidade representação gráfica e computacional
#ver codigo teses
#remover /n e letras grandes após str

import gensim.models.word2vec as w2v
import numpy as np

new_model = w2v.load('/tmp/mymodel')

M1 = new_model.wv.syn0

#loss
from gensim.models.callbacks import CallbackAny2Vec

class callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
    
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()

        if self.epoch == 0:
            print('Loss after each epoch {}: {}'.format(self.epoch, loss))

        elif self.epoch %100 == 0:
            print('Loss after each epoch {}: {}'.format(self.epoch, loss - self.loss_previous_step))

        self.epoch += 1
        self.loss_previous_step = loss

        #add callbacks = [callback()] ao training