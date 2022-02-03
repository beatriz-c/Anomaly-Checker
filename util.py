#callback to print loss after each epoch
from gensim.models.callbacks import CallbackAny2Vec

class callback(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        elif self.epoch % 100 == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
        
        self.epoch += 1
        self.loss_previous_step = loss

    
    #model training
    model.train(sentences, epochs = cycles, total_examples = model.corpus_count, 
    start_alpha = 0.01, end_alpha = 0.0005, compute_loss=True, callbacks=[callback()])

    return model


#--------------------------------------------------------------
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('model-{epoch:03d}-{val_acc:03f}.h5', verbose=1, monitor='val_acc',save_best_only=True, mode='auto')