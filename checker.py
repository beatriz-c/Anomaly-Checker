

#-----------------------------------------pre-processing---------------------------------------------------#

#data loading
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn import metrics
predata = pd.read_csv('traindata.csv', nrows = 1000) #aumentar nrows até rebentar

#delete unnecessary columns
predata.drop(predata.columns[0], axis = 1, inplace = True) 
predata.drop(['timestamp', 'ID', 'rowid', 'clientid', 'is_fraud', '#clientid_30D', 'clientid_fingerprint_30D', 
'clientid_ipaddress_30D', 'clientid_iban_orig_30D', 'clientid_iban_dest_30D', '#clientid_ipaddress_30D', 
'#clientid_fingerprint_30D', '#clientid_iban_orig_30D', '#clientid_iban_dest_30D', 'is_fraud_cons', 
'cons_freq_clientid'], axis = 1, inplace = True)

#amounts -> classes
import numpy as np 

#automatic labels
import string

class LabelCategorizer():
    def __init__(self, base_word = 'gato'):
        self._alphabet_index = 0
        self.base_word = base_word
        self.current_word = self.base_word
        self.shift = 0

    def __str__(self):
        return 'Class: Label Categorizer\nBase word: ' + self.base_word + '\nCurrent Word: ' + self.current_word

    def get_next_word(self):
        if self.shift > 0:
            self.current_word = self.current_word[-1] + self.current_word[:-1]
            self.shift -= 1
        else:
            self.current_word = self.current_word + string.ascii_lowercase[self._alphabet_index]
            self._alphabet_index = (self._alphabet_index + 1) % len(string.ascii_lowercase)
            self.shift = len(self.current_word) - 1

        return self.current_word


#replacement of the old columns with the new ones with classes
def cutter (col, number, word):
    bins_a = np.linspace(predata[col].min(), predata[col].max(), num = number)
    labels_a = []
    a = LabelCategorizer(base_word = word)

    for i in range(number - 1):
        labels_a.append(a.get_next_word())

    predata[col] = pd.cut(predata[col], bins = bins_a, labels = labels_a)


columns = ['amount', 'accountbalance', 'mean_amount_clientid_30D', 'mean_amount_iban_dest_30D', 'amount_ratio_avg_clientid_30D',
'amount_ratio_avg_iban_dest_30D', 'hour', 'week', 'amount_ratio_avg_30D_per_clientid', 'amount_ratio_max_30D_per_clientid',
'amount_over_account_balance']
number_bins = [100, 250, 100, 100, 150, 200, 4, 13, 350, 200, 200]
base_words = ['blue', 'green', 'white', 'black', 'pink', 'purple', 'yellow', 'red', 'grey', 'brown', 'silver']


for i in range (len(columns)):
    cutter(columns[i], number_bins[i], base_words[i])


#amount (float) n14 -> min, max = 0.01; 2000000
#accountbalance (float) n15 -> min,max = -99943772.97; 359308117.81
#mean_amount_clientid_30D (float) n41 -> min,max = 0.0099999999990905; 2000000.0
#mean_amount_iban_dest_30D (float) n42 -> min,max = 0.0099999999999857; 2000000.0
#amount_ratio_avg_clientid_30D (float) n48 -> min,max = -1.0; 12138118.28
#amount_ratio_avg_iban_dest_30D (float) n49 -> min,max = -1.0; 58857542.99999999
#hour (int) n83 -> min,max = 0; 23
#week (int) n84 -> min,max = 1; 52
#amount_ratio_avg_30D_per_clientid (float) n85 -> min,max = -32597575.973856207; 1517178021.5142858
#amount_ratio_max_30D_per_clientid (float) n86 -> min,max = -233411.5; 2567787246.0
#amount_over_account_balance (float) n87 -> min, max = -1726484.0; 150000000.0



#---------------------------------------------------training------------------------------------------------------#

#dataframe to numpy array of arrays
listoflists = predata.reset_index().to_numpy() #list of lists

#converting numbers to words
sentences = [str(int) for int in listoflists]

#inicialization and training word2vec
import gensim.models.word2vec as w2v
import multiprocessing

def training (sentences, cycles ,dim, architecture, context):
    model = w2v.Word2Vec (
        sg = architecture, #1 - skip-gram , 0 - cbow
        workers = multiprocessing.cpu_count(), #uses all the cores 
        vector_size = dim, #dimension of the embedding space
        window = context, #words befores and after center word
        hs = 0, #0 - negative sampling optimization , 1 - hierarchical softmax
        sample = 0, #whithout subsampling  
)
    
    #vocabulary creation
    model.build_vocab(sentences) 

    #model training
    model.train(sentences, epochs = cycles, total_examples = model.corpus_count, 
    start_alpha = 0.01, end_alpha = 0.0005, compute_loss = True)

    return model

#model creation
model = training(sentences, 10, 300, 1, 5)

#metrics
#loss after each epoch
#accuracy


#saving the model
model.save('Trained.w2v')

#print(listoflists)