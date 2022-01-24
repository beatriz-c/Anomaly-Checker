#perguntar luis: porque nao usou dicionario - atribui id a cada palavra
#ver se tenho o mesmo nmr de fraudes e não fraudes
#métricas de qualidade de algoritmos de machine learning
#A maior accuracy conseguida foi de 78% com aprendizagem supervisionada
#aumentar nrows até rebentar
#fazer função para passar os parâmetros do pd.cut
#ver crossvalidation e loss function
#preciso de dicionario?
#ver unknonws
#nao dividi em tokens - um só registo grande
#converter de w2v para numpy array?

#-----------------------------------------pre-processamento---------------------------------------------------#

#carregar dados
from cProfile import run
import pandas as pd
from pandas.core.frame import DataFrame
predados = pd.read_csv('traindata.csv', nrows = 1000)

#eliminar colunas desnecessarias
predados.drop(predados.columns[0], axis = 1, inplace = True) 
predados.drop(['timestamp', 'ID', 'rowid', 'clientid', 'is_fraud', '#clientid_30D', 'clientid_fingerprint_30D', 
'clientid_ipaddress_30D', 'clientid_iban_orig_30D', 'clientid_iban_dest_30D', '#clientid_ipaddress_30D', 
'#clientid_fingerprint_30D', '#clientid_iban_orig_30D', '#clientid_iban_dest_30D', 'is_fraud_cons', 
'cons_freq_clientid'], axis = 1, inplace = True)

#conversao de montantes para classes
import numpy as np 

#geracao de labels automaticos
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


#colunas antigas a serem substituidas pelas novas com classes

#amount (float) n14
#min, max = 0.01; 2000000
bins_amount = np.linspace(predados['amount'].min(), predados['amount'].max(), num = 100)
labels_amount = []
a = LabelCategorizer(base_word = 'azul')

for i in range(99):
    labels_amount.append(a.get_next_word())

predados['amount'] = pd.cut(predados['amount'], bins = bins_amount, labels = labels_amount)


#accountbalance (float) n15
#min,max = -99943772.97; 359308117.81
bins_balance = np.linspace(predados['accountbalance'].min(), predados['accountbalance'].max(), num = 250)
labels_balance = []
v = LabelCategorizer(base_word = 'verde')

for i in range(249):
    labels_balance.append(v.get_next_word())

predados['accountbalance'] = pd.cut(predados['accountbalance'], bins = bins_balance, labels = labels_balance)


#mean_amount_clientid_30D (float) n41
#min,max = 0.0099999999990905; 2000000.0
bins_amount_clientid = np.linspace(predados['mean_amount_clientid_30D'].min(), predados['mean_amount_clientid_30D'].max(), num = 100)
labels_amount_clientid = []
b = LabelCategorizer(base_word = 'branco')

for i in range(99):
    labels_amount_clientid.append(b.get_next_word())

predados['mean_amount_clientid_30D'] = pd.cut(predados['mean_amount_clientid_30D'], bins = bins_amount_clientid, labels = labels_amount_clientid)


#mean_amount_iban_dest_30D (float) n42
#min,max = 0.0099999999999857; 2000000.0
bins_amount_iban_dest = np.linspace(predados['mean_amount_iban_dest_30D'].min(), predados['mean_amount_iban_dest_30D'].max(), num = 100)
labels_amount_iban_dest = []
p = LabelCategorizer(base_word = 'preto')

for i in range(99):
    labels_amount_iban_dest.append(p.get_next_word())

predados['mean_amount_iban_dest_30D'] = pd.cut(predados['mean_amount_iban_dest_30D'], bins = bins_amount_iban_dest, labels = labels_amount_iban_dest)


#amount_ratio_avg_clientid_30D (float) n48
#min,max = -1.0; 12138118.28
bins_amount_ratio_avg = np.linspace(predados['amount_ratio_avg_clientid_30D'].min(), predados['amount_ratio_avg_clientid_30D'].max(), num = 150)
labels_amount_ratio_avg = []
s = LabelCategorizer(base_word = 'rosa')

for i in range(149):
    labels_amount_ratio_avg.append(s.get_next_word())

predados['amount_ratio_avg_clientid_30D'] = pd.cut(predados['amount_ratio_avg_clientid_30D'], bins = bins_amount_ratio_avg, labels = labels_amount_ratio_avg)


#amount_ratio_avg_iban_dest_30D (float) n49
#min,max = -1.0; 58857542.99999999
bins_amount_ratio_iban_dest = np.linspace(predados['amount_ratio_avg_iban_dest_30D'].min(), predados['amount_ratio_avg_iban_dest_30D'].max(), num = 200)
labels_amount_ratio_iban_dest = []
r = LabelCategorizer(base_word = 'roxo')

for i in range(199):
    labels_amount_ratio_iban_dest.append(r.get_next_word())

predados['amount_ratio_avg_iban_dest_30D'] = pd.cut(predados['amount_ratio_avg_iban_dest_30D'], bins = bins_amount_ratio_iban_dest, labels = labels_amount_ratio_iban_dest)


#hour (int) n83
#min,max = 0; 23
bins_hour = np.linspace(predados['hour'].min(), predados['hour'].max(), num = 4)
labels_hour = []
m = LabelCategorizer(base_word = 'marfim')

for i in range(3):
    labels_hour.append(m.get_next_word())

predados['hour'] = pd.cut(predados['hour'], bins = bins_hour, labels = labels_hour)


#week (int) n84
#min,max = 1; 52
bins_week = np.linspace(predados['week'].min(), predados['week'].max(), num = 13)
labels_week = []
o = LabelCategorizer(base_word = 'ouro')

for i in range(12):
    labels_week.append(o.get_next_word())

predados['week'] = pd.cut(predados['week'], bins = bins_week, labels = labels_week)


#amount_ratio_avg_30D_per_clientid (float) n85
#min,max = -32597575.973856207; 1517178021.5142858
bins_amount_ratio_per_clientid = np.linspace(predados['amount_ratio_avg_30D_per_clientid'].min(), predados['amount_ratio_avg_30D_per_clientid'].max(), num = 350)
labels_amount_ratio_per_clientid = []
t = LabelCategorizer(base_word = 'prata')

for i in range(349):
    labels_amount_ratio_per_clientid.append(t.get_next_word())

predados['amount_ratio_avg_30D_per_clientid'] = pd.cut(predados['amount_ratio_avg_30D_per_clientid'], bins = bins_amount_ratio_per_clientid, labels = labels_amount_ratio_per_clientid)


#amount_ratio_max_30D_per_clientid (float) n86 
#min,max = -233411.5; 2567787246.0
bins_amount_max_per_clientid = np.linspace(predados['amount_ratio_max_30D_per_clientid'].min(), predados['amount_ratio_max_30D_per_clientid'].max(), num = 200)
labels_amount_max_per_clientid = []
c = LabelCategorizer(base_word = 'cinza')

for i in range(199):
    labels_amount_max_per_clientid.append(c.get_next_word())

predados['amount_ratio_max_30D_per_clientid'] = pd.cut(predados['amount_ratio_max_30D_per_clientid'], bins = bins_amount_max_per_clientid, labels = labels_amount_max_per_clientid)


#amount_over_account_balance (float) n87
#min, max = -1726484.0; 150000000.0
bins_amount_over_balance = np.linspace(predados['amount_over_account_balance'].min(), predados['amount_over_account_balance'].max(), num = 200)
labels_amount_over_balance = []
z = LabelCategorizer(base_word = 'bronze')

for i in range(199):
    labels_amount.append(z.get_next_word())

predados['amount_over_account_balance'] = pd.cut(predados['amount_over_account_balance'], bins = bins_amount_over_balance, labels = labels_amount_max_per_clientid)



#---------------------------------------------------treino------------------------------------------------------#

#conversao de dataframe para arrays de arrays numpy
sentences = predados.reset_index().to_numpy() #list of lists

#inicia word2vec e faz treino
import gensim.models.word2vec as w2v
import multiprocessing

def treino (sentences, cycles ,dim, architecture, context):
    model = w2v.Word2Vec (
        sg = architecture, #1 - skip-gram , 0 - cbow
        workers = multiprocessing.cpu_count(), #nmr cores usados pelo pc 
        vector_size = dim, #dimensao do espaco embedding
        window = context, #palavras antes e depois de uma dada palavra
        hs = 0, #0 - negative sampling optimization , 1 - hierarchical softmax
        sample = 0, #limiar para o qual palavras que ocorrem com alta frequência são down-sampled - sem subsampling
)
    
    #criacao vocabulario
    model.build_vocab(sentences) 

    #train model
    model.train(sentences, epochs = cycles, total_examples = model.corpus_count, start_alpha = 0.01, end_alpha = 0.0005)

    return model


#criacao modelo
model = treino(sentences, 10, 300, 1, 5)

#salvar modelo
model.save('/tmp/mymodel')