#-----------------------------------------pre-processamento---------------------------------------------------#

from nltk.tokenize import word_tokenize
from nltk import sent_tokenize

#carregar ficheiro 
file = open('dados.csv')
text = file.read()
file.close()

#converter numeros para classes - le o csv para uma classe com cada linha do csv sendo uma instancia da classe
import csv

#timestamp - é criado um id para cada timestamp que atua como a representação de string na lista
class timestamp:
    def __init__(self, row, header, id):
       self.__dict__ = dict(zip(header, row)) 
       self.the_id = id
    def __repr__(self):
       return self.id

data = list(csv.reader(open('dados.csv')))
instances = [timestamp(a, data[0], "timestamp_{}".format(i+1)) for i, a in enumerate(data[1:])]

#fazer para o resto das colunas-headers


#converter classes para palavra
palavras = str(instances)


#juntar num registo(frase) as classes convertidas com o que já era palavras 
registo = []
registo.append(palavras)
#registo.append(o que já era palavras como definir)


#converter registo para vetores - recebe uma string que contém uma frase e converte-a num vetor incluindo as palavras na frase
def tagging(sent):
    sent = word_tokenize(sent)
    return sent

#divide o vetor com a frase em palavras - dividir em tokens (palavras)
sentences = sent_tokenize(text)

for sentence in sentences:
    print("Comecou")
    #substitui hifens por espaco
    sentence = sentence.replace('_', ' ')
    sentence = sentence.replace('-', ' ')

    words = word_tokenize(sentence)

print("Terminado!")


#-----------------------------------------treino word2vec---------------------------------------------------#

import gensim.models.word2vec as w2v
import os
import multiprocessing

#carregar ficheiro pre-processado (etapa anterior)
file = open('')
book = file.read()
file.close()
print("Book loaded")

#quero 70% dos dados pre-processados para treino
from sklearn.model_selection import train_test_split
train = book #sao os dados todos?
test = #remover so coluna fraudes
X_train, X_test, y_train, y_test = train_test_split(train, test, test_size = 0.2, shuffle = True, random_state = 8) #confirmar 0.2

#importar 70% dos dados pre-processados num vetor e dividi-lo em tokens (cada frase é convertida em palavras)
frases_limpas = sent_tokenize(book)

#Alimentar o vetor tokenizado ao word2vec 
dadosmodelo = []
#determinar tamanho do vetor 
for frase_limpa in frases_limpas:
    if len(frase_limpa) > 0:
        dadosmodelo.append(frase_limpa)
print(frases_limpas[6])
token_count = sum([len(dados) for dados in dadosmodelo])
print("This corpus contains {} tokens.".format(token_count))

#word2vec
data2vec = w2v.Word2Vec(
    sg = 1, #skip-gram
    seed = 1,
    workers = multiprocessing.cpu_count(), #nmr cores usados pelo pc 
    size = 300, #dimensao do espaco
    min_count = 0, #elimina palavras que só aparecem x vezes 
    window = 8, #palavras antes e depois de uma dada palavra
    sample = 1e-4 #limiar para o qual palavras que ocorrem com alta frequência são down-sampled
)

#criacao vocabulario
data2vec.build_vocab(dadosmodelo)

#salvar modelo treinado
data2vec.train(dadosmodelo, total_examples = data2vec.corpus_count, epochs = data2vec.iter)

if not os.path.exists("Treinado"):
    os.makedirs("Treinado")

data2vec.save(os.path.join("Treinado", "Treino.w2v"))


#-----------------------------------------reducao dimensionalidade---------------------------------------------------#

from __future__ import absolute_import, division, print_function
import numpy as np
from sklearn.manifold import TSNE

#abrir o modelo treinado
model = w2v.Word2Vec.load("Treino.w2v")
print("Modelo carregado")

#tamanho do vocabulário
vocab_len = len(model.wv.vocab)
print("O vocabulário tem", vocab_len)

#definir matriz
vetores_palavras_matriz = np.ndarray(shape = (vocab_len, 300), dtype = 'float64')
word_list = []
i = 0

#preencher a matriz
for word in model.wv.vocab:
    vetores_palavras_matriz[i] = model[word]
    word_list.append(word)
    i += 1
    if i == vocab_len:
        break


#comprimir os vetores de palavras para espaco 2d
tsne = TSNE(n_components = 2, random_state = 0, metric="cosine")
vetores_palavras_matriz_2d = tsne.fit_transform(vetores_palavras_matriz)

#salvar matriz
np.save("Mtx_name", vetores_palavras_matriz)
np.save("Mtx_2d_name", vetores_palavras_matriz_2d)