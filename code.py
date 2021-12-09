#COM DASK

#-----------------------------------------pre-processamento---------------------------------------------------#

#carregar dados
from dask import dataframe as dd
from pandas.core.reshape.tile import cut
dtype = {'trusted_indicator': object}
predados = dd.read_csv('dados.csv', dtype = dtype, blocksize = "5GB", engine = 'c')

#eliminar colunas desnecessarias
del predados[predados.columns[0]] #coluna 1 sem nome

predados = predados.drop(['ID', 'rowid', 'canal', 'operativa', 'clientid', 'fingerprint', 'is_fraud', 'is_mobile', 'is_tablet',
'is_pc', 'is_touch', 'is_bot', 'browser_family', 'os_family', '#clientid_30D', 'clientid_fingerprint_30D', 'clientid_ipaddress_30D', 
'clientid_iban_orig_30D', 'clientid_iban_dest_30D', '#clientid_ipaddress_30D', '#clientid_fingerprint_30D', '#clientid_iban_orig_30D', 
'#clientid_iban_dest_30D', 'is_fraud_cons', 'cons_freq_clientid'], axis = 1) 


#conversão de nmrs para classes
import pandas as pd

def cutter (predados, col, bins, labels):
    return pd.cut(x = predados[col], bins = bins, labels = labels)

#timestamp (06-Aug-19 00:08:50)

#client (nmr inteiro)

#entity (nmr inteiro)

#reference (nmr inteiro)

#trusted_indicator (nmr inteiro)

#iban_orig (nmr inteiro)

#iban_dest (nmr inteiro)

#amount (float)
#print(max(predados['amount'])) 2000000.0
#print(min(predados['amount'])) 0.0
bins_amount = [0, 400000, 800000, 1200000, 1600000, 2000000]
labels_amount = ['Low', 'Medium', 'High', 'Rich', 'Filthy Rich']
predados.map_partitions(cutter, 'amount', bins = bins_amount, labels = labels_amount) 

#accountbalance (float)
#print(max(predados['accountbalance'])) 359308117.81
#print(min(predados['accountbalance'])) -99943772.97
bins_balance = [-99943773, 0, 35930900, 71861800, 107792700, 143723600, 179654500, 215585400, 251516300, 287447200, 323378100, 359309000]
labels_balance = ['Negative', 'Average', 'Average High', 'Richer', 'Super Rich', 'More Rich', 'Millionaire', 'Billionaire', 
'Trillionaire', 'Quadrillionaire', 'Donald Duck']
predados.map_partitions(cutter, 'accountbalance', bins = bins_balance, labels = labels_balance)

#description_originator (nmr inteiro)

#description_beneficiary (nmr inteiro)

#dummy_var (nmr inteiro)

##fingerprint_30D (nmr inteiro)

##ipadress_30D (nmr inteiro)

##iban_orig_30D (nmr inteiro)

##iban_dest_30D (nmr inteiro)

#fingerprint_time_diff (nmr decimal)

#iban_dest_time_diff (nmr inteiro)

#mean_amount_iban_dest_30D (nmr decimal)

#amount_ratio_avg_iban_dest_30D (nmr decimal)

#canal_MBE (nmr inteiro)

#canal_MBP (nmr inteiro)

#canal_NBE (nmr inteiro)

#canal_NBP (nmr inteiro)

#canal_OBE (nmr inteiro)

#operativa_PAGSRV (nmr inteiro)

#operativa TRFINT (nmr inteiro)

#operativa TRFIPS (nmr inteiro)

#operativa TRFITC (nmr inteiro)

#operativa TRFSEP (nmr inteiro)

#operativa_others (nmr inteiro)

#entity__3 (nmr inteiro)

#entity__37 (nmr inteiro)

#entity__469 (nmr inteiro)

#entity__5623 (nmr inteiro)

#entity__others (nmr inteiro)

#hour (int) 
bins_hour = [0, 4, 8, 12, 16, 20, 24]
labels_hour = ['Madrugada', 'Manha', 'Almoco', 'Tarde', 'Anoitecer', 'Noite']
predados.map_partitions(cutter, 'hour', bins = bins_hour, labels = labels_hour)

#week (int)
#print(max(predados['week'])) 52
#print(min(predados['week'])) 1
bins_week = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52]
labels_week = ['Janeiro', 'Fevereiro', 'Marco', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro',
'Dezembro', 'Janeiro1', 'Fevereiro1', 'Marco1', 'Abril1', 'Maio1']
predados.map_partitions(cutter, 'amount', bins = bins_week, labels = labels_week) 


#criar dicionário onde é associado um nmr (id) a cada string (várias instâncias de cada classe)

