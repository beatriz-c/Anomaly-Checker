#COM PANDAS - MENOS LINHAS

#-----------------------------------------pre-processamento---------------------------------------------------#

#carregar dados
import pandas as pd
predados = pd.read_csv('dados.csv', nrows = 1000000)

#eliminar colunas desnecessarias
del predados[predados.columns[0]] #coluna 1 sem nome

predados.drop(['ID', 'rowid', 'clientid', 'is_fraud', '#clientid_30D', 'clientid_fingerprint_30D', 
'clientid_ipaddress_30D', 'clientid_iban_orig_30D', 'clientid_iban_dest_30D', '#clientid_ipaddress_30D', '#clientid_fingerprint_30D', 
'#clientid_iban_orig_30D', '#clientid_iban_dest_30D', 'is_fraud_cons', 'cons_freq_clientid'], axis = 1, inplace = True)

print(predados.iloc[90000:100000,9:16])
