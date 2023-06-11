import pandas as pd
import csv
import numpy as np
import time
from datetime import datetime
import names

nome_file = 'seed.rls'

# Lista per i valori della quarta colonna
realization = []

# Apri il file in modalità lettura
with open(nome_file, 'r') as file:
    # Leggi il contenuto del file riga per riga
    for riga in file:
        # Dividi la riga in colonne utilizzando uno spazio come delimitatore
        colonne = riga.strip().split()

        # Verifica se ci sono abbastanza colonne nella riga
        if len(colonne) >= 4:
            # Aggiungi il valore della quarta colonna alla lista dei valori
            realization.append(float(colonne[2]))

# Stampa la lista dei valori della quarta colonna
print('realizations',realization)

# Nome del file della tabella
nome_file = 'seed_and_target.dtt'

# Numero di colonne della tabella
num_colonne = 8

# Apri il file in modalità lettura
with open(nome_file, 'r') as file:
    # Leggi la prima riga
    prima_riga = file.readline().strip().split()

    # Estrai gli ultimi 3 numeri dalla prima riga
    numeri_percentili = [int(num) for num in prima_riga[-3:]]

    # Definisci la lista di percentili
    percentiles = [f'{num}%ile' for num in numeri_percentili]

    print(percentiles)

    # Salta le prime 4 righe
    for _ in range(3):
        next(file)

# Apri il file in modalità lettura
with open(nome_file, 'r') as file:
    # Crea un lettore CSV
    lettore_csv = csv.reader(file)

    # Leggi le prime 4 righe come header
    header = []
    for _ in range(3):
        riga = next(lettore_csv)
        header.append(riga)

    # Leggi il resto dei dati
    dati = []
    for riga in lettore_csv:
        # print(riga)
        dati.append(riga[0].split())

header=["idx_expert","expert_name", "idx_q", "type_and_idx_q", "scale"] + percentiles

# print(header)    
# Crea un DataFrame utilizzando l'header e i dati
df = pd.DataFrame(data=dati,columns=header)
        
# Leggi il resto dei dati e organizzali in un DataFrame
# df = pd.read_csv(file, skiprows=3, header=None, usecols=range(1, num_colonne+1),
#                     )

    
# Stampa il DataFrame
print(df)

idx_experts = np.unique(np.array(df['idx_expert'],dtype=int))
print(idx_experts)
n_experts = idx_experts.shape[0]

rand_first_name = []
rand_last_name = []

for i in range(n_experts):
    first_name = names.get_first_name()
    rand_first_name.append(first_name)
    last_name = names.get_last_name()
    rand_last_name.append(last_name)
    print(first_name+','+last_name)

header_quest = ['IDX','LABEL','SHORT Q','LONG Q_ENG','UNITS','SCALE','MINVAL','MAXVAL','REALIZATION','QUEST_TYPE','IDXMIN','IDXMAX','SUM50','PARENT','IMAGE']
data_quest = []

for count,idx in enumerate(idx_experts):

    print('SQ',idx)
    columns = []
    data = []
    rslt_df = df[df['idx_expert'] == str(idx)]

    name = list(rslt_df['expert_name'])[0]

    columns.append('First Name')
    data.append(rand_first_name[count])

    columns.append('Last Name')
    data.append(rand_last_name[count])

    columns.append('Email address')
    data.append(rand_last_name[count]+'@mail.com')

    columns.append('Group(s)')
    data.append('0')

    n_SQ = 0

    for index,row in rslt_df.iterrows():

        row_sq = []

        if 'SQ' in row['type_and_idx_q']:
            n_SQ +=1
            seed_idx = int(str(row['type_and_idx_q']).replace('SQ',''))
            columns.append(row['type_and_idx_q']+' - 5%ile (0;inf) []')
            data.append(row['5%ile'])
            columns.append(row['type_and_idx_q']+' - 50%ile (0;inf) []')
            data.append(row['50%ile'])
            columns.append(row['type_and_idx_q']+' - 95%ile (0;inf) []')
            data.append(row['95%ile'])

            if idx == 1:

                row_sq.append(int(row['type_and_idx_q'].replace('SQ','')))
                row_sq.append(str(row['type_and_idx_q']))
                row_sq.append(row['type_and_idx_q'])
                row_sq.append('Long text')
                row_sq.append('[]')
                row_sq.append(str(row['scale']))
                row_sq.append(0)
                row_sq.append('inf')
                row_sq.append(str(realization[seed_idx-1]))
                row_sq.append('seed')
                row_sq.append(0)
                row_sq.append(0)
                row_sq.append(0)
                row_sq.append(-1)
                row_sq.append('')
                
                data_quest.append(row_sq)
            
            

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    new_filename = './DTT_TO_CSV/seed/questionnaire_' + dt_string + '_Output.csv'
        
        
    # Write the merged data to a new CSV file
    with open(new_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(columns)  # Write the updated header row
        writer.writerow(data)  # Write the merged data row

    time.sleep(2)
        
            
for count,idx in enumerate(idx_experts):

    print('TQ',idx)
    columns = []
    data = []
    rslt_df = df[df['idx_expert'] == str(idx)]

    name = list(rslt_df['expert_name'])[0]

    columns.append('First Name')
    data.append(rand_first_name[count])

    columns.append('Last Name')
    data.append(rand_last_name[count])

    columns.append('Email address')
    data.append(rand_last_name[count]+'@mail.com')

    columns.append('Group(s)')
    data.append('0')
    
    n_TQ = 0

    for index,row in rslt_df.iterrows():

        row_tq = []

        if 'TQ' in row['type_and_idx_q']:
            n_TQ +=1
            columns.append(row['type_and_idx_q']+' - 5%ile (0;inf) []')
            data.append(row['5%ile'])
            columns.append(row['type_and_idx_q']+' - 50%ile (0;inf) []')
            data.append(row['50%ile'])
            columns.append(row['type_and_idx_q']+' - 95%ile (0;inf) []')
            data.append(row['95%ile'])

            if idx == 1:

                row_tq.append(int(row['type_and_idx_q'].replace('TQ','')))
                row_tq.append(str(row['type_and_idx_q']))
                row_tq.append(row['type_and_idx_q'])
                row_tq.append('Long text')
                row_tq.append('[]')
                row_tq.append(str(row['scale']))
                row_tq.append(0)
                row_tq.append('inf')
                row_tq.append(str(realization[seed_idx-1]))
                row_tq.append('target')
                row_tq.append(0)
                row_tq.append(0)
                row_tq.append(0)
                row_tq.append(-1)
                row_tq.append('')
                
                data_quest.append(row_tq)

            
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    new_filename = './DTT_TO_CSV/target/questionnaire_' + dt_string + '_Output.csv'
        
        
    # Write the merged data to a new CSV file
    with open(new_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(columns)  # Write the updated header row
        writer.writerow(data)  # Write the merged data row

    time.sleep(2)
        
new_filename = './DTT_TO_CSV/questionnaire.csv'
        
df = pd.DataFrame(data_quest, columns = header_quest)
df.to_csv(new_filename,index=False)             
            
