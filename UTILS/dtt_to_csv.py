import pandas as pd
import csv
import numpy as np
import time
from datetime import datetime
import os
import os.path


folder = "DTT_TO_CSV"

if os.path.isdir(folder):
    pass
else:
    os.mkdir(folder)
    os.mkdir(os.path.join(os.getcwd(), folder, "seed"))
    os.mkdir(os.path.join(os.getcwd(), folder, "target"))

# -------

nome_file = 'Vesuvio_2015_ok.rls'

# Lista per i valori della quarta colonna
realization = []

# Apri il file in modalità lettura
with open(nome_file, 'r', encoding='latin-1') as file:
    # Leggi il contenuto del file riga per riga
    for riga in file:
        # Dividi la riga in colonne utilizzando uno spazio come delimitatore
        colonne = riga.strip().split()

        # Verifica se ci sono abbastanza colonne nella riga
        if len(colonne) >= 4:
            # Aggiungi il valore della quarta colonna alla lista dei valori
            realization.append(float(colonne[2]))

# Stampa la lista dei valori della quarta colonna
print('realizations', realization)

# Nome del file della tabella
nome_file = 'Vesuvio_2015_ok.dtt'

# Numero di colonne della tabella
num_colonne = 8

# Apri il file in modalità lettura
with open(nome_file, 'r', encoding='latin-1') as file:

    # Leggi la prima riga
    prima_riga = file.readline().strip().split()

    # Estrai gli ultimi 3 numeri dalla prima riga
    numeri_percentili = [int(num) for num in prima_riga[-3:]]

    # Definisci la lista di percentili
    percentiles = [f'{num}%ile' for num in numeri_percentili]

    print('percentiles', percentiles)

    # Salta la prima riga
    for _ in range(1):
        next(file)

# Apri il file in modalità lettura
with open(nome_file, 'r', encoding='latin-1') as file:
    # Crea un lettore CSV
    lettore_csv = csv.reader(file)

    # Leggi la prima riga come header, poi leggi fino all'ultima domanda del
    # primo esperto per ottenere il testo delle domande e i nomi - mod AT
    header = []
    names_f = []
    qst_txt_f = []
    dati = []

    for _ in range(1):
        riga = next(lettore_csv)
        header.append(riga)
    for _ in range(1):
        riga = next(lettore_csv)
        dati.append(riga[0].split()[:8])
        names_f.append(riga[0].split()[-2:])
        qst_txt_f.append(riga[0].split()[8:-2])
        txt_cn1 = ' '.join(riga[0].split()[8:-2])
        txt_cn1bis = riga[0].split()[8:-2]
        len_cn1bis = len(txt_cn1bis)

    for riga in lettore_csv:
        txt_cnn = ' '.join(riga[0].split()[8:])
        txt_cnnbis = riga[0].split()[8:]
        if len(txt_cnnbis) > 0 and txt_cnnbis[0] == txt_cn1bis[
                0] and txt_cnnbis[1] == txt_cn1bis[1]:
            n_int = riga[0].split()[8:]
            names_f.append(n_int[len_cn1bis:])
            dati.append(riga[0].split()[:8])
        elif len(riga[0].split()) > 8:
            dati.append(riga[0].split()[:8])
            qst_txt_f.append(riga[0].split()[8:])
        else:
            dati.append(riga[0].split()[:8])

names = []
qst_txt = []
for j in names_f:
    n = ' '.join(j)
    names.append(n)
for j in qst_txt_f:
    n = ' '.join(j)
    qst_txt.append(n)
# print(qst_txt)

header = ["idx_expert", "expert_name", "idx_q", "type_and_idx_q", "scale"
          ] + percentiles

print(header)
# Crea un DataFrame utilizzando l'header e i dati
df = pd.DataFrame(data=dati, columns=header)
df = df.replace('-9.99500E+0002', '')
df = df.replace('-9.99600E+0002', '')

# Stampa il DataFrame
print(df)

idx_experts = np.unique(np.array(df['idx_expert'], dtype=int))
n_experts = idx_experts.shape[0]

header_quest = [
    'IDX', 'LABEL', 'SHORT Q', 'LONG Q_ENG', 'UNITS', 'SCALE', 'MINVAL',
    'MAXVAL', 'REALIZATION', 'QUEST_TYPE', 'IDXMIN', 'IDXMAX', 'SUM50',
    'PARENT', 'IMAGE'
]
data_quest = []

for count, idx in enumerate(idx_experts):

    print('Seed Questions for Exp', idx)
    columns = []
    data = []
    rslt_df = df[df['idx_expert'] == str(idx)]

    name = list(rslt_df['expert_name'])[0]

    columns.append('First Name')
    data.append(names[count].rsplit(' ', 1)[0])

    columns.append('Last Name')
    data.append(names[count].rsplit(' ', 1)[1])

    columns.append('Email address')
    data.append(names[count].rsplit(' ', 1)[0] +
                names[count].rsplit(' ', 1)[1] + '@mail.com')

    columns.append('Group(s)')
    data.append('0')
    print(names[count].rsplit(' ', 1)[1])

    n_SQ = 0
    k = 0

    for index, row in rslt_df.iterrows():

        row_sq = []

        if realization[k] != -999.5 and realization[k] != -999.6:
            n_SQ += 1
            k += 1
            # seed_idx = re.findall(r'\d+', row['type_and_idx_q'])[0]
            seed_idx = k
            columns.append(row['type_and_idx_q'] + ' - 5%ile (0;inf) []')
            data.append(row['5%ile'])
            columns.append(row['type_and_idx_q'] + ' - 50%ile (0;inf) []')
            data.append(row['50%ile'])
            columns.append(row['type_and_idx_q'] + ' - 95%ile (0;inf) []')
            data.append(row['95%ile'])

            if idx == 1:
                row_sq.append(seed_idx)
                row_sq.append(row['type_and_idx_q'])
                row_sq.append(row['type_and_idx_q'])
                row_sq.append(qst_txt[index])
                row_sq.append('[]')
                row_sq.append(str(row['scale']).lower())
                row_sq.append(0)
                row_sq.append('inf')
                row_sq.append(str(realization[int(seed_idx) - 1]))
                row_sq.append('seed')
                row_sq.append(0)
                row_sq.append(0)
                row_sq.append(0)
                row_sq.append(-1)
                row_sq.append('')

                data_quest.append(row_sq)
        else:
            k += 1

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

    # Write the merged data to a new CSV file
    if all(data) is True:
        new_filename = './DTT_TO_CSV/seed/questionnaire_' + \
                       dt_string + '_Output.csv'
        with open(new_filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(columns)  # Write the updated header row
            writer.writerow(data)  # Write the merged data row
    else:
        print(
            "Warning: Exp %s has not given answers to one or more seed, \
             saving his/her answers in a separate folder"
            % names[count].rsplit(' ', 1)[1])
        if os.path.isdir('./DTT_TO_CSV/seed_missing'):
            pass
        else:
            os.mkdir('./DTT_TO_CSV/seed_missing')
        new_filename = './DTT_TO_CSV/seed_missing/questionnaire_' + \
                       dt_string + '_Output.csv'
        with open(new_filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(columns)  # Write the updated header row
            writer.writerow(data)  # Write the merged data row

    time.sleep(2)

for count, idx in enumerate(idx_experts):

    print('Target Questions for Exp', idx)
    columns = []
    data = []
    rslt_df = df[df['idx_expert'] == str(idx)]

    name = list(rslt_df['expert_name'])[0]

    columns.append('First Name')
    data.append(names[count].rsplit(' ', 1)[0])

    columns.append('Last Name')
    data.append(names[count].rsplit(' ', 1)[1])

    columns.append('Email address')
    data.append(names[count].rsplit(' ', 1)[0] +
                names[count].rsplit(' ', 1)[1] + '@mail.com')

    columns.append('Group(s)')
    data.append('0')
    print(names[count].rsplit(' ', 1)[1])

    n_TQ = 0
    k = 0

    for index, row in rslt_df.iterrows():

        row_tq = []

        if realization[k] == -999.5 or realization[k] == -999.6:
            n_TQ += 1
            k += 1
            # target_idx = re.findall(r'^\D*(\d+)', row['type_and_idx_q'])[0]
            target_idx = k
            columns.append(row['type_and_idx_q'] + ' - 5%ile (0;inf) []')
            data.append(row['5%ile'])
            columns.append(row['type_and_idx_q'] + ' - 50%ile (0;inf) []')
            data.append(row['50%ile'])
            columns.append(row['type_and_idx_q'] + ' - 95%ile (0;inf) []')
            data.append(row['95%ile'])

            if idx == 1:
                row_tq.append(n_TQ)
                row_tq.append(row['type_and_idx_q'])
                row_tq.append(row['type_and_idx_q'])
                row_tq.append(qst_txt[index])
                row_tq.append('[]')
                row_tq.append(str(row['scale']).lower())
                row_tq.append(0)
                row_tq.append('inf')
                row_tq.append(' ')
                row_tq.append('target')
                row_tq.append(0)
                row_tq.append(0)
                row_tq.append(0)
                row_tq.append(-1)
                row_tq.append('')

                data_quest.append(row_tq)
        else:
            k += 1

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")

    # Write the merged data to a new CSV file
    if all(data) is True:
        new_filename = './DTT_TO_CSV/target/questionnaire_' + dt_string \
                      + '_Output.csv'
        with open(new_filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(columns)  # Write the updated header row
            writer.writerow(data)  # Write the merged data row
    else:
        print(
            "Warning: Exp %s has not given answers to one or more target, \
             saving his/her answers in a separate folder"
            % names[count].rsplit(' ', 1)[1])
        if os.path.isdir('./DTT_TO_CSV/target_missing'):
            pass
        else:
            os.mkdir('./DTT_TO_CSV/target_missing')
        new_filename = './DTT_TO_CSV/target_missing/questionnaire_' + \
                       dt_string + '_Output.csv'
        with open(new_filename, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(columns)  # Write the updated header row
            writer.writerow(data)  # Write the merged data row

    time.sleep(2)

new_filename = './DTT_TO_CSV/questionnaire.csv'

df = pd.DataFrame(data_quest, columns=header_quest)
df.to_csv(new_filename, index=False)
