import csv
import os
import glob
import sys
import time
from datetime import datetime


def similar(a, b):

    from difflib import SequenceMatcher

    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

quest_type = '2'

current_path = os.getcwd()

if quest_type == '1':
    quest_type = 'seed'
else:
     quest_type = 'target'

file_to_insert = 'file1.csv'

# legge il file CSV
filename = 'questionnaire.csv'
with open(filename, 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)  # legge la prima riga come header
    data = list(csv_reader)  # legge il resto del file come dati

# chiede all'utente una stringa da cercare nella colonna "LABEL"
search_string = input("Insert question 'LABEL': ")

# cerca la riga con il valore corrispondente nella colonna "LABEL"
search_col_index = header.index("LABEL")
quest_type_col_index = header.index("QUEST_TYPE")
minval_col_index = header.index("MINVAL")
maxval_col_index = header.index("MAXVAL")
units_col_index = header.index("UNITS")
short_q_index = header.index("SHORT Q")


found_row = None
for count,row in enumerate(data):
    print(row[search_col_index])
    print(row[search_col_index] == search_string)
    print(row[quest_type_col_index])
    print(row[quest_type_col_index] == quest_type)
    if row[search_col_index] == search_string and row[quest_type_col_index] == quest_type:
        found_row = row
        row_index = count        
        break

if found_row:
    print(f"String '{search_string}' found")
    # print(header)
    # print(found_row)

    short_q_value = found_row[short_q_index]
    short_q_next_value = data[count+1][short_q_index]

    minval = found_row[minval_col_index]
    maxval = found_row[maxval_col_index]
    units = found_row[units_col_index]

    
print('New question:')
print(short_q_value)

print('Must be before question:')
print(short_q_next_value)

name1 = input("Insert a name to find the relative CSVs (field 'Last Name'): ")

os.chdir('./' + quest_type)

extension = "csv"
all_filenames = [i for i in glob.glob("*.{}".format(extension))]

filelist = []

for file in all_filenames:

    with open(file, 'r') as csv_file:
            
        reader2 = csv.reader(csv_file)
        header2 = next(reader2)
        data2 = list(reader2)

        last_name_index = header2.index('Last Name')

        name2 = data2[0][last_name_index]
        sim1 = similar(name1, name2)

        if sim1 > 0.8:

            filelist.append(file)
            
print(filelist)

pctls = [5,50,95]

for pct in pctls:

    Qstr = short_q_value + ' - ' + str(int(pct)) + '%ile (' + str(minval) + ';' + str(maxval) + ')' + ' [' + units + ']'
    Qstr_list.append(Qstr)
            
    ans_list.append(input(Qstr))

for file in filelist:

    Qstr_list = []
    ans_list = []
    
    # Open the second CSV file and read its contents
    with open(file, 'r') as file2:
        reader2 = csv.reader(file2)
        header2 = next(reader2)  # Save the header row
        data2 = next(reader2)  # Save the data row

    for col_index, col_name in enumerate(header2):
            
        if short_q_next_value in col_name:
            insert_index = col_index
            break

        
    for i in range(len(Qstr_list)):
        header2.insert(insert_index + i, Qstr_list[i])
        data2.insert(insert_index + i, ans_list[i])

    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    new_filename = 'questionnaire_' + dt_string + '_Output.csv'
        
        
    # Write the merged data to a new CSV file
    with open(new_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header2)  # Write the updated header row
        writer.writerow(data2)  # Write the merged data row

    time.sleep(5)
