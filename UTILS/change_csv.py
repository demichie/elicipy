import csv
import os
import glob
import sys
from datetime import datetime


def similar(a, b):

    from difflib import SequenceMatcher

    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def ask_change(quest_type):

    current_path = os.getcwd()

    if quest_type == '1':
        quest_type = 'seed'
    else:
        quest_type = 'target'

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

    found_row = None
    for row in data:
        if row[search_col_index] == search_string and row[
                quest_type_col_index] == quest_type:
            found_row = row
            break

    if found_row:
        print(f"String '{search_string}' found")
        # print(header)
        # print(found_row)

        short_q_index = header.index("SHORT Q")
        short_q_value = found_row[short_q_index]
        # minval_index = header.index("MINVAL")
        # minval_value = found_row[minval_index]
        # maxval_index = header.index("MAXVAL")
        # maxval_value = found_row[maxval_index]

        print('Sort q')
        print(short_q_value)

        name = input(
            "Insert a name to find the relative CSVs (field 'Last Name'): ")

        os.chdir('./' + quest_type)
        new_path = os.getcwd()
        print('Current path', new_path)
        # cerca il file CSV nella cartella corrente

        filelist = []
        times = []

        extension = "csv"
        all_filenames = [i for i in glob.glob("*.{}".format(extension))]

        for file in all_filenames:

            # print('File:',file)
            with open(file, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                csv_header = next(csv_reader)
                csv_data = list(csv_reader)
                # print('data')
                # print(csv_data)

                last_name_index = csv_header.index('Last Name')
                # print(csv_data[0][last_name_index])
                # print('Index',last_name_index)
                # A = input('PAUSE')

                name1 = csv_data[0][last_name_index]
                sim1 = similar(name1, name)

                if sim1 > 0.8:
                    print(f"The file '{file}' contains the expert '{name1}'")
                    # print(csv_header)
                    # print(csv_row)
                    filelist.append(file)
                    split_f = file.split("_")
                    timestamp = ("-".join(split_f[1:-1]))
                    times.append(
                        datetime.strptime(timestamp, "%Y-%m-%d-%H-%M-%S"))

        # print('timestamps')
        # print(times)

        imax = 0
        datemax = datetime(2000, 10, 12, 10, 10)
        for count, timestamp in enumerate(times):

            file = filelist[count]
            # print('csv_file',file)
            # print(timestamp,datemax)
            # print(timestamp>datemax)

            if timestamp > datemax:

                with open(file, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    csv_header = next(csv_reader)
                    csv_data = list(csv_reader)

                    for col_index, col_name in enumerate(csv_header):
                        if short_q_value in col_name:
                            # print('col_index',col_index)
                            # print(csv_data[0][col_index])
                            if csv_data[0][col_index] != '':
                                imax = count
                                datemax = timestamp

            # A = input('PAUSE')

        print('Last file with valid data for question ' + search_string)
        # print('imax',imax)
        # print(times[imax])

        file = filelist[imax]
        print('csv_file', file)
        with open(file, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            csv_header = next(csv_reader)
            csv_data = list(csv_reader)
        print('')

        # cerca le colonne che contengono la stringa "short_q_value"
        new_csv_data = []
        for col_index, col_name in enumerate(csv_header):
            # print(col_name)
            if short_q_value in col_name:
                old_value = csv_data[0][col_index]
                print(f"'{col_name}'")
                print(f"Old value '{old_value}'")
                new_value = input("Type the new value: ")
                new_csv_data.append(new_value)
            elif (col_name == 'First Name'):
                new_csv_data.append(csv_data[0][col_index])
            elif (col_name == 'Last Name'):
                new_csv_data.append(csv_data[0][col_index])
            elif (col_name == 'Email address'):
                new_csv_data.append(csv_data[0][col_index])
            elif (col_name == 'Group(s)'):
                new_csv_data.append(csv_data[0][col_index])
            else:
                new_csv_data.append(None)

        now = datetime.now()
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        new_filename = 'questionnaire_' + dt_string + '_Output.csv'

        print('new_file')
        print(new_filename)

        # print('new_data')
        # print(new_csv_data)

        with open(new_filename, 'w', newline='') as new_file:
            csv_writer = csv.writer(new_file)
            csv_writer.writerow(csv_header)
            csv_writer.writerow(new_csv_data)

    os.chdir(current_path)


def main(argv):

    while True:
        quest_type = input('seed (1), target (2), exit (0): ')
        if quest_type not in ('0', '1', '2'):
            print("Not an appropriate choice.")
        else:
            if quest_type in ('1', '2'):
                ask_change(quest_type)
            else:
                break


if __name__ == "__main__":
    main(sys.argv[1:])
