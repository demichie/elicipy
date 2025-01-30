from tools import printProgressBar


def clean_folder(folder_path):

    import os
    import shutil

    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove file or symbolic link
                # print(f"Deleted file: {item_path}")
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directory and its contents
                # print(f"Deleted folder: {item_path}")
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")


def similar(a, b):

    from difflib import SequenceMatcher

    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def merge_csv(input_dir, seed, target, group, csv_file, label_flag,
              write_flag):

    import os
    import time
    import glob
    import pandas as pd
    import numpy as np
    from itertools import combinations
    from datetime import datetime

    verbose = False

    print("       Merging of individual csv files")

    # ----------------------------------------- #
    # ------- READ LABELS FROM CSV_FILE ------- #
    # ----------------------------------------- #

    df_read = pd.read_csv(input_dir + "/" + csv_file, header=0)

    df_SQ = df_read[df_read["QUEST_TYPE"] == "seed"]

    if label_flag:
        label_indexes = df_SQ["LABEL"].astype(str).tolist()
    else:
        label_indexes = np.asarray(df_SQ["IDX"])
        label_indexes = label_indexes.astype(str).tolist()

    # print("label_indexes", label_indexes)

    seed_label_row = ['', '', '', '']
    for label in label_indexes:

        seed_label_row.append(label)
        seed_label_row.append(label)
        seed_label_row.append(label)

    # print(seed_label_row)

    df_TQ = df_read[df_read["QUEST_TYPE"] == "target"]

    if label_flag:
        label_indexes = df_TQ["LABEL"].astype(str).tolist()
    else:
        label_indexes = np.asarray(df_TQ["IDX"])
        label_indexes = label_indexes.astype(str).tolist()

    # print("label_indexes", label_indexes)

    target_label_row = ['', '', '', '']
    for label in label_indexes:

        target_label_row.append(label)
        target_label_row.append(label)
        target_label_row.append(label)

    # print(target_label_row)

    current_path = os.getcwd()

    foldername = "seed"
    path = input_dir + "/" + foldername

    # Check whether the specified path exists or not
    seedExists = os.path.exists(path) and seed

    if seedExists:

        print("       Merging seeds")
        os.chdir(path)

        extension = "csv"
        all_filenames = [i for i in glob.glob("*.{}".format(extension))]

        seedfiles_to_remove = []

        for f in all_filenames:

            if verbose:
                print(f)
            seed_df = pd.read_csv(f)
            fgroup = seed_df["Group(s)"].to_list()[0]

            if type(fgroup) is str:

                fgroup = fgroup.split(";")
                fgroup = [eval(i) for i in fgroup]

            else:

                fgroup = [fgroup]

            if verbose:
                print("")
                print(f)
                print(seed_df["Last Name"].to_list()[0], "Group ", fgroup)

            if (group in fgroup) or (group == 0):

                if verbose:
                    print("Check ok")

            else:

                seedfiles_to_remove.append(f)

        for f in seedfiles_to_remove:

            all_filenames.remove(f)

        # print("All seed filenames", len(all_filenames))

        # get the timestamp of all the files
        timestamp = []
        time_list = []
        for f in all_filenames:

            split_f = f.split("_")
            timestamp.append("-".join(split_f[1:-1]))
            time_list.append(
                datetime.strptime(timestamp[-1], "%Y-%m-%d-%H-%M-%S"))

        # sort by list of all files by timestamp (more recent first)
        sorted_time_list = sorted(time_list, reverse=True)
        time_index = [time_list.index(i) for i in sorted_time_list]
        timestamp_sorted = []
        filelist_sorted = []

        for i in time_index:
            timestamp_sorted.append(timestamp[i])
            filelist_sorted.append(all_filenames[i])

        timestamp = timestamp_sorted
        all_filenames = filelist_sorted

        # combine all files in the list
        combined_seed_csv = pd.concat([pd.read_csv(f) for f in all_filenames],
                                      ignore_index=True,
                                      axis=0)

        combined_seed_csv.insert(loc=0, column="timestamp", value=timestamp)

        combined_seed_csv.sort_values(by='timestamp',
                                      ascending=False,
                                      inplace=True)

        combined_seed_csv.drop("Group(s)", axis=1, inplace=True)

        # print('combined_seed_csv',combined_seed_csv)

        fname = combined_seed_csv["First Name"].to_list()
        lname = combined_seed_csv["Last Name"].to_list()
        flname_seed = []
        for f, l in zip(fname, lname):

            flname_seed.append(str(f) + " " + str(l))

        if verbose:
            print("Seed experts:", flname_seed)

        # Start a loop to search and discard experts with
        # more than one entry, keeping only the more recent
        # entry.

        for i in range(len(timestamp)):

            fname = combined_seed_csv["First Name"].to_list()
            lname = combined_seed_csv["Last Name"].to_list()
            timestamp = combined_seed_csv["timestamp"].to_list()
            flname_seed = []
            lfname_seed = []

            for f, l in zip(fname, lname):

                flname_seed.append(str(f) + " " + str(l))
                lfname_seed.append(str(l) + " " + str(f))

            test_list = range(len(flname_seed))
            res = list(combinations(test_list, 2))

            for (previous, current) in res:

                sim1 = similar(flname_seed[previous], flname_seed[current])
                sim2 = similar(flname_seed[previous], lfname_seed[current])
                sim = max(sim1, sim2)

                check_sim = True

                if sim > 0.8:

                    check_sim = False

                    if verbose:

                        print("")
                        print("High similarity (", sim,
                              ") found in names of two experts:")
                        print(flname_seed[previous], ",", flname_seed[current])

                    time0 = datetime.strptime(timestamp[previous],
                                              "%Y-%m-%d-%H-%M-%S")
                    time1 = datetime.strptime(timestamp[current],
                                              "%Y-%m-%d-%H-%M-%S")

                    if verbose:
                        print(
                            "Keeping the answers with more recent timestamp:")

                    combined_seed_csv.iloc[
                        previous,
                        combined_seed_csv.columns.
                        get_loc("First Name")] = combined_seed_csv[
                            "First Name"].iloc[current]
                    combined_seed_csv.iloc[
                        previous,
                        combined_seed_csv.columns.
                        get_loc("Last Name")] = combined_seed_csv[
                            "Last Name"].iloc[current]

                    if time0 > time1:

                        df2 = combined_seed_csv.iloc[[current, previous], :]

                        # print(time0)
                        disc = current
                        break

                    else:

                        df2 = combined_seed_csv.iloc[[previous, current], :]

                        # print(time1)
                        disc = previous
                        break

            if check_sim:

                break

            else:

                # take the last valid value
                for col in df2.columns:
                    if len(df2[col].dropna()) > 0:

                        combined_seed_csv.iloc[
                            previous,
                            combined_seed_csv.columns.get_loc(col)] = (
                                df2[col].dropna().iloc[-1])
                        combined_seed_csv.iloc[
                            current,
                            combined_seed_csv.columns.get_loc(col)] = (
                                df2[col].dropna().iloc[-1])

                if verbose:
                    print("Remove duplicate expert from seed:",
                          flname_seed[disc])
                combined_seed_csv = combined_seed_csv.drop(
                    [combined_seed_csv.index[disc]])
                combined_seed_csv = combined_seed_csv.reset_index(drop=True)

        # export to csv
        merged_file = "../" + foldername + ".csv"
        if os.path.exists(merged_file):
            os.remove(merged_file)

        column_list = combined_seed_csv.columns
        column_names = []

        for i, label in enumerate(seed_label_row):

            if verbose:
                print(i, label + '. ' + column_list[i])

            if label != "":

                column_names.append(label + '. ' + column_list[i])

            else:

                column_names.append(column_list[i])

        combined_seed_csv.columns = column_names

        combined_seed_csv.to_csv(merged_file,
                                 index=False,
                                 encoding="utf-8-sig")

        if write_flag:

            seed_new_dir = "../SEED_NEW"

            # Check whether the specified output path exists or not
            isExist = os.path.exists(seed_new_dir)

            if not isExist:

                # Create a new directory because it does not exist
                os.makedirs(seed_new_dir)
                print("       The new directory " + seed_new_dir +
                      " is created!")

            else:

                print("       Cleaning " + seed_new_dir)
                clean_folder(seed_new_dir)

            n_experts = len(combined_seed_csv.index)

            print('       Saving seed answers')

            for i in range(n_experts):

                printProgressBar(i, n_experts - 1, prefix='      ')

                df_test = combined_seed_csv.iloc[[i]]
                now = datetime.now()
                dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
                file_test = '../SEED_NEW/questionnaire_' + dt_string + \
                            '_Output.csv'
                df_test.to_csv(file_test, index=False, encoding="utf-8-sig")
                time.sleep(1)

    os.chdir(current_path)

    foldername = "target"
    path = input_dir + "/" + foldername

    # Check whether the specified path exists or not
    targetExists = os.path.exists(path)

    if targetExists and target:

        print("       Merging target")
        os.chdir(path)

        extension = "csv"
        target_filenames = [i for i in glob.glob("*.{}".format(extension))]

        if verbose:
            print("Number of target files", len(target_filenames))

        targetfiles_to_remove = []

        for f in target_filenames:

            target_df = pd.read_csv(f)

            fgroup = target_df["Group(s)"].to_list()[0]

            if type(fgroup) is str:

                fgroup = fgroup.split(";")
                fgroup = [eval(i) for i in fgroup]

            else:

                fgroup = [fgroup]

            if verbose:

                print("")
                print(f)
                print(target_df["Last Name"].to_list()[0], "Group ", fgroup)

            if (group in fgroup) or (group == 0):

                if verbose:
                    print("Check ok")

            else:

                targetfiles_to_remove.append(f)

        for f in targetfiles_to_remove:

            target_filenames.remove(f)

        if verbose:
            print("All target filenames", len(target_filenames))

        timestamp = []
        time_list = []
        for f in target_filenames:

            split_f = f.split("_")
            timestamp.append("-".join(split_f[1:-1]))
            time_list.append(
                datetime.strptime(timestamp[-1], "%Y-%m-%d-%H-%M-%S"))

        # sort by list of all files by timestamp (more recent first)
        sorted_time_list = sorted(time_list, reverse=True)
        time_index = [time_list.index(i) for i in sorted_time_list]
        timestamp_sorted = []
        filelist_sorted = []

        for i in time_index:
            timestamp_sorted.append(timestamp[i])
            filelist_sorted.append(target_filenames[i])

        timestamp = timestamp_sorted
        target_filenames = filelist_sorted

        # print('Timestamps')
        # print(times)
        # combine all files in the list
        combined_target_csv = pd.concat(
            [pd.read_csv(f) for f in target_filenames])

        timestamp_series = pd.Series(timestamp, name='timestamp')

        # Reset the index of combined_target_csv to avoid the
        # "Reindexing" error
        combined_target_csv = combined_target_csv.reset_index(drop=True)

        combined_target_csv = pd.concat(
            [timestamp_series, combined_target_csv], axis=1)

        combined_target_csv.sort_values(by='timestamp',
                                        ascending=False,
                                        inplace=True)

        # print(combined_target_csv["timestamp"])
        # A = input('PAUSE')

        combined_target_csv.drop("Group(s)", axis=1, inplace=True)

        # print('combined_target_csv',combined_target_csv)

        for i in range(len(timestamp)):

            fname = combined_target_csv["First Name"].to_list()
            lname = combined_target_csv["Last Name"].to_list()
            timestamp = combined_target_csv["timestamp"].to_list()
            flname_target = []
            lfname_target = []

            for f, l in zip(fname, lname):

                flname_target.append(str(f) + " " + str(l))
                lfname_target.append(str(l) + " " + str(f))

            test_list = range(len(flname_target))
            res = list(combinations(test_list, 2))

            for (previous, current) in res:

                if previous == current:

                    continue

                sim1 = similar(flname_target[previous], flname_target[current])
                sim2 = similar(flname_target[previous], lfname_target[current])
                sim = max(sim1, sim2)

                check_sim = True

                if sim > 0.8:

                    check_sim = False

                    if verbose:

                        print("")
                        print("High similarity (", sim,
                              ") found in names of two experts:")
                        print(flname_target[previous], ",",
                              flname_target[current])

                    time0 = datetime.strptime(timestamp[previous],
                                              "%Y-%m-%d-%H-%M-%S")
                    time1 = datetime.strptime(timestamp[current],
                                              "%Y-%m-%d-%H-%M-%S")

                    # print('time0',time0)
                    # print('time1',time1)

                    if verbose:
                        print(
                            "Keeping the answers with more recent timestamp:")

                    if time0 > time1:

                        df2 = combined_target_csv.iloc[[current, previous], :]

                        # print(time0)
                        disc = current
                        break

                    else:

                        df2 = combined_target_csv.iloc[[previous, current], :]

                        # print(time1)
                        disc = previous
                        break

            if check_sim:

                combined_target_csv = combined_target_csv.reset_index(
                    drop=True)
                break

            else:

                # take the last valid value
                for count, col in enumerate(df2.columns):

                    # print('')
                    # print(flname_target[current])
                    # print('Question', count, time0, time1)
                    # print(df2[col])

                    if len(df2[col].dropna()) > 0:

                        combined_target_csv.iloc[
                            previous,
                            combined_target_csv.columns.get_loc(col)] = (
                                df2[col].dropna().iloc[-1])
                        combined_target_csv.iloc[
                            current,
                            combined_target_csv.columns.get_loc(col)] = (
                                df2[col].dropna().iloc[-1])

                combined_target_csv = combined_target_csv.reset_index(
                    drop=True)
                if verbose:
                    print("Remove duplicate expert from target:",
                          flname_target[disc])
                combined_target_csv = combined_target_csv.drop(
                    [combined_target_csv.index[disc]])
                combined_target_csv = combined_target_csv.reset_index(
                    drop=True)
                # print('new',combined_target_csv)
                # a = input('PAUSE')

        if seedExists:

            print(
                "       Checking correspondance between seed and target " +
                "experts"
            )

            check_matrix = np.zeros((len(flname_target), len(flname_seed)))

            for i in range(len(flname_target)):

                for j in range(len(flname_seed)):

                    sim1 = similar(flname_target[i], flname_seed[j])
                    sim2 = similar(flname_target[i], lfname_seed[j])
                    sim = max(sim1, sim2)

                    if sim > 0.8:

                        check_matrix[i, j] = 1

            if verbose:

                print("Seed experts:", flname_seed)
                print("Target experts:", flname_target)

            # print('check_matrix\n',check_matrix)

            check_seed = np.sum(check_matrix, axis=0)
            check_target = np.sum(check_matrix, axis=1)

            # print('check_seed',check_seed)
            disc_seed = [
                i for i in range(len(check_seed)) if check_seed[i] == 0
            ]
            # print(disc_seed)
            if verbose:
                print("Removed from seed:",
                      [flname_seed[i] for i in disc_seed])
            # print('check_target',check_target)
            disc_target = [
                i for i in range(len(check_target)) if check_target[i] == 0
            ]
            # print(disc_target)
            if verbose:
                print("Removed from target:",
                      [flname_target[i] for i in disc_target])

            combined_seed_csv = combined_seed_csv.drop(disc_seed)

            # export to csv
            merged_file = "../" + "seed" + ".csv"
            if os.path.exists(merged_file):
                os.remove(merged_file)

            combined_seed_csv.to_csv(merged_file,
                                     index=False,
                                     encoding="utf-8-sig")
            # print(combined_target_csv)
            # print(disc_target)
            combined_target_csv = combined_target_csv.drop(disc_target)

        # export to csv
        merged_file = "../" + foldername + ".csv"

        column_list = combined_target_csv.columns
        column_names = []

        for i, label in enumerate(target_label_row):

            if verbose:
                print(i, label + '. ' + column_list[i])

            if label != "":

                column_names.append(label + '. ' + column_list[i])

            else:

                column_names.append(column_list[i])

        combined_target_csv.columns = column_names

        combined_target_csv.to_csv(merged_file,
                                   index=False,
                                   encoding="utf-8-sig")

        if write_flag:

            target_new_dir = "../TARGET_NEW"

            # Check whether the specified output path exists or not
            isExist = os.path.exists(target_new_dir)

            if not isExist:

                # Create a new directory because it does not exist
                os.makedirs(target_new_dir)
                print("       The new directory " + target_new_dir +
                      " is created!")

            else:

                print("       Cleaning " + target_new_dir)
                clean_folder(target_new_dir)

            n_experts = len(combined_target_csv.index)

            print('       Saving target answers')

            for i in range(n_experts):

                printProgressBar(i, n_experts - 1, prefix='      ')

                df_test = combined_target_csv.iloc[[i]]
                now = datetime.now()
                dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
                file_test = '../TARGET_NEW/questionnaire_' + dt_string + \
                            '_Output.csv'
                df_test.to_csv(file_test, index=False, encoding="utf-8-sig")
                time.sleep(1)

    os.chdir(current_path)
