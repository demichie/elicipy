def similar(a, b):

    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def merge_csv(input_dir, target, group):

    import os
    import glob
    import pandas as pd
    import numpy as np
    from itertools import combinations
    from datetime import datetime

    current_path = os.getcwd()

    foldername = 'seed'
    path = input_dir + '/' + foldername

    # Check whether the specified path exists or not
    seedExists = os.path.exists(path)

    if seedExists:

        print('Merging seeds')
        os.chdir(path)

        extension = 'csv'
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

        seedfiles_to_remove = []

        for f in all_filenames:

            seed_df = pd.read_csv(f)
            fgroup = seed_df['Group(s)'].to_list()[0]

            if type(fgroup) == str:

                fgroup = fgroup.split(';')
                fgroup = [eval(i) for i in fgroup]

            else:

                fgroup = [fgroup]

            print('')
            print(f)
            print(seed_df['Last Name'].to_list()[0], 'Group ', fgroup)

            if (group in fgroup) or (group == 0):

                print('Check ok')

            else:

                seedfiles_to_remove.append(f)

        for f in seedfiles_to_remove:

            all_filenames.remove(f)

        print('All seed filenames', len(all_filenames))

        timestamp = []
        for f in all_filenames:

            split_f = f.split('_')
            timestamp.append('-'.join(split_f[1:-1]))

        # combine all files in the list
        combined_seed_csv = pd.concat([pd.read_csv(f) for f in all_filenames],
                                      ignore_index=True)

        combined_seed_csv.insert(loc=0, column='timestamp', value=timestamp)

        combined_seed_csv.drop('Group(s)', axis=1, inplace=True)

        # print('combined_seed_csv',combined_seed_csv)

        fname = combined_seed_csv['First Name'].to_list()
        lname = combined_seed_csv['Last Name'].to_list()
        flname_seed = []
        for f, l in zip(fname, lname):

            flname_seed.append(f + ' ' + l)

        print('Seed experts:', flname_seed)

        # Start a loop to search and discard experts with
        # more than one entry, keeping only the more recent
        # entry.

        check_sim = True

        for i in range(len(timestamp)):

            fname = combined_seed_csv['First Name'].to_list()
            lname = combined_seed_csv['Last Name'].to_list()
            timestamp = combined_seed_csv['timestamp'].to_list()
            flname_seed = []
            lfname_seed = []

            for f, l in zip(fname, lname):

                flname_seed.append(f + ' ' + l)
                lfname_seed.append(l + ' ' + f)

            test_list = range(len(flname_seed))
            res = list(combinations(test_list, 2))

            for (previous, current) in res:

                sim1 = similar(flname_seed[previous], flname_seed[current])
                sim2 = similar(flname_seed[previous], lfname_seed[current])
                sim = max(sim1, sim2)

                check_sim = True

                if sim > 0.8:

                    check_sim = False

                    print('')
                    print('High similarity (', sim,
                          ') found in names of two experts:')
                    print(flname_seed[previous], ',', flname_seed[current])

                    time0 = datetime.strptime(timestamp[previous],
                                              "%Y-%m-%d-%H-%M-%S")
                    time1 = datetime.strptime(timestamp[current],
                                              "%Y-%m-%d-%H-%M-%S")

                    print('Keeping the answers with more recent timestamp:')

                    if time0 > time1:

                        print(time0)
                        disc = current
                        break

                    else:

                        print(time1)
                        disc = previous
                        break

            if check_sim:

                break

            else:

                print('Remove duplicate expert from seed:', flname_seed[disc])
                combined_seed_csv = combined_seed_csv.drop(
                    [combined_seed_csv.index[disc]])
                combined_seed_csv = combined_seed_csv.reset_index(drop=True)
                # print('new',combined_seed_csv)

        # export to csv
        merged_file = '../' + foldername + '.csv'

        combined_seed_csv.to_csv(merged_file,
                                 index=False,
                                 encoding='utf-8-sig')

    os.chdir(current_path)

    foldername = 'target'
    path = input_dir + '/' + foldername

    # Check whether the specified path exists or not
    targetExists = os.path.exists(path)

    if targetExists and target:

        print('')
        print('Merging target')
        os.chdir(path)

        extension = 'csv'
        target_filenames = [i for i in glob.glob('*.{}'.format(extension))]

        print('Number of target files', len(target_filenames))

        targetfiles_to_remove = []

        for f in target_filenames:

            print('')
            print(f)

            target_df = pd.read_csv(f)

            fgroup = target_df['Group(s)'].to_list()[0]

            if type(fgroup) == str:

                fgroup = fgroup.split(';')
                fgroup = [eval(i) for i in fgroup]

            else:

                fgroup = [fgroup]

            print('')
            print(f)
            print(target_df['Last Name'].to_list()[0], 'Group ', fgroup)

            if (group in fgroup) or (group == 0):

                print('Check ok')

            else:

                targetfiles_to_remove.append(f)

        for f in targetfiles_to_remove:

            target_filenames.remove(f)

        print('All target filenames', len(target_filenames))

        timestamp = []
        for f in target_filenames:

            split_f = f.split('_')
            timestamp.append('-'.join(split_f[1:-1]))

        # combine all files in the list
        combined_target_csv = pd.concat(
            [pd.read_csv(f) for f in target_filenames])

        combined_target_csv.insert(loc=0, column='timestamp', value=timestamp)

        combined_target_csv.drop('Group(s)', axis=1, inplace=True)

        # print('combined_target_csv',combined_target_csv)

        for i in range(len(timestamp)):

            fname = combined_target_csv['First Name'].to_list()
            lname = combined_target_csv['Last Name'].to_list()
            timestamp = combined_target_csv['timestamp'].to_list()
            flname_target = []
            lfname_target = []

            for f, l in zip(fname, lname):

                flname_target.append(f + ' ' + l)
                lfname_target.append(l + ' ' + f)

            test_list = range(len(flname_target))
            res = list(combinations(test_list, 2))

            for (previous, current) in res:

                sim1 = similar(flname_target[previous], flname_target[current])
                sim2 = similar(flname_target[previous], lfname_target[current])
                sim = max(sim1, sim2)

                check_sim = True

                if sim > 0.8:

                    check_sim = False

                    print('')
                    print('High similarity (', sim,
                          ') found in names of two experts:')
                    print(flname_target[previous], ',', flname_target[current])

                    time0 = datetime.strptime(timestamp[previous],
                                              "%Y-%m-%d-%H-%M-%S")
                    time1 = datetime.strptime(timestamp[current],
                                              "%Y-%m-%d-%H-%M-%S")

                    print(time0)
                    print(time1)

                    print('Keeping the answers with more recent timestamp:')

                    if time0 > time1:

                        print(time0)
                        disc = current
                        break

                    else:

                        print(time1)
                        disc = previous
                        break

            if check_sim:

                combined_target_csv = combined_target_csv.reset_index(
                    drop=True)
                break

            else:

                combined_target_csv = combined_target_csv.reset_index(
                    drop=True)
                print('Remove duplicate expert from target:',
                      flname_target[disc])
                combined_target_csv = combined_target_csv.drop(
                    [combined_target_csv.index[disc]])
                combined_target_csv = combined_target_csv.reset_index(
                    drop=True)
                # print('new',combined_target_csv)

        if seedExists:

            print('')
            print('Check correspondance between seed and target experts')

            check_matrix = np.zeros((len(flname_target), len(flname_seed)))

            for i in range(len(flname_target)):

                for j in range(len(flname_seed)):

                    sim1 = similar(flname_target[i], flname_seed[j])
                    sim2 = similar(flname_target[i], lfname_seed[j])
                    sim = max(sim1, sim2)

                    if sim > 0.8:

                        check_matrix[i, j] = 1

            print('Seed experts:', flname_seed)
            print('Target experts:', flname_target)

            check_seed = np.sum(check_matrix, axis=0)
            check_target = np.sum(check_matrix, axis=1)

            # print('check_seed',check_seed)
            disc_seed = [
                i for i in range(len(check_seed)) if check_seed[i] == 0
            ]
            # print(disc_seed)
            print('Removed from seed:', [flname_seed[i] for i in disc_seed])
            # print('check_target',check_target)
            disc_target = [
                i for i in range(len(check_target)) if check_target[i] == 0
            ]
            # print(disc_target)
            print('Removed from target:',
                  [flname_target[i] for i in disc_target])

            combined_seed_csv = combined_seed_csv.drop(disc_seed)

            # export to csv
            merged_file = '../' + 'seed' + '.csv'

            combined_seed_csv.to_csv(merged_file,
                                     index=False,
                                     encoding='utf-8-sig')
            print(combined_target_csv)
            print(disc_target)
            combined_target_csv = combined_target_csv.drop(disc_target)

        # export to csv
        merged_file = '../' + foldername + '.csv'

        combined_target_csv.to_csv(merged_file,
                                   index=False,
                                   encoding='utf-8-sig')

    os.chdir(current_path)
