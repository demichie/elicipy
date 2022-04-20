def merge_csv(input_dir):

    import os
    import glob
    import pandas as pd

    current_path = os.getcwd()

    foldername = 'seed'
    path = "./"+input_dir+'/'+foldername

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)

    if isExist:

        os.chdir(path)

        extension = 'csv'
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

        timestamp = []
        for f in all_filenames:

            split_f = f.split('_')
            timestamp.append('-'.join(split_f[1:-1]))
    
        #combine all files in the list
        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

        combined_csv.insert(loc=0,column='timestamp',value=timestamp)

        #export to csv
        merged_file = '../'+foldername + '.csv'

        combined_csv.to_csv( merged_file, index=False, encoding='utf-8-sig')

    os.chdir(current_path)

    foldername = 'target'
    path = "./"+input_dir+'/'+foldername

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)

    if isExist:

        os.chdir(path)

        extension = 'csv'
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

        timestamp = []
        for f in all_filenames:

            split_f = f.split('_')
            timestamp.append('-'.join(split_f[1:-1]))
    
        #combine all files in the list
        combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

        combined_csv.insert(loc=0,column='timestamp',value=timestamp)

        #export to csv
        merged_file = '../'+foldername + '.csv'

        combined_csv.to_csv( merged_file, index=False, encoding='utf-8-sig')

    os.chdir(current_path)
