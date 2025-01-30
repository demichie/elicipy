import pandas as pd
import os
import shutil as st


def replace_strings(working_dir, df, header, i_row):

    main_dir = os.getcwd()
    os.chdir(working_dir)
    directory = os.getcwd()

    for fname in os.listdir(directory):

        if os.path.isfile(fname):
            # print(fname)
            # Full path
            f = open(fname, "r")
            filedata = f.read()

            for name in header:

                searchstring = name
                # print(searchstring)
                f.seek(0)

                if searchstring in filedata:

                    print("found string in file %s" % fname)
                    print(searchstring)
                    f.seek(0)

                    for line in f:
                        # replacing the string and write to output file
                        filedata = filedata.replace(
                            searchstring, str(df.at[i_row, name])
                        )

            f.close()
            f_out = open(fname, "w")
            f_out.write(filedata)
            f_out.close()

    os.chdir(main_dir)


##########################################


def main():

    from ElicipyDict import elicitation_name
    from ElicipyDict import output_dir

    csv_file = output_dir + "/" + elicitation_name + "_samples.csv"

    df = pd.read_csv(csv_file)

    print(df)

    header = df.columns

    print(header)

    n_rows = len(df.index)
    for i_row in range(n_rows):

        working_dir = "ensemble." + "{:05d}".format(i_row)
        working_dir = os.path.join(os.getcwd(), working_dir)

        st.copytree("templatedir", working_dir, symlinks=True)

        replace_strings(working_dir, df, header, i_row)


if __name__ == "__main__":

    main()
