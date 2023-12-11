import datetime
import numpy as np
import os
import sys
import pandas as pd

from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE
from pptx import Presentation

from saveFromGithub import saveDataFromGithub
from createSamples import createSamples
from COOKEweights import COOKEweights
from ERFweights import ERFweights
from merge_csv import merge_csv

from createPlots import create_fig_hist
from createPlots import create_figure_violin
from createPlots import create_figure_pie
from createPlots import create_figure_trend
from createPlots import create_figure_answers
from createPlots import create_barplot

max_len_table = 21
max_len_tableB = 18

max_len_plot = 21



# Print iterations progress
def printProgressBar(iteration,
                     total,
                     prefix='',
                     suffix='',
                     decimals=1,
                     bar_length=100):
    """
    Call in a loop to create terminal progress bar

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' %
                     (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def add_date(slide):

    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(0.2),
                                   Inches(16), Inches(0.3))
    shape.shadow.inherit = False
    fill = shape.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(60, 90, 180)
    line = shape.line
    line.color.rgb = RGBColor(60, 90, 180)
    shape.text = "Expert elicitation " + \
        datetime.datetime.today().strftime("%d-%b-%Y")
    shape_para = shape.text_frame.paragraphs[0]
    shape_para.font.name = "Helvetica"
    shape_para.font.size = Pt(17)


def add_small_logo(slide, left, top, logofile):

    slide.shapes.add_picture(logofile,
                             left + Inches(13.0),
                             top + Inches(6.8),
                             width=Inches(0.8))


def add_figure(slide, figname, left, top, width):

    img = slide.shapes.add_picture(figname,
                                   left + Inches(3.4),
                                   top,
                                   width=width)

    slide.shapes._spTree.insert(2, img._element)


def add_small_figure(slide, figname, left, top, width):

    img = slide.shapes.add_picture(figname, left, top, width=width)

    slide.shapes._spTree.insert(2, img._element)


def add_title(slide, text_title):

    title_shape = slide.shapes.title
    title_shape.text = text_title
    title_shape.top = Inches(0.2)
    title_shape.width = Inches(15)
    title_shape.height = Inches(2)
    title_para = slide.shapes.title.text_frame.paragraphs[0]
    title_para.font.name = "Helvetica"
    if len(text_title) < 50:

        title_para.font.size = Pt(44)

    else:

        title_para.font.size = Pt(34)


def add_text_box(slide, left, top, text_box, font_size):

    txBox = slide.shapes.add_textbox(left - Inches(1),
                                     top + Inches(0.5),
                                     width=Inches(4),
                                     height=Inches(5))
    tf = txBox.text_frame
    tf.text = text_box
    tf.paragraphs[0].font.size = Pt(font_size)
    # tf.text = 'prova'
    tf.word_wrap = True


def iter_cells(table):
    for row in table.rows:
        for cell in row.cells:
            yield cell


def read_answers(input_dir, csv_file, group, n_pctl, df_indexes_SQ,
                 df_indexes_TQ, seed, target, output_dir, elicitation_name,
                 write_flag, label_indexes):

    try:

        from ElicipyDict import label_flag

    except ImportError:

        label_flag = False

    import difflib

    # merge the files of the different experts
    # creating one file for seeds and one for tagets
    merge_csv(input_dir, seed, target, group, csv_file, label_flag, write_flag)

    if seed:
        
        # seeds file name
        filename = input_dir + "/seed.csv"

        # Read a comma-separated values (csv) file into DataFrame df_SQ
        df_SQ = pd.read_csv(filename)

        # create a 2D numpy array with the answers to the seed questions
        cols_as_np = df_SQ[df_SQ.columns[4:]].to_numpy()

        # we want to work with a 3D array, with the following dimension:
        # n_expert X n_pctl X n_SQ
        n_experts = cols_as_np.shape[0]
        n_SQ = int(cols_as_np.shape[1] / n_pctl)

        # The second column (index 1) must contain the first name of the expert
        firstname = df_SQ[df_SQ.columns[1]].astype(str).tolist()

        # The third column (index 2) must contain the first name of the expert
        surname = df_SQ[df_SQ.columns[2]].astype(str).tolist()

        # create a list with firstname+surname
        # this is needed to search for expert matches between
        # seed and target questions
        NS_SQ = []

        for name, surname in zip(firstname, surname):

            NS_SQ.append(name + surname)

        print("NS_SQ", NS_SQ)

        csv_name = output_dir + "/" + elicitation_name + "_experts.csv"

        d = {"index": range(1, len(NS_SQ) + 1), "Expert": NS_SQ}

        df = pd.DataFrame(data=d)

        df.to_csv(csv_name, index=False)

        # reshaped numpy array with expert answers
        SQ_array = np.reshape(cols_as_np, (n_experts, n_SQ, n_pctl))

        # swap the array to have seed for the last index
        SQ_array = np.swapaxes(SQ_array, 1, 2)

        # sort according to the percentile values
        # (sometimes the expert give the percentiles in the wrong order)
        SQ_array = np.sort(SQ_array, axis=1)

        # chech and correct for equal percentiles
        for i in np.arange(n_SQ):

            for k in np.arange(n_experts):

                # if 5% and 50% percentiles are equal, reduce 5%
                if SQ_array[k, 0, i] == SQ_array[k, 1, i]:

                    SQ_array[k, 0, i] = SQ_array[k, 1, i] * 0.99

                # if 50% and 95% percentiles are equal, increase 95%
                if SQ_array[k, 2, i] == SQ_array[k, 1, i]:

                    SQ_array[k, 2, i] = SQ_array[k, 1, i] * 1.01

        # if we have a subset of the SQ, then extract from SQ_array
        # the correct slice
        if len(df_indexes_SQ) > 0:

            # print('SQ_array',SQ_array)
            SQ_array = SQ_array[:, :, df_indexes_SQ]
            n_SQ = len(df_indexes_SQ)
            # print('SQ_array',SQ_array)

        for i in np.arange(n_SQ):
        
            print("")
            print("Seed question ", label_indexes[i])
            print(SQ_array[:, :, i])
                    
    else:

        n_SQ = 0
        df_indexes_SQ = []
                    
    if target:

        filename = input_dir + "/target.csv"

        # Read a comma-separated values (csv) file into DataFrame df_TQ
        df_TQ = pd.read_csv(filename)

        # The second column (index 1) must contain the first name of the expert
        firstname = df_TQ[df_TQ.columns[1]].astype(str).tolist()

        # The third column (index 2) must contain the first name of the expert
        surname = df_TQ[df_TQ.columns[2]].astype(str).tolist()

        # create a list with firstname+surname
        # this is needed to search for expert matches between seed
        # and target questions
        NS_TQ = []

        for name, surname in zip(firstname, surname):

            NS_TQ.append(name + surname)

        if seed:    

            sorted_idx = []

            # loop to search for matches between experts in seed and target
            for SQ_name in NS_SQ:

                index = NS_TQ.index(difflib.get_close_matches(SQ_name, NS_TQ)[0])
                sorted_idx.append(index)

            print("Sorted list of experts to match the order of seeds:",
                sorted_idx)

            print(NS_SQ)
            print([NS_TQ[s_idx] for s_idx in sorted_idx])

        else:

            sorted_idx = range(len(NS_TQ))
            
        # create a 2D numpy array with the answers to the target questions
        cols_as_np = df_TQ[df_TQ.columns[4:]].to_numpy()

        # sort for expert names
        cols_as_np = cols_as_np[sorted_idx, :]

        # we want to work with a 3D array, with the following dimension:
        # n_expert X n_pctl X n_TQ
        n_experts = cols_as_np.shape[0]

        n_TQ = int(cols_as_np.shape[1] / n_pctl)

        # reshaped numpy array with expert answers
        TQ_array = np.reshape(cols_as_np, (n_experts, n_TQ, n_pctl))

        # swap the array to have pctls for the second index
        TQ_array = np.swapaxes(TQ_array, 1, 2)

        # sort according to the percentile values
        # (sometimes the expert give the percentiles in the wrong order)
        TQ_array = np.sort(TQ_array, axis=1)

        for i in np.arange(n_TQ):

            for k in np.arange(n_experts):
                if TQ_array[k, 0, i] == TQ_array[k, 1, i]:
                    TQ_array[k, 0, i] = TQ_array[k, 1, i] * 0.99
                if TQ_array[k, 2, i] == TQ_array[k, 1, i]:
                    TQ_array[k, 2, i] = TQ_array[k, 1, i] * 1.01

        if len(df_indexes_TQ) > 0:

            # print('TQ_array',TQ_array)
            TQ_array = TQ_array[:, :, df_indexes_TQ]
            n_TQ = len(df_indexes_TQ)
            # print('TQ_array',TQ_array)

        for i in np.arange(n_TQ):

            print("Target question ", label_indexes[i+n_SQ])
            print(TQ_array[:, :, i])
            
    else:
        
        n_TQ = 0
        TQ_array = np.zeros((n_experts, n_pctl, n_TQ))

    if not seed:
        
        SQ_array = np.zeros((n_experts, n_pctl, n_SQ))

        
    return n_experts, n_SQ, n_TQ, SQ_array, TQ_array


def read_questionnaire(input_dir, csv_file, seed, target):
    """Read .csv questionnaire file

    Parameters
    ----------
    input_dir : string
        name of input folder
    csv_file : string
        name of csv_file
    target : boolean
        True to read targets

    Returns
    -------
    df_indexes_SQ, df_indexes_TQ, SQ_scale, SQ_realization, TQ_scale, \
        SQ_minVals, SQ_maxVals, TQ_minVals, TQ_maxVals, SQ_units, TQ_units, \
        SQ_LongQuestion, TQ_LongQuestion, SQ_question, TQ_question, \
        idx_list, global_scale, global_log, label_indexes, parents, \
        global_idxMin, global_idxMax, global_sum50

    """

    try:

        from ElicipyDict import label_flag

    except ImportError:

        label_flag = False

    df_read = pd.read_csv(input_dir + "/" + csv_file, header=0)
    # print(df_read)

    quest_type = df_read["QUEST_TYPE"].to_list()
    n_SQ = quest_type.count("seed")

    print(quest_type)

    try:

        from ElicipyDict import seed_list

        print("seed_list read", seed_list)

    except ImportError:

        print("ImportError")
        seed_list = list(df_read["IDX"])[0:n_SQ]

    print("seed_list", seed_list)

    # extract the seed questions with index in
    # seed_list (from column IDX)
    new_indexes = []

    for idx in seed_list[:]:

        indices = df_read.index[
            (df_read["IDX"] == idx)
            & (df_read.QUEST_TYPE.str.contains("seed"))]
            
        new_indexes.append(indices[0])

    df_SQ = df_read.iloc[new_indexes]
    df_indexes_SQ = np.array(new_indexes).astype(int)

    print("df_indexes_SQ", df_indexes_SQ)

    if target:

        try:

            from ElicipyDict import target_list

            print("target_list read", target_list)

        except ImportError:

            print("ImportError")
            target_list = list(df_read["IDX"])[n_SQ:]

        print("target_list", target_list)

        # extract the target questions with index in
        # target_list (from column IDX)
        new_indexes = []

        for idx in target_list[:]:

            indices = df_read.index[
                (df_read["IDX"] == idx)
                & (df_read.QUEST_TYPE.str.contains("target"))]

            new_indexes.append(indices[0])

        df_TQ = df_read.iloc[new_indexes]
        df_indexes_TQ = np.array(new_indexes).astype(int)-n_SQ

        print("df_indexes_TQ", df_indexes_TQ)

        if seed:

            df_quest = df_SQ.append(df_TQ)

        else:

            df_quest = df_TQ

    else:

        df_indexes_TQ = []
        df_quest = df_SQ

    if label_flag:
        label_indexes = df_quest["LABEL"].tolist()
    else:
        label_indexes = np.asarray(df_quest["IDX"])
        label_indexes = label_indexes.astype(str).tolist()

    print("label_indexes", label_indexes)

    data_top = df_quest.head()

    langs = []

    # check if there are multiple languages
    for head in data_top:

        if "LONG Q" in head:

            string = head.replace("LONG Q", "")
            string2 = string.replace("_", "")

            langs.append(string2)

    print("Languages:", langs)

    try:

        from ElicipyDict import language

    except ImportError:

        language = ""

    # select the columns to use according with the language
    if len(langs) > 1:

        if language in langs:

            lang_index = langs.index(language)
            # list of column indexes to use
            index_list = [1, 2, 3, lang_index + 4] + list(
                range(len(langs) + 4,
                      len(langs) + 15))

        else:

            raise Exception("Sorry, language is not in questionnaire")

    else:

        lang_index = 0
        language = ""
        index_list = list(range(1, 16))

    # list with the short title of the target questions
    SQ_question = []
    # list with the long title of the target questions
    SQ_LongQuestion = []
    # list with min vals for target questions
    SQ_minVals = []
    # list with max vals for target questions
    SQ_maxVals = []
    # list with the units of the target questions
    SQ_units = []
    # scale for target question:
    SQ_scale = []

    SQ_realization = []

    global_log = []

    global_idxMin = []
    global_idxMax = []
    global_sum50 = []

    for i in df_quest.itertuples():

        (
            idx,
            label,
            shortQ,
            longQ,
            unit,
            scale,
            minVal,
            maxVal,
            realization,
            question,
            idxMin,
            idxMax,
            sum50,
            parent,
            image,
        ) = [i[j] for j in index_list]

        minVal = float(minVal)
        maxVal = float(maxVal)

        if scale == "uni":

            global_log.append(0)

        else:

            global_log.append(1)

        if question == "seed":

            SQ_question.append(shortQ)
            SQ_LongQuestion.append(longQ)
            SQ_units.append(unit)
            SQ_scale.append(scale)

            SQ_realization.append(float(realization))

            if minVal.is_integer():

                minVal = int(minVal)

            if maxVal.is_integer():

                maxVal = int(maxVal)

            SQ_minVals.append(minVal)
            SQ_maxVals.append(maxVal)

            if idxMin>0:
                df_min = df_quest.loc[(df_quest['IDX'] == idxMin) & (df_quest['QUEST_TYPE'] == 'seed')]
                global_idxMin.append(str(df_min['LABEL'].iloc[0]))
            else:
                 global_idxMin.append('')

            if idxMax>0:
                df_max = df_quest.loc[(df_quest['IDX'] == idxMax) & (df_quest['QUEST_TYPE'] == 'seed')]
                global_idxMax.append(str(df_max['LABEL'].iloc[0]))
            else:
                global_idxMax.append('')
            
            global_sum50.append(sum50)

    # print on screen the units
    print("Seed_units = ", SQ_units)

    # print on screen the units
    print("Seed_scales = ", SQ_scale)

    # list with the "title" of the target questions
    TQ_question = []
    # list with the long title of the target questions
    TQ_LongQuestion = []
    TQ_minVals = []
    TQ_maxVals = []
    # list with the units of the target questions
    TQ_units = []
    # scale for target question:
    TQ_scale = []

    idx_list = []
    parents = []

    if target:

        for i in df_quest.itertuples():

            (
                idx,
                label,
                shortQ,
                longQ,
                unit,
                scale,
                minVal,
                maxVal,
                realization,
                question,
                idxMin,
                idxMax,
                sum50,
                parent,
                image,
            ) = [i[j] for j in index_list]

            minVal = float(minVal)
            maxVal = float(maxVal)
            if question == "target":

                TQ_question.append(shortQ)
                TQ_LongQuestion.append(longQ)
                TQ_units.append(unit)
                TQ_scale.append(scale)

                if minVal.is_integer():

                    minVal = int(minVal)

                if maxVal.is_integer():

                    maxVal = int(maxVal)

                TQ_minVals.append(minVal)
                TQ_maxVals.append(maxVal)
                idx_list.append(int(idx))
                parents.append(int(parent))

                if (idxMin>0) and (question == "target"):
                    df_min = df_quest.loc[(df_quest['IDX'] == idxMin) & (df_quest['QUEST_TYPE'] == 'target')]
                    global_idxMin.append(str(df_min['LABEL'].iloc[0]))
                else:
                    global_idxMin.append('')

                if idxMax>0 and  question == "target":
                    df_max = df_quest.loc[(df_quest['IDX'] == idxMax) & (df_quest['QUEST_TYPE'] == 'target')]
                    global_idxMax.append(str(df_max['LABEL'].iloc[0]))
                else:
                    global_idxMax.append('')
                    
                global_sum50.append(sum50)

        # print on screen the units
        print("Target units = ", TQ_units)

        # print on screen the units
        print("Target scales = ", TQ_scale)

        global_scale = SQ_scale + TQ_scale

    else:

        global_scale = SQ_scale

    return (df_indexes_SQ, df_indexes_TQ, SQ_scale, SQ_realization, TQ_scale,
            SQ_minVals, SQ_maxVals, TQ_minVals, TQ_maxVals, SQ_units, TQ_units,
            SQ_LongQuestion, TQ_LongQuestion, SQ_question, TQ_question,
            idx_list, global_scale, global_log, label_indexes, parents,
            global_idxMin, global_idxMax, global_sum50, df_quest)


def answer_analysis(
    input_dir,
    csv_file,
    n_experts,
    n_SQ,
    n_TQ,
    SQ_array,
    TQ_array,
    realization,
    global_scale,
    global_log,
    alpha,
    overshoot,
    cal_power,
    ERF_flag,
    Cooke_flag,
    seed
):

    if Cooke_flag > 0:
    
        W, score, information, M = COOKEweights(SQ_array, TQ_array, realization,
                                            alpha, global_scale, overshoot,
                                            cal_power, Cooke_flag)
        print('score',score)
        print('information',information)
        
    else:

        W = np.ones((n_experts,5))
        score = np.zeros(n_experts)
        information = np.zeros(n_experts)
        
    if ERF_flag:
        
        W_erf, score_erf = ERFweights(realization, SQ_array)

    else:

        W_erf = np.ones((n_experts,5))
        score_erf = np.zeros(n_experts)

    sensitivity = False

    if sensitivity:

        for i in range(n_SQ):

            SQ_temp = np.delete(SQ_array, i, axis=2)
            realization_temp = np.delete(realization, i)
            global_scale_temp = np.delete(global_scale, i)

            W_temp, score_temp, information_temp, M = COOKEweights(
                SQ_temp,
                TQ_array,
                realization_temp,
                alpha,
                global_scale_temp,
                overshoot,
                cal_power,
                Cooke_flag
            )

            W_reldiff = W[:, 3] / \
                np.sum(W[:, 3]) - W_temp[:, 3] / np.sum(W_temp[:, 3])

            W_mean = np.mean(np.abs(W_reldiff))
            W_std = np.sqrt(np.sum(np.square(W_reldiff)) / n_experts)

            print(i + 1, W_mean, W_std, np.sum(W_temp[:, 4] > 0))

            W_erf_temp, score_erf_temp = ERFweights(realization_temp, SQ_temp)

            W_reldiff = W_erf[:, 4] - W_erf_temp[:, 4]

            W_mean = np.mean(np.abs(W_reldiff))
            W_std = np.sqrt(np.sum(np.square(W_reldiff)) / n_experts)

            print(i + 1, W_mean, W_std, np.sum(W_erf_temp > alpha))

    Weq = np.ones(n_experts)
    Weqok = [x / n_experts for x in Weq]

    W_gt0_01 = []
    Werf_gt0_01 = []
    expin = []

    k = 1
    for x, y in zip(W[:, 4], W_erf[:, 4]):
        if (x > 0) or (y > 0):
            W_gt0_01.append(x)
            Werf_gt0_01.append(y)
            expin.append(k)
        k += 1

    W_gt0 = [round((x * 100.0), 5) for x in W_gt0_01]
    Werf_gt0 = [round((x * 100.0), 5) for x in Werf_gt0_01]

    if ERF_flag > 0:

        print("")
        print("W_erf")
        print(W_erf[:, -1])

    if Cooke_flag > 0:

        print("")
        print("W_cooke")
        print(W[:, -1])
        print("score", score)
        print("information", information)
        print("unNormalized weight", W[:, 3])

    print("")
    print("Weq")
    print(Weqok)

    return W, W_erf, Weqok, W_gt0, Werf_gt0, expin


def save_dtt_rll(input_dir, n_experts, n_SQ, n_TQ, df_quest, target,
                 SQ_realization, SQ_scale, SQ_array, TQ_scale, TQ_array):

    # ----------------------------------------- #
    # ------------ Save dtt and rls ----------- #
    # ----------------------------------------- #

    # Save a reference to the original standard output
    original_stdout = sys.stdout

    filename = input_dir + "/seed.rls"

    with open(filename, "w") as f:
        sys.stdout = f  # Change the standard output to the file we created.

        for i in np.arange(n_SQ):

            # print(i+1,str(i+1),SQ_realization[i],SQ_scale[i])

            print(f'{i+1:>5d} {"SQ"+str(i+1):>13} {""} ' +
                  f'{SQ_realization[i]:6e} {SQ_scale[i]:4}')

        # Reset the standard output to its original value
        sys.stdout = original_stdout

    if target:

        filename = input_dir + "/seed_and_target.dtt"

    else:

        filename = input_dir + "/seed.dtt"

    with open(filename, "w") as f:
        sys.stdout = f  # Change the standard output to the file we created.

        print("* CLASS ASCII OUTPUT FILE. NQ=   3   QU=   5  50  95")
        print("")
        print("")

        for k in np.arange(n_experts):

            for i in np.arange(n_SQ):

                print(f'{k+1:5d} {"Exp"+str(k+1):>8} {i+1:4d} ' +
                      f'{"SQ"+str(i+1):>13} {SQ_scale[i]:4} ' +
                      f'{SQ_array[k, 0, i]:6e} {""} {SQ_array[k, 1, i]:6e} ' +
                      f'{" "}{SQ_array[k, 2, i]:6e}')

            if target:

                for i in np.arange(n_TQ):

                    print(
                        f'{k+1:5d} {"Exp"+str(k+1):>8} {i+1:4d} ' +
                        f'{"TQ"+str(i+1):>13} {TQ_scale[i]:4} ' +
                        f'{TQ_array[k, 0, i]:6e} {""} {TQ_array[k, 1, i]:6e} ' +
                        f'{" "}{TQ_array[k, 2, i]:6e}')

        # Reset the standard output to its original value
        sys.stdout = original_stdout

    return

def create_samples(group, n_experts, n_SQ, n_TQ, n_pctl, SQ_array, TQ_array, n_sample, W,
                   W_erf, Weqok, W_gt0, Werf_gt0, expin, global_log, 
                   global_minVal, global_maxVal, label_indexes,
                   ERF_flag, Cooke_flag, EW_flag, overshoot, globalSum, normalizeSum):

    DAT = np.zeros((n_experts * (n_SQ + n_TQ), n_pctl + 2))

    DAT[:, 0] = np.repeat(np.arange(1, n_experts + 1), n_SQ + n_TQ)
    DAT[:, 1] = np.tile(np.arange(1, n_SQ + n_TQ + 1), n_experts)

    DAT[:, 2:] = np.append(SQ_array, TQ_array,
                           axis=2).transpose(0, 2, 1).reshape(-1, 3)
    q_Cooke = np.zeros((n_SQ + n_TQ, 4))
    q_erf = np.zeros((n_SQ + n_TQ, 4))
    q_EW = np.zeros((n_SQ + n_TQ, 4))

    samples = np.zeros((n_sample, n_SQ+n_TQ))
    samples_erf = np.zeros((n_sample, n_SQ+n_TQ))
    samples_EW = np.zeros((n_sample, n_SQ+n_TQ))

    print('Creating samples for question')

    for j in np.arange(n_SQ + n_TQ):
    
        printProgressBar(j,n_SQ + n_TQ-1)

        C_EW = createSamples(
            DAT,
            j,
            Weqok,
            n_sample,
            global_log[j],
            [global_minVal[j], global_maxVal[j]],
            overshoot,
            0,
        )
            
        samples_EW[:, j] = C_EW
            
        if Cooke_flag>0:

            C = createSamples(
                DAT,
                j,
                W[:, 4].flatten(),
                n_sample,
                global_log[j],
                [global_minVal[j], global_maxVal[j]],
                overshoot,
                0,
            )
                
        else:
            
            C = C_EW    

        samples[:, j] = C
                
        if ERF_flag > 0:

            C_erf = \
                createSamples(
                    DAT,
                    j,
                    W_erf[:, 4].flatten(),
                    n_sample,
                    global_log[j],
                    [global_minVal[j], global_maxVal[j]],
                    overshoot,
                    ERF_flag,
                )

        else:

            C_erf = C_EW

        samples_erf[:, j] = C_erf
       
    if ( normalizeSum): 
               
        for triplet in globalSum:
    
            print(triplet[0],triplet[1],triplet[2])
        
            sum_samples = np.sum(samples_EW[:,n_SQ+triplet[0]:n_SQ+triplet[1]+1], axis=1)
            sum_samples = np.expand_dims(sum_samples,axis=1)
            samples_EW[:,n_SQ+triplet[0]:n_SQ+triplet[1]+1] /= sum_samples 
            samples_EW[:,n_SQ+triplet[0]:n_SQ+triplet[1]+1] *= triplet[2]

            sum_samples = np.sum(samples[:,n_SQ+triplet[0]:n_SQ+triplet[1]+1], axis=1)
            sum_samples = np.expand_dims(sum_samples,axis=1)
            samples[:,n_SQ+triplet[0]:n_SQ+triplet[1]+1] /= sum_samples 
            samples[:,n_SQ+triplet[0]:n_SQ+triplet[1]+1] *= triplet[2]

            sum_samples = np.sum(samples_erf[:,n_SQ+triplet[0]:n_SQ+triplet[1]+1], axis=1)
            sum_samples = np.expand_dims(sum_samples,axis=1)
            samples_erf[:,n_SQ+triplet[0]:n_SQ+triplet[1]+1] /= sum_samples 
            samples_erf[:,n_SQ+triplet[0]:n_SQ+triplet[1]+1] *= triplet[2]
                
    print("")
    print("Computing quantiles and mean")
    print(" j   quan05    quan50     qmean    quan95")

    for j in np.arange(n_SQ + n_TQ):

        if global_log[j]:

            qmean_EW = 10.0**np.mean(np.log10(samples_EW[:,j]))
    
        else:

            qmean_EW = np.mean(samples_EW[:,j])
    
        quan05_EW = np.quantile(samples_EW[:,j], 0.05)
        quan50_EW = np.quantile(samples_EW[:,j], 0.5)
        quan95_EW = np.quantile(samples_EW[:,j], 0.95)

        print(label_indexes[j]+ " %.2E %.2E %.2E %.2E" %
              (quan05_EW, quan50_EW, qmean_EW, quan95_EW))

        q_EW[j, 0] = quan05_EW
        q_EW[j, 1] = quan50_EW
        q_EW[j, 2] = quan95_EW
        q_EW[j, 3] = qmean_EW

        if Cooke_flag>0:

            if global_log[j]:

                qmean = 10.0**np.mean(np.log10(samples[:,j]))
    
            else:

                qmean = np.mean(samples[:,j])
    
            quan05 = np.quantile(samples[:,j], 0.05)
            quan50 = np.quantile(samples[:,j], 0.5)
            quan95 = np.quantile(samples[:,j], 0.95)

            print(label_indexes[j]+ " %.2E %.2E %.2E %.2E" %
                  (quan05, quan50, qmean, quan95))

            q_Cooke[j, 0] = quan05
            q_Cooke[j, 1] = quan50
            q_Cooke[j, 2] = quan95
            q_Cooke[j, 3] = qmean

        if ERF_flag > 0:

            if global_log[j]:

                qmean_erf = 10.0**np.mean(np.log10(samples_erf[:,j]))
    
            else:

                qmean_erf = np.mean(samples_erf[:,j])
    
            quan05_erf = np.quantile(samples_erf[:,j], 0.05)
            quan50_erf = np.quantile(samples_erf[:,j], 0.5)
            quan95_erf = np.quantile(samples_erf[:,j], 0.95)
                
            print(label_indexes[j]+ " %.2E %.2E %.2E %.2E" %
                  (quan05_erf, quan50_erf, qmean_erf, quan95_erf))

            q_erf[j, 0] = quan05_erf
            q_erf[j, 1] = quan50_erf
            q_erf[j, 2] = quan95_erf
            q_erf[j, 3] = qmean_erf

    return q_Cooke, q_erf, q_EW, samples, samples_erf, samples_EW


def main(argv):

    current_path = os.getcwd()

    if isinstance(argv, str):

        print(argv)
        repository = argv
        path = current_path + "/ELICITATIONS/" + repository

        # Check whether the specified path exists or not
        repoExists = os.path.exists(path)

    else:

        repoExists = False

    if repoExists:

        print("")
        print(repository + " found")
        os.chdir(path)

    else:

        elicitation_list = next(os.walk(current_path + "/ELICITATIONS/"))
        print("List of elicitations:")

        if len(elicitation_list[1]) == 1:

            userinput = 0
            cond = False

        else:

            for count, elicitation in enumerate(elicitation_list[1]):

                print(count, elicitation)

            print("")
            cond = True

        while cond:

            userinput = int(input("Enter Elicitation Number\n"))
            cond = not (userinput in range(len(elicitation_list[1])))
            if cond:
                print("Integer between 0 and ", len(elicitation_list[1]) - 1)

        repository = elicitation_list[1][userinput]
        print("repository", repository)
        path = current_path + "/ELICITATIONS/" + repository

    os.chdir(path)
    print("Path: ", path)
    sys.path.append(path)

    from ElicipyDict import output_dir
    from ElicipyDict import datarepo
    from ElicipyDict import target
    from ElicipyDict import elicitation_name
    from ElicipyDict import analysis
    from ElicipyDict import alpha
    from ElicipyDict import overshoot
    from ElicipyDict import cal_power
    from ElicipyDict import EW_flag
    from ElicipyDict import ERF_flag
    from ElicipyDict import Cooke_flag
    from ElicipyDict import n_sample
    from ElicipyDict import postprocessing
    from ElicipyDict import hist_type
    from ElicipyDict import n_bins
    
    if (Cooke_flag>0 or ERF_flag>0):

        seed = True

    else:

        seed = False
        
    try:

        from ElicipyDict import nolabel_flag

    except ImportError:

        nolabel_flag = False

    
    try:

        from ElicipyDict import group_list

    except ImportError:

        group_list = [0]

    if len(group_list) > 1 and not (0 in group_list):

        group_list.append(0)

    n_pctl = 3

    # download the data from github repository
    if datarepo == "github":

        from ElicipyDict import user
        from ElicipyDict import github_token
        from ElicipyDict import RepositoryData

        saveDataFromGithub(RepositoryData, user, github_token)

    sys.path.insert(0, os.getcwd())
    input_dir = "DATA"
    csv_file = "questionnaire.csv"

    # change to full path
    output_dir = path + "/" + output_dir
    input_dir = path + "/" + input_dir

    # Check whether the specified output path exists or not
    isExist = os.path.exists(output_dir)

    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(output_dir)
        print("The new directory " + output_dir + " is created!")

    # Read the questionnaire csv file
    (df_indexes_SQ, df_indexes_TQ, SQ_scale, SQ_realization, TQ_scale,
     SQ_minVals, SQ_maxVals, TQ_minVals, TQ_maxVals, SQ_units, TQ_units,
     SQ_LongQuestion, TQ_LongQuestion, SQ_question, TQ_question, idx_list,
     global_scale, global_log, label_indexes, parents, global_idxMin,
     global_idxMax, global_sum50,
     df_quest) = read_questionnaire(input_dir, csv_file, seed, target)

    try:

        from ElicipyDict import normalizeSum

    except ImportError:

        normalizeSum = False


    globalSum_temp = zip(global_idxMin,global_idxMax,global_sum50)
    
    globalSum_temp = list(set(globalSum_temp))
    
    globalSum = []
    for triplet in globalSum_temp:
    
        try:
        
            int(triplet[0])
            flag = True
            
        except ValueError:
        
            flag = False
            
        print(flag)    
        if flag:
        
            index_first = idx_list.index(int(triplet[0]))
            index_last = idx_list.index(int(triplet[1]))
            globalSum.append([index_first,index_last,triplet[2]])
 
    print('QUI',globalSum)

    # Read the asnwers of all the experts
    group = 0
    write_flag = True
    n_experts, n_SQ, n_TQ, SQ_array, TQ_array = read_answers(
        input_dir, csv_file, group, n_pctl, df_indexes_SQ, df_indexes_TQ,
        seed, target, output_dir, elicitation_name, write_flag, label_indexes)

    save_dtt_rll(input_dir, n_experts, n_SQ, n_TQ, df_quest, target,
                 SQ_realization, SQ_scale, SQ_array, TQ_scale, TQ_array)

    minval_all = np.zeros(n_SQ + n_TQ)
    minval_all[0:n_SQ] = np.amin(SQ_array[:, 0, :], axis=0)
    minval_all[n_SQ:] = np.amin(TQ_array[:, 0, :], axis=0)

    maxval_all = np.zeros(n_SQ + n_TQ)
    maxval_all[0:n_SQ] = np.amax(SQ_array[:, 2, :], axis=0)
    maxval_all[n_SQ:] = np.amax(TQ_array[:, 2, :], axis=0)

    global_units = SQ_units + TQ_units

    global_minVal = SQ_minVals + TQ_minVals
    global_maxVal = SQ_maxVals + TQ_maxVals
    global_units = SQ_units + TQ_units
    global_longQuestion = SQ_LongQuestion + TQ_LongQuestion
    global_shortQuestion = SQ_question + TQ_question

    print("")
    print("Answer ranges")

    print(df_indexes_SQ, df_indexes_TQ)

    for i in range(n_SQ + n_TQ):

        if global_log[i]:

            deltalogval_all = np.log10(maxval_all[i]) - np.log10(minval_all[i])

            minval_all[i] = 10**(np.log10(minval_all[i]) -
                                 0.1 * deltalogval_all)
            maxval_all[i] = 10**(np.log10(maxval_all[i]) +
                                 0.1 * deltalogval_all)

        else:

            deltaval_all = maxval_all[i] - minval_all[i]
            minval_all[i] = minval_all[i] - 0.1 * deltaval_all
            maxval_all[i] = maxval_all[i] + 0.1 * deltaval_all

            if global_units[i] == "%":

                minval_all[i] = np.maximum(minval_all[i], 0)
                maxval_all[i] = np.minimum(maxval_all[i], 100)

        minval_all[i] = np.maximum(minval_all[i], global_minVal[i])
        maxval_all[i] = np.minimum(maxval_all[i], global_maxVal[i])

        print(i, minval_all[i], maxval_all[i])

    print("")

    q_Cooke = np.zeros((len(global_scale), 4))
    q_erf = np.zeros((len(global_scale), 4))
    q_EW = np.zeros((len(global_scale), 4))

    if len(group_list) > 1:

        q_Cooke_groups = np.zeros((len(global_scale), 4, len(group_list)))
        q_erf_groups = np.zeros((len(global_scale), 4, len(group_list)))
        q_EW_groups = np.zeros((len(global_scale), 4, len(group_list)))

    for count, group in enumerate(group_list):

        # Read the asnwers of the experts
        write_flag = False

        n_experts, n_SQ, n_TQ, SQ_array, TQ_array = read_answers(
            input_dir, csv_file, group, n_pctl, df_indexes_SQ, df_indexes_TQ,
            seed, target, output_dir, elicitation_name, write_flag, label_indexes)

        if not target:

            # if we do not read the target questions, set empty array
            n_TQ = 0
            TQ_array = np.zeros((n_experts, n_pctl, n_TQ))

        if seed:
            
            realization = np.zeros(TQ_array.shape[2] + SQ_array.shape[2])
            realization[0:SQ_array.shape[2]] = SQ_realization

            print("")
            print("Realization")
            print(realization)
            print()

        else:

            realization = []

        if analysis:

            tree = {"IDX": idx_list, "SHORT_Q": TQ_question}
            df_tree = pd.DataFrame(data=tree)
            
            # ----------------------------------------- #
            # ------------ Compute weights ------------ #
            # ----------------------------------------- #
            
            if group == 0:

                alpha_analysis = alpha

            else:

                alpha_analysis = 0.0
                
            W, W_erf, Weqok, W_gt0, Werf_gt0, expin = answer_analysis(
                input_dir, csv_file, n_experts, n_SQ, n_TQ, SQ_array,
                TQ_array, realization, global_scale, global_log,
                alpha_analysis, overshoot, cal_power, ERF_flag, Cooke_flag,
                seed)
            
            # ----------------------------------------- #
            # ------ Create samples and bar plots ----- #
            # ----------------------------------------- #
            
            q_Cooke, q_erf, q_EW, samples, samples_erf, samples_EW = create_samples(group, 
                   n_experts, n_SQ, n_TQ, n_pctl, SQ_array, TQ_array, n_sample, W,
                   W_erf, Weqok, W_gt0, Werf_gt0, expin, global_log, 
                   global_minVal, global_maxVal, label_indexes,
                   ERF_flag, Cooke_flag, EW_flag, overshoot, globalSum, normalizeSum)
             
            print("Creating distribution plots")
                   
            create_barplot(group, n_SQ, n_TQ, n_sample, global_log, global_minVal, 
                   global_maxVal, global_units, TQ_units, label_indexes,
                   minval_all, maxval_all, ERF_flag,
                   Cooke_flag, EW_flag, hist_type, output_dir, elicitation_name,
                   n_bins, q_Cooke, q_erf, q_EW, samples, samples_erf, samples_EW)       

            if Cooke_flag > 0:

                if group == 0:

                    W, score, information, M = COOKEweights(SQ_array, TQ_array, realization,
                                                alpha, global_scale, overshoot,
                                                cal_power, Cooke_flag)

                    csv_name = output_dir + "/" + elicitation_name + "_score_information.csv"

                    d = {"index": range(1, n_experts + 1), "score": score,
                             "information": information,
                             "ans < 5%ile": M[:,0],
                             "5%ile < ans < 50%ile": M[:,1],
                             "50%ile < ans < 95%ile": M[:,2],
                             "ans > 95%ile": M[:,3],
                             }
                    df = pd.DataFrame(data=d)

                    df.to_csv(csv_name, index=False)
                
                SQ_array_DM = np.zeros((1,n_pctl,n_SQ))

                for iseed in range(n_SQ):

                    SQ_array_DM[0,0,iseed] = q_Cooke[iseed,0]
                    SQ_array_DM[0,1,iseed] = q_Cooke[iseed,1]
                    SQ_array_DM[0,2,iseed] = q_Cooke[iseed,2]
                
                TQ_array_DM = np.zeros((1,n_pctl,0))

                W_DM, score_DM, information_DM, M = COOKEweights(SQ_array_DM, TQ_array_DM, realization,
                                                alpha, global_scale, overshoot,
                                                cal_power, Cooke_flag)

                print('Information DM-Cooke', information_DM)
                print('Score DM-Cooke',score_DM)
                
            if len(group_list) > 1:

                q_Cooke_groups[:, :, count] = q_Cooke
                q_erf_groups[:, :, count] = q_erf
                q_EW_groups[:, :, count] = q_EW

    # ----------------------------------------- #
    # ---------- Save samples on csv ---------- #
    # ----------------------------------------- #

    if analysis:

        csv_name = output_dir + "/" + elicitation_name + "_weights.csv"

        Weqok_100 = [100.0 * elem for elem in Weqok]

        Weqok_formatted = ["%.2f" % elem for elem in Weqok_100]

        d = {"index": range(1, n_experts + 1), "Weq": Weqok_formatted}
        df = pd.DataFrame(data=d)

        if Cooke_flag > 0:

            df.insert(loc=2, column="WCooke", value=W_gt0)

        if ERF_flag > 0:

            df.insert(loc=2, column="WERF", value=Werf_gt0)

        df.to_csv(csv_name, index=False)

    if analysis and target:

        targets = ["target_" + str(i + 1).zfill(2) for i in range(n_TQ)]

        df_tree["EW_5"] = q_EW[n_SQ:, 0]
        df_tree["EW_50"] = q_EW[n_SQ:, 1]
        df_tree["EW_95"] = q_EW[n_SQ:, 2]
        df_tree["EW_MEAN"] = q_EW[n_SQ:, 3]

        if Cooke_flag > 0:

            df_tree["COOKE_5"] = q_Cooke[n_SQ:, 0]
            df_tree["COOKE_50"] = q_Cooke[n_SQ:, 1]
            df_tree["COOKE_95"] = q_Cooke[n_SQ:, 2]
            df_tree["COOKE_MEAN"] = q_Cooke[n_SQ:, 3]

            csv_name = output_dir + "/" + elicitation_name + "_samples.csv"
            np.savetxt(
                csv_name,
                samples[n_SQ:-1],
                header=",".join(targets),
                comments="",
                delimiter=",",
                fmt="%1.4e",
            )

        if ERF_flag > 0:

            df_tree["ERF_5"] = q_erf[n_SQ:, 0]
            df_tree["ERF_50"] = q_erf[n_SQ:, 1]
            df_tree["ERF_95"] = q_erf[n_SQ:, 2]
            df_tree["ERF_MEAN"] = q_erf[n_SQ:, 3]

            csv_name = output_dir + "/" + elicitation_name + "_samples_erf.csv"
            np.savetxt(
                csv_name,
                samples_erf[n_SQ:-1],
                header=",".join(targets),
                comments="",
                delimiter=",",
                fmt="%1.4e",
            )

        if EW_flag > 0:

            csv_name = output_dir + "/" + elicitation_name + "_samples_EW.csv"
            np.savetxt(
                csv_name,
                samples_EW[n_SQ:-1],
                header=",".join(targets),
                comments="",
                delimiter=",",
                fmt="%1.4e",
            )

        df_tree["PARENT"] = parents
        df_tree.to_csv("tree.csv", index=False)

    if not postprocessing:

        print("Analysis completed!")
        sys.exit()

    if analysis:

        # ------------------------------------------ #
        # --------- Create trend. figures ---------- #
        # ------------------------------------------ #

        try:

            from ElicipyDict import trend_groups

            print("trend_groups read", trend_groups)

        except ImportError:

            print("No trend group defined")
            trend_groups = []

        for count, trend_group in enumerate(trend_groups):
        
            print("Creating trend plots")


            create_figure_trend(
                count,
                trend_group,
                n_SQ,
                q_EW,
                q_Cooke,
                q_erf,
                global_units,
                Cooke_flag,
                ERF_flag,
                EW_flag,
                global_log,
                TQ_minVals,
                TQ_maxVals,
                output_dir,
                elicitation_name,
            )

        # ------------------------------------------ #
        # --------- Create violin figures ---------- #
        # ------------------------------------------ #

        try:

            from ElicipyDict import violin_groups

            print("violin_groups read", violin_groups)

        except ImportError:

            print("No violin group defined")
            violin_groups = []

        for count, violin_group in enumerate(violin_groups):
        
            print("Creating violin plots")

            create_figure_violin(
                count,
                violin_group,
                n_SQ,
                samples_EW,
                samples,
                samples_erf,
                q_EW,
                q_Cooke,
                q_erf,
                global_units,
                Cooke_flag,
                ERF_flag,
                EW_flag,
                global_log,
                TQ_minVals,
                TQ_maxVals,
                output_dir,
                elicitation_name,
            )

        # ------------------------------------------ #
        # ----------- Create pie figures ----------- #
        # ------------------------------------------ #

        try:

            from ElicipyDict import pie_groups

            print("pie_groups read", pie_groups)

        except ImportError:

            print("No pie group defined")
            pie_groups = []

        for count, pie_group in enumerate(pie_groups):
        
            print("Creating violin plots")

            create_figure_pie(count,pie_group, n_SQ, q_EW, q_Cooke, q_erf, Cooke_flag,
                              ERF_flag, EW_flag, output_dir, elicitation_name)


    # ----------------------------------------- #
    # --------- Create answ. figures ---------- #
    # ----------------------------------------- #

    n_panels = int(np.ceil(n_experts / max_len_plot))

    print("Creating answer plots")

    for h in np.arange(n_SQ + n_TQ):
 
        printProgressBar(h, n_SQ+n_TQ-1)
           
        for k in np.arange(n_panels):

            create_figure_answers(
                h,
                k,
                n_experts,
                max_len_plot,
                n_SQ,
                SQ_array,
                TQ_array,
                realization,
                analysis,
                Cooke_flag,
                ERF_flag,
                EW_flag,
                global_units,
                output_dir,
                q_Cooke,
                q_erf,
                q_EW,
                elicitation_name,
                global_log,
                label_indexes,
                nolabel_flag
            )

    print("End Creating answer plots")


    # ----------------------------------------- #
    # ------- Create .pptx presentation ------- #
    # ----------------------------------------- #

    prs = Presentation()
    prs.slide_width = Inches(16)
    prs.slide_height = Inches(9)
    left = Inches(2)
    top = Inches(1.5)

    # ------------- Title slide ----------------#
    lyt = prs.slide_layouts[0]  # choosing a slide layout
    slide = prs.slides.add_slide(lyt)  # adding a slide
    title = slide.shapes.title  # assigning a title
    subtitle = slide.placeholders[1]  # placeholder for subtitle
    title.text = "Expert elicitation - " + elicitation_name  # title

    title_para = slide.shapes.title.text_frame.paragraphs[0]
    title_para.font.name = "Helvetica"

    Current_Date_Formatted = datetime.datetime.today().strftime("%d-%b-%Y")

    subtitle.text = Current_Date_Formatted  # subtitle

    subtitle_para = slide.shapes.placeholders[1].text_frame.paragraphs[0]
    subtitle_para.font.name = "Helvetica"

    logofile = current_path + "/logo.png"

    slide.shapes.add_picture(logofile,
                             left + Inches(11.3),
                             top + Inches(5.4),
                             width=Inches(2.4))

    title_slide_layout = prs.slide_layouts[5]

    # ------------- Weights slide -------------#

    if analysis and seed:

        slide = prs.slides.add_slide(title_slide_layout)

        text_title = "Experts' weights"

        title_shape = slide.shapes.title
        title_shape.text = text_title
        title_shape.top = Inches(3.0)
        title_shape.width = Inches(15)
        title_shape.height = Inches(2)
        title_para = slide.shapes.title.text_frame.paragraphs[0]
        title_para.font.name = "Helvetica"
        title_para.font.size = Pt(54)
        add_date(slide)
        add_small_logo(slide, left, top, logofile)

        n_tables = int(np.ceil(len(W_gt0) / max_len_table))

        for i_table in range(n_tables):

            slide = prs.slides.add_slide(title_slide_layout)

            text_title = "Experts' weights"
            add_title(slide, text_title)

            # ---add table weights to slide---
            x, y, cx = Inches(2), Inches(1.7), Inches(8)

            fisrt_j = i_table * max_len_table
            last_j = np.minimum((i_table + 1) * max_len_table, len(W_gt0))

            if ERF_flag > 0:

                shape = slide.shapes.add_table(last_j - fisrt_j + 1, 3, x, y,
                                               cx,
                                               MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT)

            else:

                shape = slide.shapes.add_table(last_j - fisrt_j + 1, 2, x, y,
                                               cx,
                                               MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT)

            table = shape.table

            cell = table.cell(0, 0)
            cell.text = "Expert ID"

            cell = table.cell(0, 1)
            cell.text = "Cooke"

            if ERF_flag > 0:

                cell = table.cell(0, 2)
                cell.text = "ERF"

            for j in np.arange(fisrt_j, last_j):
                j_mod = np.remainder(j, max_len_table)
                cell = table.cell(j_mod + 1, 0)
                cell.text = "Exp" + str(expin[j])

                cell = table.cell(j_mod + 1, 1)

                if W_gt0[j] > 0.0:

                    cell.text = "%6.2f" % W_gt0[j]

                else:

                    cell.text = "Below threshold"

                if ERF_flag > 0:

                    cell = table.cell(j_mod + 1, 2)
                    cell.text = "%6.2f" % Werf_gt0[j]

            for cell in iter_cells(table):
                for paragraph in cell.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(12)

            if EW_flag:

                text_box = "Equal weight = " + f"{Weqok[0]*100:.2f}"
                add_text_box(slide, Inches(12), Inches(2), text_box, 18)

                text_box = ("For Cookes' method, below threshold weight does "
                            "not mean zero score. It simply means that this "
                            "expert's knowledge was already contributed by "
                            "other experts and adding this expert would not "
                            "change significantly the results.")
                add_text_box(slide, Inches(12), Inches(3), text_box, 18)

            add_date(slide)
            add_small_logo(slide, left, top, logofile)

    # ------------- Answers slides ------------#

    if seed:
        
        slide = prs.slides.add_slide(title_slide_layout)
    
        text_title = "Seed Answers"

        title_shape = slide.shapes.title
        title_shape.text = text_title
        title_shape.top = Inches(3.0)
        title_shape.width = Inches(15)
        title_shape.height = Inches(2)
        title_para = slide.shapes.title.text_frame.paragraphs[0]
        title_para.font.name = "Helvetica"
        title_para.font.size = Pt(54)
        add_date(slide)
        add_small_logo(slide, left, top, logofile)

    for h in np.arange(n_SQ + n_TQ):

        if h == n_SQ:

            slide = prs.slides.add_slide(title_slide_layout)

            text_title = "Target Answers"

            title_shape = slide.shapes.title
            title_shape.text = text_title
            title_shape.top = Inches(3.0)
            title_shape.width = Inches(15)
            title_shape.height = Inches(2)
            title_para = slide.shapes.title.text_frame.paragraphs[0]
            title_para.font.name = "Helvetica"
            title_para.font.size = Pt(54)
            add_date(slide)
            add_small_logo(slide, left, top, logofile)

        text_box = global_longQuestion[h]

        if h >= n_SQ:

            j = h - n_SQ
            string = "Target"

            if len(text_box) < 250:

                fontsize = 18

            else:

                fontsize = 18.0 * np.sqrt(250.0 / len(text_box))

        else:

            j = h
            string = "Seed"

            if len(text_box) < 500:

                fontsize = 18

            else:

                fontsize = 18.0 * np.sqrt(500.0 / len(text_box))

        for k in np.arange(n_panels):

            slide = prs.slides.add_slide(title_slide_layout)

            add_text_box(slide, left, top, text_box, fontsize)

            figname = (output_dir + "/" + elicitation_name + "_" + string +
                       "_" + str(j + 1).zfill(2) + "_" + str(k + 1).zfill(2) +
                       ".png")
            add_figure(slide, figname, left, top, Inches(10))

            if analysis and (string == "Target"):

                figname = (output_dir + "/" + elicitation_name +
                           "_cum_group0_" + str(j + 1).zfill(2) + ".png")

                width = Inches(2.5)
                add_small_figure(slide, figname, left + Inches(1.1),
                                 top + Inches(4.9), width)

                figname = (output_dir + "/" + elicitation_name +
                           "_PDF_group0_" + str(j + 1).zfill(2) + ".png")

                width = Inches(2.5)
                add_small_figure(slide, figname, left - Inches(1.2),
                                 top + Inches(4.9), width)

                if global_idxMin[h] != global_idxMax[h]:

                    longQ_NB = ("N.B. The sum of 50%iles for questions " +
                                str(global_idxMin[h]) + "-" +
                                str(global_idxMax[h]) + " have to sum to " +
                                str(global_sum50[h]) + ".")

                    add_text_box(slide, left, top + Inches(3.79), longQ_NB, 13)

            add_date(slide)
            add_small_logo(slide, left, top, logofile)

            text_title = global_shortQuestion[h]
            add_title(slide, text_title)

    # ------------- Pctls slides -------------#

    if analysis and target:

        slide = prs.slides.add_slide(title_slide_layout)

        text_title = "Target Percentiles"

        title_shape = slide.shapes.title
        title_shape.text = text_title
        title_shape.top = Inches(3.0)
        title_shape.width = Inches(15)
        title_shape.height = Inches(2)
        title_para = slide.shapes.title.text_frame.paragraphs[0]
        title_para.font.name = "Helvetica"
        title_para.font.size = Pt(54)
        add_date(slide)
        add_small_logo(slide, left, top, logofile)

        n_tables = int(np.ceil(n_TQ / max_len_tableB))

        for i_table in range(n_tables):

            slide = prs.slides.add_slide(prs.slide_layouts[5])

            fisrt_j = i_table * max_len_tableB
            last_j = np.minimum((i_table + 1) * max_len_tableB, n_TQ)

            n_columns = 1

            if Cooke_flag > 0:

                n_columns += 4

            if EW_flag > 0:

                n_columns += 4

            if ERF_flag > 0:

                n_columns += 4

            # ---add table to slide---
            x, y, cx = Inches(1), Inches(2), Inches(14)
            shape = slide.shapes.add_table(
                last_j - fisrt_j + 1,
                n_columns,
                x,
                y,
                cx,
                MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT,
            )
            table = shape.table

            cell = table.cell(0, 1)
            cell.text = "Q05 (EW)"

            cell = table.cell(0, 2)
            cell.text = "Q50 (EW)"

            cell = table.cell(0, 3)
            cell.text = "Q95 (EW)"

            cell = table.cell(0, 4)
            cell.text = "Qmean (EW)"

            j_column = 4

            if Cooke_flag > 0:

                cell = table.cell(0, j_column + 1)
                cell.text = "Q05 (Cooke)"

                cell = table.cell(0, j_column + 2)
                cell.text = "Q50 (Cooke)"

                cell = table.cell(0, j_column + 3)
                cell.text = "Q95 (Cooke)"

                cell = table.cell(0, j_column + 4)
                cell.text = "Qmean (Cooke)"
                j_column += 4

            if ERF_flag > 0:

                cell = table.cell(0, j_column + 1)
                cell.text = "Q05 (ERF)"

                cell = table.cell(0, j_column + 2)
                cell.text = "Q50 (ERF)"

                cell = table.cell(0, j_column + 3)
                cell.text = "Q95 (ERF)"

                cell = table.cell(0, j_column + 4)
                cell.text = "Qmean (ERF)"

            for h in np.arange(fisrt_j, last_j):

                h_mod = np.remainder(h, max_len_tableB)

                j = h + n_SQ

                cell = table.cell(h_mod + 1, 0)
                cell.text = "TQ " + label_indexes[j]

                for li in range(4):

                    cell = table.cell(h_mod + 1, li + 1)
                    if global_units[j] == "%" and global_log[j] == 0:
                        cell.text = "%6.2f" % q_EW[j, li]
                    else:
                        cell.text = "%.2E" % q_EW[j, li]

                    j_column = 4

                    if Cooke_flag > 0:

                        cell = table.cell(h_mod + 1, j_column + li + 1)
                        if global_units[j] == "%" and global_log[j] == 0:
                            cell.text = "%6.2f" % q_Cooke[j, li]
                        else:
                            cell.text = "%.2E" % q_Cooke[j, li]

                        j_column += 4

                    if ERF_flag > 0:

                        cell = table.cell(h_mod + 1, j_column + li + 1)
                        if global_units[j] == "%" and global_log[j] == 0:
                            cell.text = "%6.2f" % q_erf[j, li]
                        else:
                            cell.text = "%.2E" % q_erf[j, li]

            for cell in iter_cells(table):
                for paragraph in cell.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(14)

            add_date(slide)
            add_small_logo(slide, left, top, logofile)

            text_title = "Percentiles of target questions"
            add_title(slide, text_title)

    # ----------- Trend groups slides --------#

    if analysis and len(trend_groups) > 0:

        slide = prs.slides.add_slide(title_slide_layout)

        text_title = "Trend plots"

        title_shape = slide.shapes.title
        title_shape.text = text_title
        title_shape.top = Inches(3.0)
        title_shape.width = Inches(15)
        title_shape.height = Inches(2)
        title_para = slide.shapes.title.text_frame.paragraphs[0]
        title_para.font.name = "Helvetica"
        title_para.font.size = Pt(54)
        add_date(slide)
        add_small_logo(slide, left, top, logofile)

        for count, trend_group in enumerate(trend_groups):

            slide = prs.slides.add_slide(title_slide_layout)

            figname = (output_dir + "/" + elicitation_name + "_trend_" +
                       str(count + 1).zfill(2) + ".png")

            text_box = ""
            for i in trend_group:

                text_box = (text_box + "TQ" + str(i) + ". " +
                            TQ_question[i - 1] + ".\n\n")

            if len(text_box) < 500:

                fontsize = 18

            else:

                fontsize = 20.0 * np.sqrt(500.0 / len(text_box))

            add_text_box(slide, left, top, text_box, fontsize)

            add_date(slide)
            add_small_logo(slide, left, top, logofile)
            add_figure(slide, figname, left - Inches(0.8), top, Inches(10))

            text_title = "Target questions Group " + str(count + 1)
            add_title(slide, text_title)

    # ----------- Violin groups slides --------#

    if analysis and len(violin_groups) > 0:

        slide = prs.slides.add_slide(title_slide_layout)

        text_title = "Violin plots"

        title_shape = slide.shapes.title
        title_shape.text = text_title
        title_shape.top = Inches(3.0)
        title_shape.width = Inches(15)
        title_shape.height = Inches(2)
        title_para = slide.shapes.title.text_frame.paragraphs[0]
        title_para.font.name = "Helvetica"
        title_para.font.size = Pt(54)
        add_date(slide)
        add_small_logo(slide, left, top, logofile)

        for count, violin_group in enumerate(violin_groups):

            slide = prs.slides.add_slide(title_slide_layout)

            figname = (output_dir + "/" + elicitation_name + "_violin_" +
                       str(count + 1).zfill(2) + ".png")

            text_box = ""
            for i in violin_group:

                text_box = (text_box + "TQ" + str(i) + ". " +
                            TQ_question[i - 1] + ".\n\n")

            if len(text_box) < 500:

                fontsize = 18

            else:

                fontsize = 20.0 * np.sqrt(500.0 / len(text_box))

            add_text_box(slide, left, top, text_box, fontsize)

            add_date(slide)
            add_small_logo(slide, left, top, logofile)
            add_figure(slide, figname, left - Inches(0.34), top+ Inches(1.0), Inches(10.85))

            text_title = "Target questions Group " + str(count + 1)
            add_title(slide, text_title)

    # ----------- Pie groups slides --------#

    if analysis and len(pie_groups) > 0:

        slide = prs.slides.add_slide(title_slide_layout)

        text_title = "Pie charts"

        title_shape = slide.shapes.title
        title_shape.text = text_title
        title_shape.top = Inches(3.0)
        title_shape.width = Inches(15)
        title_shape.height = Inches(2)
        title_para = slide.shapes.title.text_frame.paragraphs[0]
        title_para.font.name = "Helvetica"
        title_para.font.size = Pt(54)
        add_date(slide)
        add_small_logo(slide, left, top, logofile)

        for count, pie_group in enumerate(pie_groups):

            slide = prs.slides.add_slide(title_slide_layout)

            figname = (output_dir + "/" + elicitation_name + "_pie_" +
                       str(count + 1).zfill(2) + ".png")

            text_box = ""
            for i in pie_group:

                text_box = (text_box + "TQ" + str(i) + ". " +
                            TQ_question[i - 1] + ".\n\n")

            if len(text_box) < 500:

                fontsize = 18

            else:

                fontsize = 20.0 * np.sqrt(500.0 / len(text_box))

            add_text_box(slide, left, top, text_box, fontsize)

            add_date(slide)
            add_small_logo(slide, left, top, logofile)
            add_figure(slide, figname, left - Inches(0.34), top+ Inches(1.0), Inches(10.85))

            text_title = "Target questions Group " + str(count + 1)
            add_title(slide, text_title)
            
    # ------------ Barplot slides ------------#

    for j in np.arange(n_SQ + n_TQ):

        if analysis:

            if j == n_SQ:

                slide = prs.slides.add_slide(title_slide_layout)

                text_title = "Target Barplots"

                title_shape = slide.shapes.title
                title_shape.text = text_title
                title_shape.top = Inches(3.0)
                title_shape.width = Inches(15)
                title_shape.height = Inches(2)
                title_para = slide.shapes.title.text_frame.paragraphs[0]
                title_para.font.name = "Helvetica"
                title_para.font.size = Pt(54)
                add_date(slide)
                add_small_logo(slide, left, top, logofile)

            if j >= n_SQ:

                slide = prs.slides.add_slide(title_slide_layout)

                figname = (output_dir + "/" + elicitation_name + "_hist_" +
                           str(j - n_SQ + 1).zfill(2) + ".png")

                text_box = TQ_LongQuestion[j - n_SQ]

                if len(text_box) < 400:

                    fontsize = 18

                else:

                    fontsize = 18.0 * np.sqrt(400.0 / len(text_box))

                add_text_box(slide, left, top, text_box, fontsize)

                add_date(slide)
                add_small_logo(slide, left, top, logofile)
                add_figure(slide, figname, left - Inches(0.8), top, Inches(10))

                text_title = TQ_question[j - n_SQ]
                add_title(slide, text_title)

                # ---add table to slide---

                n_rows = 1

                if Cooke_flag > 0:

                    n_rows += 1

                if EW_flag > 0:

                    n_rows += 1

                if ERF_flag > 0:

                    n_rows += 1
                x, y, cx = Inches(1), Inches(6.5), Inches(4.0)
                shape = slide.shapes.add_table(n_rows, 5, x, y, cx,
                                               MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT)
                table = shape.table

                cell = table.cell(0, 1)
                cell.text = "Q05"

                cell = table.cell(0, 2)
                cell.text = "Q50"

                cell = table.cell(0, 3)
                cell.text = "Q95"

                cell = table.cell(0, 4)
                cell.text = "Qmean"

                j_row = 0

                if Cooke_flag > 0:

                    cell = table.cell(j_row + 1, 0)
                    cell.text = "Cooke"

                    for li in range(4):

                        cell = table.cell(j_row + 1, li + 1)
                        if global_units[j] == "%" and global_log == 0:
                            cell.text = "%6.2f" % q_Cooke[j, li]
                        else:
                            cell.text = "%.2E" % q_Cooke[j, li]

                    j_row += 1

                if ERF_flag > 0:

                    cell = table.cell(j_row + 1, 0)
                    cell.text = "ERF"

                    for li in range(4):

                        cell = table.cell(j_row + 1, li + 1)
                        if global_units[j] == "%" and global_log == 0:
                            cell.text = "%6.2f" % q_erf[j, li]
                        else:
                            cell.text = "%.2E" % q_erf[j, li]

                    j_row += 1

                if EW_flag > 0:

                    cell = table.cell(j_row + 1, 0)
                    cell.text = "EW"

                    for li in range(4):

                        cell = table.cell(j_row + 1, li + 1)
                        if global_units[j] == "%" and global_log == 0:
                            cell.text = "%6.2f" % q_EW[j, li]
                        else:
                            cell.text = "%.2E" % q_EW[j, li]

                    j_row += 1

                for cell in iter_cells(table):
                    for paragraph in cell.text_frame.paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(10)

    # ------------ Group slides ------------#

    for j in np.arange(n_SQ + n_TQ):

        if analysis and len(group_list) > 1:

            if j == n_SQ:

                slide = prs.slides.add_slide(title_slide_layout)

                text_title = "Groups PDFs"

                title_shape = slide.shapes.title
                title_shape.text = text_title
                title_shape.top = Inches(3.0)
                title_shape.width = Inches(15)
                title_shape.height = Inches(2)
                title_para = slide.shapes.title.text_frame.paragraphs[0]
                title_para.font.name = "Helvetica"
                title_para.font.size = Pt(54)
                add_date(slide)
                add_small_logo(slide, left, top, logofile)

            if j >= n_SQ:

                slide = prs.slides.add_slide(title_slide_layout)

                for count, group in enumerate(group_list):

                    figname = (output_dir + "/" + elicitation_name +
                               "_PDF_group" + str(group) + "_" +
                               str(j - n_SQ + 1).zfill(2) + ".png")

                    if group == 0:

                        width = Inches(6.0)
                        add_small_figure(slide, figname, Inches(5.5),
                                         Inches(2.0), width)

                    else:

                        width = Inches(4.22)
                        add_small_figure(
                            slide,
                            figname,
                            Inches(11.18),
                            Inches(2.2) + count * Inches(3.02),
                            width,
                        )

                text_box = TQ_LongQuestion[j - n_SQ]

                if len(text_box) < 200:

                    fontsize = 18

                else:

                    fontsize = 18.0 * np.sqrt(200.0 / len(text_box))

                add_text_box(slide, left, top, text_box, fontsize)

                add_date(slide)
                add_small_logo(slide, left, top, logofile)

                text_title = TQ_question[j - n_SQ]
                add_title(slide, text_title)

                n_rows = 4

                if EW_flag > 0:

                    n_rows += 3

                if ERF_flag > 0:

                    n_rows += 3

                x, y, cx = Inches(1), Inches(5.5), Inches(4.0)
                shape = slide.shapes.add_table(n_rows, 5, x, y, cx,
                                               MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT)

                table = shape.table

                cell = table.cell(0, 1)
                cell.text = "Q05"

                cell = table.cell(0, 2)
                cell.text = "Q50"

                cell = table.cell(0, 3)
                cell.text = "Q95"

                cell = table.cell(0, 4)
                cell.text = "Qmean"

                j_row = 0

                if Cooke_flag > 0:

                    for count, group in enumerate(group_list):

                        cell = table.cell(j_row + 1 + count, 0)

                        if group == 0:
                            cell.text = "Cooke"

                        else:
                            cell.text = "Cooke " + str(group)

                        for li in range(4):

                            cell = table.cell(j_row + 1 + count, li + 1)

                            if global_units[j] == "%" and global_log == 0:
                                cell.text = "%6.3f" % q_Cooke_groups[j, li,
                                                                     count]
                            else:
                                cell.text = "%.2E" % q_Cooke_groups[j, li,
                                                                    count]

                    j_row += len(group_list)

                if ERF_flag > 0:

                    for count, group in enumerate(group_list):

                        cell = table.cell(j_row + 1 + count, 0)

                        if group == 0:
                            cell.text = "ERF"

                        else:
                            cell.text = "ERF " + str(group)

                        for li in range(4):

                            cell = table.cell(j_row + 1 + count, li + 1)

                            if global_units[j] == "%" and global_log == 0:
                                cell.text = "%6.3f" % q_erf_groups[j, li,
                                                                   count]
                            else:
                                cell.text = "%.2E" % q_erf_groups[j, li, count]

                    j_row += len(group_list)

                if EW_flag > 0:

                    for count, group in enumerate(group_list):

                        cell = table.cell(j_row + 1 + count, 0)

                        if group == 0:
                            cell.text = "EW"

                        else:
                            cell.text = "EW " + str(group)

                        for li in range(4):

                            cell = table.cell(j_row + 1 + count, li + 1)

                            if global_units[j] == "%" and global_log == 0:
                                cell.text = "%6.2f" % q_EW_groups[j, li, count]
                            else:
                                cell.text = "%.2E" % q_EW_groups[j, li, count]

                    j_row += len(group_list)

                for cell in iter_cells(table):
                    for paragraph in cell.text_frame.paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(10)

    prs.save(output_dir + "/" + elicitation_name + ".pptx")  # saving file
    save_dtt_rll(input_dir, n_experts, n_SQ, n_TQ, df_quest, target,
                 SQ_realization, SQ_scale, SQ_array, TQ_scale, TQ_array)


if __name__ == "__main__":
    main(sys.argv[1:])
