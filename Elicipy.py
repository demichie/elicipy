import datetime
import copy
import numpy as np
import os
import sys
from pdf2image import convert_from_path
from pathlib import Path
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import PercentFormatter

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
from ElicipyDict import *

max_len_table = 21
max_len_tableB = 18

max_len_plot = 21

plt.rcParams.update({'font.size': 8})

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica']

matplotlib.use("TkAgg")


def create_fig_hist(group, j, n_sample, n_SQ, hist_type, C, C_erf, C_EW,
                    colors, legends, legendsPDF, global_units, Cooke_flag,
                    ERF_flag, EW_flag, global_minVal, global_maxVal,
                    output_dir, elicitation_name, del_rows, TQ_units,
                    label_indexes, minval_all, maxval_all):

    from scipy import stats

    fig = plt.figure()
    axs_h = fig.add_subplot(111)
    C_stack = np.stack((C, C_erf, C_EW), axis=0)
    C_stack = np.delete(C_stack, del_rows, 0)
    wg = np.ones_like(C_stack.T) / n_sample

    if hist_type == 'step':

        axs_h.hist(C_stack.T,
                   bins=n_bins,
                   weights=wg,
                   histtype='step',
                   fill=False,
                   rwidth=0.95,
                   color=colors)

    elif hist_type == 'bar':

        axs_h.hist(C_stack.T,
                   bins=n_bins,
                   weights=wg,
                   histtype='bar',
                   rwidth=0.95,
                   ec="k",
                   color=colors)

    axs_h.set_xlabel(TQ_units[j - n_SQ])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    xt = plt.xticks()[0]

    axs_h2 = axs_h.twinx()

    if global_units[j] == "%":

        xmin = 0.0
        xmax = 100.0

    else:

        xmin = np.amin(C_stack)
        xmax = np.amax(C_stack)

    lnspc = np.linspace(xmin, xmax, 1000)

    if (Cooke_flag > 0):
        gkde = stats.gaussian_kde(C)
        gkde_norm = gkde.integrate_box_1d(global_minVal[j], global_maxVal[j])
        kdepdf = gkde.evaluate(lnspc) / gkde_norm
        axs_h2.plot(lnspc, kdepdf, 'r--')

    if (ERF_flag > 0):
        gkde_erf = stats.gaussian_kde(C_erf)
        gkde_erf_norm = gkde_erf.integrate_box_1d(global_minVal[j],
                                                  global_maxVal[j])
        kdepdf_erf = gkde_erf.evaluate(lnspc) / gkde_erf_norm
        axs_h2.plot(lnspc, kdepdf_erf, '--', color='tab:purple')

    if (EW_flag > 0):
        gkde_EW = stats.gaussian_kde(C_EW)
        gkde_EW_norm = gkde_EW.integrate_box_1d(global_minVal[j],
                                                global_maxVal[j])
        kdepdf_EW = gkde_EW.evaluate(lnspc) / gkde_EW_norm
        axs_h2.plot(lnspc, kdepdf_EW, 'g--')

    axs_h.set_xlim(xmin, xmax)
    axs_h2.set_xlim(xmin, xmax)

    axs_h2.set_ylabel('PDF', color='b')

    axs_h2.set_ylim(bottom=0)
    plt.legend(legends)
    plt.title('Target Question ' + str(label_indexes[j]))

    figname = output_dir + '/' + elicitation_name + \
        '_hist_' + str(j - n_SQ + 1).zfill(2) + '.pdf'
    fig.savefig(figname)

    images = convert_from_path(figname)
    figname = output_dir + '/' + elicitation_name + \
        '_hist_' + str(j - n_SQ + 1).zfill(2) + '.png'
    images[0].save(figname, 'PNG')

    plt.close()

    # create figure with PDFs only for small inset and groups
    fig2 = plt.figure()
    axs_h2 = fig2.add_subplot(111)

    if (Cooke_flag > 0):

        axs_h2.plot(lnspc, kdepdf, 'r--', linewidth=2)

    if (ERF_flag > 0):

        axs_h2.plot(lnspc, kdepdf_erf, '--', color='tab:purple', linewidth=2)

    if (EW_flag > 0):

        axs_h2.plot(lnspc, kdepdf_EW, 'g--', linewidth=2)

    if global_units[j] == "%":

        axs_h2.set_xlim(xmin, xmax)

    else:

        axs_h2.set_xlim(minval_all[j], maxval_all[j])

    axs_h2.set_ylabel('PDF', color='b')

    axs_h2.set_ylim(bottom=0)
    leg = plt.legend(legends, prop={"size": 18})

    if group == 0:

        plt.title('Target Question ' + str(label_indexes[j]), fontsize=18)

    else:

        plt.title('Target Question ' + str(label_indexes[j]) + ' Group ' +
                  str(group),
                  fontsize=18)

    figname = output_dir + '/' + elicitation_name + \
        '_PDF_' + str(j - n_SQ + 1).zfill(2) + '.pdf'
    fig2.savefig(figname)

    images = convert_from_path(figname)
    figname = output_dir + '/' + elicitation_name + \
        '_PDF_group'+str(group) + '_' + str(j - n_SQ + 1).zfill(2) + '.png'
    images[0].save(figname, 'PNG')

    plt.close()

def create_figure_trend(count,trend_group,n_SQ,q_EW,q_Cooke,q_erf,global_units,
                        Cooke_flag, ERF_flag, EW_flag, global_log, 
                        TQ_minVals, TQ_maxVals, 
                        output_dir):
                        
    fig = plt.figure()
    axs = fig.add_subplot(111)
    
    x = np.arange(len(trend_group))+1

    handles = []
    
    trend_group = np.array(trend_group)
    
    # print(trend_group+n_SQ-1)
    # print(q_EW[trend_group+n_SQ-1,0])
    # print(q_EW[trend_group+n_SQ-1,1])
    # print(q_EW[trend_group+n_SQ-1,2])

    if EW_flag:

        y = q_EW[trend_group+n_SQ-1,1]
        lower_error = q_EW[trend_group+n_SQ-1,1] - q_EW[trend_group+n_SQ-1,0]
        upper_error = q_EW[trend_group+n_SQ-1,2] - q_EW[trend_group+n_SQ-1,1]
        asymmetric_error = [lower_error, upper_error]
    
        line1 = axs.errorbar(x-0.1, y, yerr=asymmetric_error, fmt='g-o',label='EW')
        axs.plot(x-0.1, y - lower_error, 'gx')
        axs.plot(x-0.1, y + upper_error, 'gx')
        handles.append(line1)

    if Cooke_flag > 0:

        y = q_Cooke[trend_group+n_SQ-1,1]
        lower_error = q_Cooke[trend_group+n_SQ-1,1] - q_Cooke[trend_group+n_SQ-1,0]
        upper_error = q_Cooke[trend_group+n_SQ-1,2] - q_Cooke[trend_group+n_SQ-1,1]
        asymmetric_error = [lower_error, upper_error]
    
        line2 = axs.errorbar(x, y, yerr=asymmetric_error, fmt='r-o',label='CM')
        axs.plot(x, y - lower_error, 'rx')
        axs.plot(x, y + upper_error, 'rx')
        handles.append(line2)

    if ERF_flag > 0:
    
        y = q_erf[trend_group+n_SQ-1,1]
        lower_error = q_erf[trend_group+n_SQ-1,1] - q_erf[trend_group+n_SQ-1,0]
        upper_error = q_erf[trend_group+n_SQ-1,2] - q_erf[trend_group+n_SQ-1,1]
        asymmetric_error = [lower_error, upper_error]
    
        line3 = axs.errorbar(x+0.1, y, yerr=asymmetric_error, fmt='b-o', label = 'ERF')
        axs.plot(x+0.1, y - lower_error, 'bx')
        axs.plot(x+0.1, y + upper_error, 'bx')
        handles.append(line3)

    axs.set_xticks(x)

    xtick = []
    for i in x:
        xtick.append('TQ' + str(trend_group[i-1]))

    axs.set_xticklabels(xtick)

    # ax1.set_yscale('log')

    axs.set_ylim(TQ_minVals[trend_group[0]], TQ_maxVals[trend_group[0]])

    # axs.grid(linewidth=0.4)

    plt.title('Question Group ' + str(count+1))

    axs.legend(handles=handles)
    
    figname = output_dir + '/' + elicitation_name + '_trend_'+ \
              str(count + 1).zfill(2)+'.pdf'
    fig.savefig(figname)

    images = convert_from_path(figname)
    figname = output_dir + '/' + elicitation_name + '_trend_'+ \
              str(count + 1).zfill(2)+'.png'
    images[0].save(figname, 'PNG')
    plt.close()

    return


def create_figure(h, k, n_experts, max_len_plot, n_SQ, SQ_array, TQ_array,
                  realization, analysis, Cooke_flag, ERF_flag, EW_flag,
                  global_units, output_dir, q_Cooke, q_erf, q_EW,
                  elicitation_name, global_log, label_indexes):

    idx0 = k * max_len_plot
    idx1 = min((k + 1) * max_len_plot, n_experts)

    if (h >= n_SQ):

        j = h - n_SQ
        Q_array = TQ_array[idx0:idx1, :, j]
        string = 'Target'

        xmin = np.amin(TQ_array[:, 0, j])
        xmax = np.amax(TQ_array[:, 2, j])

    else:

        j = h
        Q_array = SQ_array[idx0:idx1, :, j]
        string = 'Seed'

        xmin = np.amin(SQ_array[:, 0, j])
        xmax = np.amax(SQ_array[:, 2, j])

    if (global_log[h] == 1):

        log_xmin = np.log(xmin)
        log_xmax = np.log(xmax)

        delta_logx = 0.05 * (log_xmax - log_xmin)
        log_xmin -= delta_logx
        log_xmax += delta_logx

        xmin = np.exp(log_xmin)
        xmax = np.exp(log_xmax)

    else:

        deltax = 0.05 * (xmax - xmin)
        xmin -= deltax
        xmax += deltax

    x = Q_array[:, 1]
    y = np.arange(idx1 - idx0) + 1

    # creating error
    x_errormax = Q_array[:, 2] - Q_array[:, 1]
    x_errormin = Q_array[:, 1] - Q_array[:, 0]

    x_error = [x_errormin, x_errormax]

    fig = plt.figure()
    axs = fig.add_subplot(111)
    axs.errorbar(x, y, xerr=x_error, fmt='bo')
    axs.plot(x - x_errormin, y, 'bx')
    axs.plot(x + x_errormax, y, 'bx')

    if (realization[j] > 999):
        txt = '%5.2e' % realization[h]
    else:
        txt = '%6.2f' % realization[h]

    ytick = []
    for i in y:
        ytick.append('Exp.' + str(int(i + idx0)))

    yerror = idx1 - idx0
    if analysis:

        if Cooke_flag > 0:

            yerror = yerror + 1
            axs.errorbar(q_Cooke[h, 1],
                         yerror,
                         xerr=[[q_Cooke[h, 1] - q_Cooke[h, 0]],
                               [q_Cooke[h, 2] - q_Cooke[h, 1]]],
                         fmt='ro')
            axs.plot(q_Cooke[h, 0], yerror, 'rx')
            axs.plot(q_Cooke[h, 2], yerror, 'rx')

            ytick.append('DM-Cooke')

        if ERF_flag > 0:

            yerror = yerror + 1
            axs.errorbar(q_erf[h, 1], [yerror],
                         xerr=[[q_erf[h, 1] - q_erf[h, 0]],
                               [q_erf[h, 2] - q_erf[h, 1]]],
                         fmt='o',
                         color='tab:purple')
            axs.plot(q_erf[h, 0], yerror, 'x', color='tab:purple')
            axs.plot(q_erf[h, 2], yerror, 'x', color='tab:purple')

            ytick.append('DM-ERF')

        if EW_flag > 0:

            yerror = yerror + 1
            axs.errorbar([q_EW[h, 1]], [yerror],
                         xerr=[[q_EW[h, 1] - q_EW[h, 0]],
                               [q_EW[h, 2] - q_EW[h, 1]]],
                         fmt='go')

            axs.plot(q_EW[h, 0], yerror, 'gx')
            axs.plot(q_EW[h, 2], yerror, 'gx')

            ytick.append('DM-Equal')

        if (h < n_SQ):

            yerror = yerror + 1
            axs.plot(realization[h], yerror, 'kx')
            axs.annotate(txt, (realization[h] * 1.02, yerror + 0.15))

            ytick.append('Realization')

    else:

        if (h < n_SQ):

            axs.plot(realization[h], idx1 - idx0 + 1, 'kx')
            axs.annotate(txt, (realization[j] * 1.02, yerror + 0.15))
            ytick.append('Realization')

    y = np.arange(len(ytick)) + 1

    ytick_tuple = tuple(i for i in ytick)
    axs.set_yticks(y)

    axs.set_yticklabels(ytick_tuple)
    axs.set_xlabel(global_units[h])

    if (global_log[h] == 1):

        axs.set_xscale('log')

    axs.set_ylim(0.5, len(ytick) + 1.0)
    axs.set_xlim(xmin, xmax)

    axs.grid(linewidth=0.4)

    plt.title(string + ' Question ' + str(label_indexes[h]))
    figname = output_dir + '/' + elicitation_name + \
        '_'+string+'_' + str(j + 1).zfill(2) + \
        '_' + str(k + 1).zfill(2) + '.pdf'
    fig.savefig(figname)

    images = convert_from_path(figname)
    figname = output_dir + '/' + elicitation_name + \
        '_'+string+'_' + str(j + 1).zfill(2) + \
        '_' + str(k + 1).zfill(2) + '.png'
    images[0].save(figname, 'PNG')
    plt.close()


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
        datetime.datetime.today().strftime('%d-%b-%Y')
    shape_para = shape.text_frame.paragraphs[0]
    shape_para.font.name = "Helvetica"
    shape_para.font.size = Pt(17)


def add_small_logo(slide, left, top):

    img = slide.shapes.add_picture('../logo.png',
                                   left + Inches(13.0),
                                   top + Inches(6.8),
                                   width=Inches(0.8))


def add_figure(slide, figname, left, top):

    img = slide.shapes.add_picture(figname,
                                   left + Inches(3.4),
                                   top,
                                   width=Inches(10))

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


def add_text_box(slide, left, top, text_box,font_size):

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
                 df_indexes_TQ):

    import difflib

    # merge the files of the different experts
    # creating one file for seeds and one for tagets
    merge_csv(input_dir, target, group)

    # seeds file name
    filename = input_dir + '/seed.csv'

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
    # this is needed to search for expert matches between seed and target questions
    NS_SQ = []

    for name, surname in zip(firstname, surname):

        NS_SQ.append(name + surname)

    print('NS_SQ', NS_SQ)

    csv_name = output_dir + '/' + elicitation_name + '_experts.csv'

    d = {'index': range(1, len(NS_SQ) + 1), 'Expert': NS_SQ}

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

        print('')
        print('Seed question ', i)
        print(SQ_array[:, :, i])

    if target:

        filename = input_dir + '/target.csv'

        # Read a comma-separated values (csv) file into DataFrame df_TQ
        df_TQ = pd.read_csv(filename)

        # The second column (index 1) must contain the first name of the expert
        firstname = df_TQ[df_TQ.columns[1]].astype(str).tolist()

        # The third column (index 2) must contain the first name of the expert
        surname = df_TQ[df_TQ.columns[2]].astype(str).tolist()

        # create a list with firstname+surname
        # this is needed to search for expert matches between seed and target questions
        NS_TQ = []

        for name, surname in zip(firstname, surname):

            NS_TQ.append(name + surname)

        sorted_idx = []

        # loop to search for matches between experts in seed and target
        for SQ_name in NS_SQ:

            index = NS_TQ.index(difflib.get_close_matches(SQ_name, NS_TQ)[0])
            sorted_idx.append(index)

        print('Sorted list of experts to match the order of seeds:',
              sorted_idx)

        print(NS_SQ)
        print([NS_TQ[s_idx] for s_idx in sorted_idx])

        # create a 2D numpy array with the answers to the target questions
        cols_as_np = df_TQ[df_TQ.columns[4:]].to_numpy()

        # sort for expert names
        cols_as_np = cols_as_np[sorted_idx, :]

        # we want to work with a 3D array, with the following dimension:
        # n_expert X n_pctl X n_TQ
        n_experts_TQ = cols_as_np.shape[0]

        # check if number of experts in seed and target is the same
        if (n_experts_TQ != n_experts):

            print('Error: number of experts in seeds and targets different')
            sys.exit()

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

            print('Target question ', i)
            print(TQ_array[:, :, i])

    # if we have a subset of the SQ, then extract from SQ_array
    # the correct slice
    if len(df_indexes_SQ) > 0:

        # print('SQ_array',SQ_array)
        SQ_array = SQ_array[:, :, df_indexes_SQ]
        n_SQ = len(df_indexes_SQ)
        # print('SQ_array',SQ_array)

    if target:

        if len(df_indexes_TQ) > 0:

            # print('TQ_array',TQ_array)
            TQ_array = TQ_array[:, :, df_indexes_TQ]
            n_TQ = len(df_indexes_TQ)
            # print('TQ_array',TQ_array)
            
    else:
        n_TQ = 0
        TQ_array = np.zeros((n_experts,n_pctl,n_TQ))   

    return n_experts, n_SQ, n_TQ, SQ_array, TQ_array

    ########## READ csv question file ############


def read_questionnaire(input_dir, csv_file):

    df_read = pd.read_csv(input_dir + '/' + csv_file, header=0)
    # print(df_read)

    quest_type = df_read['QUEST_TYPE'].to_list()
    n_SQ = quest_type.count('seed')
    n_TQ = quest_type.count('target')

    print(quest_type)

    try:

        from ElicipyDict import seed_list
        print('seed_list read', seed_list)

    except ImportError:

        print('ImportError')
        seed_list = list(df_read['IDX'])

    print('seed_list', seed_list)

    if len(seed_list) > 0:
    
        # extract the seed questions with index in seed_list (from column IDX)
        df_SQ = df_read[df_read['IDX'].isin(seed_list)
                        & df_read.QUEST_TYPE.str.contains('seed')]
        print('Seed dataframe')
        print(df_SQ)

        # find the python indexes as rows of the dataframe
        df_indexes = np.asarray(
            np.where(df_read['IDX'].isin(seed_list)
                     & df_read.QUEST_TYPE.str.contains('seed')))

    else:

        df_SQ = df_read[df_read['QUEST_TYPE'] == 'seed']
        df_indexes = np.arange(len(df_quest.index))

    # find the indexes of the seed questions (0<idx<n_SQ)
    df_indexes_SQ = df_indexes[(df_indexes < n_SQ)]
    print('df_indexes_SQ', df_indexes_SQ)

    if target:

        try:

            from ElicipyDict import target_list
            print('target_list read', target_list)

        except ImportError:

            print('ImportError')
            target_list = list(df_read['IDX'])

        print('target_list', target_list)

        if len(target_list) > 0:

            # extract the target questions with index in target_list (from column IDX)
            df_TQ = df_read[df_read['IDX'].isin(target_list)
                            & df_read.QUEST_TYPE.str.contains('target')]
            print('Target dataframe')
            print(df_TQ)

            # find the python indexes as rows of the dataframe
            df_indexes = np.append(
                df_indexes,
                np.asarray(
                    np.where(df_read['IDX'].isin(target_list)
                             & df_read.QUEST_TYPE.str.contains('target'))))
            print('df_indexes', df_indexes)

        else:

            df_TQ = df_read[df_read['QUEST_TYPE'] == 'target']

            df_indexes = np.arange(len(df_quest.index))

        # find the indexes of the target questions (0<idx<n_TQ) in the extracted dataframe
        df_indexes_TQ = df_indexes[(df_indexes >= n_SQ)] - n_SQ
        print('df_indexes_TQ', df_indexes_TQ)

        df_quest = df_SQ.append(df_TQ)
        print('df_quest')
        print(df_quest)

    else:

        df_quest = df_SQ

    label_indexes = np.asarray(df_quest['IDX'])
    print('label_indexes', label_indexes)

    data_top = df_quest.head()

    langs = []

    # check if there are multiple languages
    for head in data_top:

        if 'LONG Q' in head:

            string = head.replace('LONG Q', '')
            string2 = string.replace('_', '')

            langs.append(string2)

    print('Languages:', langs)

    try:

        from ElicipyDict import language

    except ImportError:

        language = ''

    # select the columns to use according with the language
    if (len(langs) > 1):

        if language in langs:

            lang_index = langs.index(language)
            # list of column indexes to use
            index_list = [1, 2, lang_index+3] + \
                list(range(len(langs)+3, len(langs)+14))

        else:

            raise Exception("Sorry, language is not in questionnaire")

    else:

        lang_index = 0
        language = ''
        index_list = list(range(1, 15))

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

        idx, shortQ, longQ, unit, scale, minVal, maxVal, realization, question,\
            idxMin, idxMax, sum50, parent, image = [i[j] for j in index_list]

        minVal = float(minVal)
        maxVal = float(maxVal)

        if scale == 'uni':

            global_log.append(0)

        else:

            global_log.append(1)

        if (question == 'seed'):

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
            
            global_idxMin.append(idxMin)
            global_idxMax.append(idxMax)
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

    df_indexes_TQ = []
    
    idx_list = []
    parents = []

    if target:

        for i in df_quest.itertuples():

            idx, shortQ, longQ, unit, scale, minVal, maxVal, realization, \
                question, idxMin, idxMax, sum50, parent, image = \
                [i[j] for j in index_list]

            minVal = float(minVal)
            maxVal = float(maxVal)
            if (question == 'target'):

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

                global_idxMin.append(idxMin)
                global_idxMax.append(idxMax)
                global_sum50.append(sum50)


        # print on screen the units
        print("Target units = ", TQ_units)

        # print on screen the units
        print("Target scales = ", TQ_scale)

        global_scale = SQ_scale + TQ_scale

    else:

        global_scale = SQ_scale

    return df_indexes_SQ, df_indexes_TQ, SQ_scale, SQ_realization, TQ_scale, \
        SQ_minVals, SQ_maxVals, TQ_minVals, TQ_maxVals, SQ_units, TQ_units, \
        SQ_LongQuestion, TQ_LongQuestion, SQ_question, TQ_question, \
        idx_list, global_scale, global_log, label_indexes, parents, \
        global_idxMin, global_idxMax, global_sum50



def answer_analysis(input_dir, csv_file, n_experts, n_SQ, n_TQ, SQ_array, TQ_array,
             realization, global_scale, global_log, alpha):

    W, score, information = COOKEweights(SQ_array, TQ_array, realization,
                                           alpha, global_scale, overshoot,
                                           cal_power)
        
    W_erf, score_erf = ERFweights(realization, SQ_array)
         
    sensitivity = True                                       

    if sensitivity:
        
        for i in range(n_SQ):
            
            SQ_temp = np.delete(SQ_array, i, axis=2)
            realization_temp = np.delete(realization, i)
            global_scale_temp = np.delete(global_scale,i)
                
        
            W_temp,score_temp,information_temp = COOKEweights(SQ_temp, TQ_array, realization_temp,
                                           alpha, global_scale_temp, overshoot,
                                           cal_power)  
                 
            W_reldiff = W[:,3]/np.sum(W[:,3]) - W_temp[:,3]/np.sum(W_temp[:,3])    
                
            W_mean = np.mean( np.abs( W_reldiff) )
            W_std = np.sqrt( np.sum( np.square(W_reldiff) ) / n_experts )                           
                
            print(i+1,W_mean,W_std,np.sum(W_temp[:,4]>0))                               
                                                                                   
            W_erf_temp, score_erf_temp = ERFweights(realization_temp, SQ_temp)

            W_reldiff = W_erf[:,4] - W_erf_temp[:,4]    
                
            W_mean = np.mean( np.abs( W_reldiff) )
            W_std = np.sqrt( np.sum( np.square(W_reldiff) ) / n_experts )                           
                
            print(i+1,W_mean,W_std,np.sum(W_erf_temp>alpha))                               


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

    W_gt0 = [round((x * 100.0), 2) for x in W_gt0_01]
    Werf_gt0 = [round((x * 100.0), 2) for x in Werf_gt0_01]

    if ERF_flag > 0:

        print("")
        print('W_erf')
        print(W_erf[:, -1])

    if Cooke_flag:

        print("")
        print('W_cooke')
        print(W[:, -1])
        print('score', score)
        print('information', information)
        print('unNormalized weight', W[:, 3])

    print("")
    print('Weq')
    print(Weqok)

    return W, W_erf, Weqok, W_gt0, Werf_gt0, expin


def save_dtt_rll(input_dir):

    # ----------------------------------------- #
    # ------------ Save dtt and rls ----------- #
    # ----------------------------------------- #

    original_stdout = sys.stdout  # Save a reference to the original standard output

    filename = input_dir + '/seed.dtt'

    with open(filename, 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.

        print('* CLASS ASCII OUTPUT FILE. NQ=   3   QU=   5  50  95')
        print('')
        print('')

        for k in np.arange(n_experts):

            for i in np.arange(n_SQ):

                print(
                    f'{k+1:5d} {"Exp"+str(k+1):>8} {i+1:4d} {"SQ"+str(i+1):>13} {SQ_scale[i]:4} {SQ_array[k, 0, i]:6e} {""} {SQ_array[k, 1, i]:6e} {" "}{SQ_array[k, 2, i]:6e}'
                )

        sys.stdout = original_stdout  # Reset the standard output to its original value

    filename = input_dir + '/seed.rls'

    with open(filename, 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.

        for i in np.arange(n_SQ):

            # print(i+1,str(i+1),SQ_realization[i],SQ_scale[i])

            print(
                f'{i+1:>5d} {"SQ"+str(i+1):>13} {""} {SQ_realization[i]:6e} {SQ_scale[i]:4}'
            )

        sys.stdout = original_stdout  # Reset the standard output to its original value

    if target:

        filename = input_dir + '/target.dtt'

        with open(filename, 'w') as f:
            # Change the standard output to the file we created.
            sys.stdout = f

            print('* CLASS ASCII OUTPUT FILE. NQ=   3   QU=   5  50  95')
            print('')
            print('')

            for k in np.arange(n_experts):

                for i in np.arange(n_TQ):

                    print(
                        f'{k+1:>4d} {"Exp"+str(k+1):>10} {i+1:>4d} {"TQ"+str(i+1):>15} {TQ_scale[i]:>4} {TQ_array[k, 0, i]:>6e} {TQ_array[k, 1, i]:>6e} {TQ_array[k, 2, i]:>6e}'
                    )

            sys.stdout = original_stdout  # Reset the standard output to its original value

    return


def create_samples_and_barplot(group, n_experts, n_SQ, n_TQ, n_pctl, SQ_array,
                               TQ_array, n_sample, W, W_erf, Weqok, W_gt0,
                               Werf_gt0, expin, global_log, global_minVal,
                               global_maxVal, global_units, TQ_units,
                               label_indexes, minval_all, maxval_all,
                               postprocessing):

    DAT = np.zeros((n_experts * (n_SQ + n_TQ), n_pctl + 2))

    DAT[:, 0] = np.repeat(np.arange(1, n_experts + 1), n_SQ + n_TQ)
    DAT[:, 1] = np.tile(np.arange(1, n_SQ + n_TQ + 1), n_experts)

    DAT[:, 2:] = np.append(SQ_array, TQ_array,
                           axis=2).transpose(0, 2, 1).reshape(-1, 3)
    q_Cooke = np.zeros((n_SQ + n_TQ, 3))
    q_erf = np.zeros((n_SQ + n_TQ, 3))
    q_EW = np.zeros((n_SQ + n_TQ, 3))

    samples = np.zeros((n_sample, n_TQ))
    samples_erf = np.zeros((n_sample, n_TQ))
    samples_EW = np.zeros((n_sample, n_TQ))

    print("")
    if analysis:
        print(" j   quan05    quan50     qmean    quan95")

    del_rows = []
    keep_rows = []

    if (Cooke_flag == 0):
        del_rows.append(int(0))
    else:
        keep_rows.append(int(0))
    if (ERF_flag == 0):
        del_rows.append(int(1))
    else:
        keep_rows.append(int(1))
    if (EW_flag == 0):
        del_rows.append(int(2))
    else:
        keep_rows.append(int(2))

    colors = ['tomato', 'purple', 'springgreen']
    colors = [colors[index] for index in keep_rows]

    legends = ['CM', 'ERF', 'EW']
    legends = [legends[index] for index in keep_rows]

    for j in np.arange(n_SQ + n_TQ):

        if analysis:

            quan05_EW, quan50_EW, qmean_EW, quan95_EW, C_EW = createSamples(
                DAT, j, Weqok, n_sample, global_log[j],
                [global_minVal[j], global_maxVal[j]], 0)

            print("%2i %9.2f %9.2f %9.2f %9.2f" %
                  (j, quan05_EW, quan50_EW, qmean_EW, quan95_EW))

            q_EW[j, 0] = quan05_EW
            q_EW[j, 1] = quan50_EW
            q_EW[j, 2] = quan95_EW

            if Cooke_flag:

                quan05, quan50, qmean, quan95, C = createSamples(
                    DAT, j, W[:, 4].flatten(), n_sample, global_log[j],
                    [global_minVal[j], global_maxVal[j]], 0)

                print("%2i %9.2f %9.2f %9.2f %9.2f" %
                      (j, quan05, quan50, qmean, quan95))

                q_Cooke[j, 0] = quan05
                q_Cooke[j, 1] = quan50
                q_Cooke[j, 2] = quan95

            else:

                C = C_EW

            quan05_erf, quan50_erf, qmean_erf, quan95_erf, C_erf = createSamples(
                DAT, j, W_erf[:, 4].flatten(), n_sample, global_log[j],
                [global_minVal[j], global_maxVal[j]], ERF_flag)

            print("%2i %9.2f %9.2f %9.2f %9.2f" %
                  (j, quan05_erf, quan50_erf, qmean_erf, quan95_erf))

            q_erf[j, 0] = quan05_erf
            q_erf[j, 1] = quan50_erf
            q_erf[j, 2] = quan95_erf

            if (j >= n_SQ):

                if Cooke_flag:

                    samples[:, j - n_SQ] = C

                if ERF_flag > 0:

                    samples_erf[:, j - n_SQ] = C_erf

                samples_EW[:, j - n_SQ] = C_EW

            if (j >= n_SQ) and postprocessing:

                legendsPDF = []
                legendsPDF.append('CM' + '%9.2f' % quan05 + '%9.2f' % quan50 +
                                  '%9.2f' % quan95)
                legendsPDF.append('ERF' + '%9.2f' % quan05_erf +
                                  '%9.2f' % quan50_erf + '%9.2f' % quan95_erf)
                legendsPDF.append('EW' + '%9.2f' % quan05_EW +
                                  '%9.2f' % quan50_EW + '%9.2f' % quan95_EW)
                legendsPDF = [legendsPDF[index] for index in keep_rows]

                create_fig_hist(group, j, n_sample, n_SQ, hist_type, C, C_erf,
                                C_EW, colors, legends, legendsPDF,
                                global_units, Cooke_flag, ERF_flag, EW_flag,
                                global_minVal, global_maxVal, output_dir,
                                elicitation_name, del_rows, TQ_units,
                                label_indexes, minval_all, maxval_all)

    return q_Cooke, q_erf, q_EW, samples, samples_erf, samples_EW


def main():

    from ElicipyDict import output_dir

    from ElicipyDict import datarepo
    from ElicipyDict import Repository
    from ElicipyDict import group_list

    if len(group_list) > 1 and not (0 in group_list):

        group_list.append(0)

    n_pctl = 3

    # download the data from github repository
    if datarepo == 'github':

        from ElicipyDict import user
        from ElicipyDict import github_token

        saveDataFromGithub(Repository, user, github_token)

    # get current path
    path = os.getcwd()
    # change current path to elicitation folder
    path = path + '/' + Repository
    print('Path', path)

    os.chdir(path)

    sys.path.insert(0, os.getcwd())
    input_dir = 'DATA'
    csv_file = 'questionnaire.csv'

    # change to full path
    output_dir = path + '/' + output_dir
    input_dir = path + '/' + input_dir

    # Check whether the specified output path exists or not
    isExist = os.path.exists(output_dir)

    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(output_dir)
        print('The new directory ' + output_dir + ' is created!')

    # Read the questionnaire csv file
    df_indexes_SQ, df_indexes_TQ, SQ_scale, SQ_realization, TQ_scale, \
        SQ_minVals, SQ_maxVals, TQ_minVals, TQ_maxVals, SQ_units, TQ_units, \
        SQ_LongQuestion, TQ_LongQuestion, SQ_question, TQ_question, idx_list, \
        global_scale, global_log, label_indexes, parents, global_idxMin, \
        global_idxMax, global_sum50 = \
        read_questionnaire(input_dir, csv_file)

    # Read the asnwers of all the experts
    group = 0
    n_experts, n_SQ, n_TQ, SQ_array, TQ_array = read_answers(
        input_dir, csv_file, group, n_pctl, df_indexes_SQ, df_indexes_TQ)

    minval_all = np.zeros(n_SQ + n_TQ)
    minval_all[0:n_SQ] = np.amin(SQ_array[:, 0, :], axis=0)
    minval_all[n_SQ:] = np.amin(TQ_array[:, 0, :], axis=0)

    maxval_all = np.zeros(n_SQ + n_TQ)
    maxval_all[0:n_SQ] = np.amax(SQ_array[:, 2, :], axis=0)
    maxval_all[n_SQ:] = np.amax(TQ_array[:, 2, :], axis=0)

    print('')
    print('Answer ranges')

    for i in range(n_SQ + n_TQ):

        print(i, minval_all[i], maxval_all[i])

    print('')
    
    q_Cooke= np.zeros((len(global_scale), 3))
    q_erf = np.zeros((len(global_scale), 3))
    q_EW = np.zeros((len(global_scale), 3))

    if len(group_list) > 1:

        q_Cooke_groups = np.zeros((len(global_scale), 3, len(group_list)))
        q_erf_groups = np.zeros((len(global_scale), 3, len(group_list)))
        q_EW_groups = np.zeros((len(global_scale), 3, len(group_list)))

    for count, group in enumerate(group_list):

        # Read the asnwers of the experts
        n_experts, n_SQ, n_TQ, SQ_array, TQ_array = read_answers(
            input_dir, csv_file, group, n_pctl, df_indexes_SQ, df_indexes_TQ)

        if target:

            nTot = TQ_array.shape[2] + SQ_array.shape[2]

        else:

            nTot = SQ_array.shape[2]
            # if we do not read the target questions, set empty array
            n_TQ = 0
            TQ_array = np.zeros((n_experts, n_pctl, n_TQ))

        realization = np.zeros(TQ_array.shape[2] + SQ_array.shape[2])
        realization[0:SQ_array.shape[2]] = SQ_realization

        global_minVal = SQ_minVals + TQ_minVals
        global_maxVal = SQ_maxVals + TQ_maxVals
        global_units = SQ_units + TQ_units
        global_longQuestion = SQ_LongQuestion + TQ_LongQuestion
        global_shortQuestion = SQ_question + TQ_question

        print("")
        print('Realization', realization)

        if analysis:

            tree = {'IDX': idx_list, 'SHORT_Q': TQ_question}
            df_tree = pd.DataFrame(data=tree)

            # ----------------------------------------- #
            # ------------ Compute weights ------------ #
            # ----------------------------------------- #

            if group == 0:

                W, W_erf, Weqok, W_gt0, Werf_gt0, expin = answer_analysis(
                    input_dir, csv_file, n_experts, n_SQ, n_TQ, SQ_array, TQ_array,
                    realization, global_scale, global_log, alpha)

            else:
    
                # set alpha to zero
                W, W_erf, Weqok, W_gt0, Werf_gt0, expin = answer_analysis(
                    input_dir, csv_file, n_experts, n_SQ, n_TQ, SQ_array, TQ_array,
                    realization, global_scale, global_log, 0.0)

            # ----------------------------------------- #
            # ------ Create samples and bar plots ----- #
            # ----------------------------------------- #

            q_Cooke, q_erf, q_EW, samples, samples_erf, samples_EW = \
                create_samples_and_barplot(group, n_experts, n_SQ, n_TQ, n_pctl,
                                           SQ_array, TQ_array, n_sample, W, W_erf,
                                           Weqok, W_gt0, Werf_gt0, expin,
                                           global_log, global_minVal,
                                           global_maxVal, global_units, TQ_units,
                                           label_indexes, minval_all, maxval_all,
                                           postprocessing)

            if len(group_list) > 1:
    
                q_Cooke_groups[:, :, count] = q_Cooke
                q_erf_groups[:, :, count] = q_erf
                q_EW_groups[:, :, count] = q_EW

    # ----------------------------------------- #
    # ---------- Save samples on csv ---------- #
    # ----------------------------------------- #

    if analysis:

        csv_name = output_dir + '/' + elicitation_name + '_weights.csv'

        Weqok_100 = [ 100.0*elem for elem in Weqok ]

        Weqok_formatted = [ '%.2f' % elem for elem in Weqok_100 ]

        d = {'index': range(1, n_experts + 1), 'Weq': Weqok_formatted}
        df = pd.DataFrame(data=d)

        if Cooke_flag > 0:

            df.insert(loc=2, column='WCooke', value=W_gt0)

        if ERF_flag > 0:

            df.insert(loc=2, column='WERF', value=Werf_gt0)
            
        df.to_csv(csv_name, index=False)

    if analysis and target:

        targets = ['target_' + str(i).zfill(2) for i in range(n_TQ)]

        df_tree["EW_5"] = q_EW[n_SQ:, 0]
        df_tree["EW_50"] = q_EW[n_SQ:, 1]
        df_tree["EW_95"] = q_EW[n_SQ:, 2]

        if Cooke_flag > 0:

            df_tree["COOKE_5"] = q_Cooke[n_SQ:, 0]
            df_tree["COOKE_50"] = q_Cooke[n_SQ:, 1]
            df_tree["COOKE_95"] = q_Cooke[n_SQ:, 2]

            csv_name = output_dir + '/' + elicitation_name + '_samples.csv'
            np.savetxt(csv_name,
                       samples,
                       header=','.join(targets),
                       comments='',
                       delimiter=",",
                       fmt='%1.4e')

        if ERF_flag > 0:

            df_tree["ERF_5"] = q_erf[n_SQ:, 0]
            df_tree["ERF_50"] = q_erf[n_SQ:, 1]
            df_tree["ERF_95"] = q_erf[n_SQ:, 2]

            csv_name = output_dir + '/' + elicitation_name + '_samples_erf.csv'
            np.savetxt(csv_name,
                       samples_erf,
                       header=','.join(targets),
                       comments='',
                       delimiter=",",
                       fmt='%1.4e')

        if EW_flag > 0:

            csv_name = output_dir + '/' + elicitation_name + '_samples_EW.csv'
            np.savetxt(csv_name,
                       samples_EW,
                       header=','.join(targets),
                       comments='',
                       delimiter=",",
                       fmt='%1.4e')

        df_tree["PARENT"] = parents
        df_tree.to_csv('tree.csv', index=False)

    
    if not postprocessing:
    
        print('Analysis completed!')
        sys.exit()
    
    if analysis:

        # ------------------------------------------ #
        # --------- Create trend. figures ---------- #
        # ------------------------------------------ #

        try:

            from ElicipyDict import trend_groups
            print('trend_groups read', trend_groups)

        except ImportError:

            print('No trend group defined')
            trend_groups = []

        for count,trend_group in enumerate(trend_groups):

            create_figure_trend(count,trend_group,n_SQ,q_EW,q_Cooke,q_erf,global_units,
                            Cooke_flag, ERF_flag, EW_flag, global_log, 
                            TQ_minVals, TQ_maxVals, output_dir)


    # ----------------------------------------- #
    # --------- Create answ. figures ---------- #
    # ----------------------------------------- #

    n_panels = int(np.ceil(n_experts / max_len_plot))

    for h in np.arange(n_SQ + n_TQ):

        for k in np.arange(n_panels):

            create_figure(h, k, n_experts, max_len_plot, n_SQ, SQ_array,
                          TQ_array, realization, analysis, Cooke_flag,
                          ERF_flag, EW_flag, global_units, output_dir, q_Cooke,
                          q_erf, q_EW, elicitation_name, global_log,
                          label_indexes)

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

    Current_Date_Formatted = datetime.datetime.today().strftime('%d-%b-%Y')

    subtitle.text = Current_Date_Formatted  # subtitle

    subtitle_para = slide.shapes.placeholders[1].text_frame.paragraphs[0]
    subtitle_para.font.name = "Helvetica"

    img = slide.shapes.add_picture('../logo.png',
                                   left + Inches(11.3),
                                   top + Inches(5.4),
                                   width=Inches(2.4))

    title_slide_layout = prs.slide_layouts[5]

    # ------------- Weights slide -------------#

    if analysis:

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
        add_small_logo(slide, left, top)

        n_tables = int(np.ceil(len(W_gt0) / max_len_table))

        for i_table in range(n_tables):

            slide = prs.slides.add_slide(title_slide_layout)

            text_title = "Experts' weights"
            add_title(slide, text_title)

            # ---add table weights to slide---
            x, y, cx, cy = Inches(2), Inches(1.7), Inches(8), Inches(4)

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
            cell.text = 'Expert ID'

            cell = table.cell(0, 1)
            cell.text = 'Cooke'

            if ERF_flag > 0:

                cell = table.cell(0, 2)
                cell.text = 'ERF'

            for j in np.arange(fisrt_j, last_j):
                j_mod = np.remainder(j, max_len_table)
                cell = table.cell(j_mod + 1, 0)
                cell.text = 'Exp' + str(expin[j])

                cell = table.cell(j_mod + 1, 1)

                if W_gt0[j] > 0.0:

                    cell.text = '%6.2f' % W_gt0[j]

                else:

                    cell.text = 'Below threshold'

                if ERF_flag > 0:

                    cell = table.cell(j_mod + 1, 2)
                    cell.text = '%6.2f' % Werf_gt0[j]

            for cell in iter_cells(table):
                for paragraph in cell.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(12)

            if EW_flag:

                text_box = 'Equal weight = ' + f"{Weqok[0]*100:.2f}"
                add_text_box(slide, Inches(12), Inches(2), text_box,18)

                text_box = "For Cookes' method, below threshold weight does not mean zero score. It simply means that this expert's knowledge was already contributed by other experts and adding this expert would not change significantly the results."
                add_text_box(slide, Inches(12), Inches(3), text_box,18)

            add_date(slide)
            add_small_logo(slide, left, top)

    # ------------- Answers slides ------------#

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
    add_small_logo(slide, left, top)

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
            add_small_logo(slide, left, top)

        if (h >= n_SQ):

            j = h - n_SQ
            string = 'Target'

        else:

            j = h
            string = 'Seed'

        for k in np.arange(n_panels):

            slide = prs.slides.add_slide(title_slide_layout)

            text_box = global_longQuestion[h]
            add_text_box(slide, left, top, text_box,18)

            figname = output_dir + '/' + elicitation_name + \
                '_'+string+'_' + str(j + 1).zfill(2) + \
                '_' + str(k + 1).zfill(2) + '.png'
            add_figure(slide, figname, left, top)

            if analysis and (string == 'Target'):

                figname = output_dir + '/' + elicitation_name + \
                    '_PDF_group0_' + str(j + 1).zfill(2) + '.png'

                width = Inches(3.3)
                add_small_figure(slide, figname, left - Inches(1),
                                 top + Inches(4.5), width)
                                 
                if global_idxMin[h] < global_idxMax[h]:

                    longQ_NB = "N.B. The sum of 50%iles for questions " + \
                        str(global_idxMin[h])+"-"+str(global_idxMax[h]) + \
                        " have to sum to "+str(global_sum50[h])+"."                 

                    add_text_box(slide, left, top+Inches(3.39), longQ_NB,15)


            add_date(slide)
            add_small_logo(slide, left, top)

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
        add_small_logo(slide, left, top)

        n_tables = int(np.ceil(n_TQ / max_len_tableB))

        for i_table in range(n_tables):

            slide = prs.slides.add_slide(prs.slide_layouts[5])

            fisrt_j = i_table * max_len_tableB
            last_j = np.minimum((i_table + 1) * max_len_tableB, n_TQ)

            n_columns = 4

            if EW_flag > 0:

                n_columns += 3

            if ERF_flag > 0:

                n_columns += 3

            # ---add table to slide---
            x, y, cx, cy = Inches(1), Inches(2), Inches(14), Inches(4)
            shape = slide.shapes.add_table(last_j - fisrt_j + 1, n_columns, x,
                                           y, cx,
                                           MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT)
            table = shape.table

            cell = table.cell(0, 1)
            cell.text = 'Q05 (EW)'

            cell = table.cell(0, 2)
            cell.text = 'Q50 (EW)'

            cell = table.cell(0, 3)
            cell.text = 'Q95 (EW)'

            j_column = 3

            if EW_flag > 0:

                cell = table.cell(0, j_column + 1)
                cell.text = 'Q05 (Cooke)'

                cell = table.cell(0, j_column + 2)
                cell.text = 'Q50 (Cooke)'

                cell = table.cell(0, j_column + 3)
                cell.text = 'Q95 (Cooke)'
                j_column += 3

            if ERF_flag > 0:

                cell = table.cell(0, j_column + 1)
                cell.text = 'Q05 (ERF)'

                cell = table.cell(0, j_column + 2)
                cell.text = 'Q50 (ERF)'

                cell = table.cell(0, j_column + 3)
                cell.text = 'Q95 (ERF)'

            for h in np.arange(fisrt_j, last_j):

                h_mod = np.remainder(h, max_len_tableB)

                j = h + n_SQ

                cell = table.cell(h_mod + 1, 0)
                cell.text = 'TQ' + str(h + 1)

                for l in range(3):

                    cell = table.cell(h_mod + 1, l + 1)
                    cell.text = '%6.2f' % q_EW[j, l]
                    j_column = 3

                    if EW_flag > 0:

                        cell = table.cell(h_mod + 1, j_column + l + 1)
                        cell.text = '%6.2f' % q_Cooke[j, l]
                        j_column += 3

                    if ERF_flag > 0:

                        cell = table.cell(h_mod + 1, j_column + l + 1)
                        cell.text = '%6.2f' % q_erf[j, l]

            for cell in iter_cells(table):
                for paragraph in cell.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(14)

            add_date(slide)
            add_small_logo(slide, left, top)

            text_title = "Percentiles of target questions"
            add_title(slide, text_title)

    # ----------- Trend groups slides --------#
    
    if analysis and len(trend_groups) >0:
    
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
        add_small_logo(slide, left, top)
    
        for count,trend_group in enumerate(trend_groups):

            slide = prs.slides.add_slide(title_slide_layout)

            figname = output_dir + '/' + elicitation_name + '_trend_'+ \
                     str(count + 1).zfill(2)+'.png'
            
            text_box = ''
            for i in trend_group:
            
                text_box = text_box + 'TQ' + str(i) + '. ' + TQ_question[i-1] + '.\n\n'
            
            
            add_text_box(slide, left, top, text_box,18)

            add_date(slide)
            add_small_logo(slide, left, top)
            add_figure(slide, figname, left - Inches(0.8), top)

            text_title = 'Target questions Group ' +str(count+1)
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
                add_small_logo(slide, left, top)

            if (j >= n_SQ):

                slide = prs.slides.add_slide(title_slide_layout)

                figname = output_dir + '/' + elicitation_name + \
                    '_hist_' + str(j - n_SQ + 1).zfill(2) + '.png'

                text_box = TQ_LongQuestion[j - n_SQ]
                add_text_box(slide, left, top, text_box,18)

                add_date(slide)
                add_small_logo(slide, left, top)
                add_figure(slide, figname, left - Inches(0.8), top)

                text_title = TQ_question[j - n_SQ]
                add_title(slide, text_title)

                # ---add table to slide---

                n_rows = 2

                if EW_flag > 0:

                    n_rows += 1

                if ERF_flag > 0:

                    n_rows += 1
                x, y, cx, cy = Inches(1), Inches(6), Inches(3.5), Inches(3)
                shape = slide.shapes.add_table(n_rows, 4, x, y, cx,
                                               MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT)
                table = shape.table

                cell = table.cell(0, 1)
                cell.text = 'Q05'

                cell = table.cell(0, 2)
                cell.text = 'Q50'

                cell = table.cell(0, 3)
                cell.text = 'Q95'

                j_row = 0

                if EW_flag > 0:

                    cell = table.cell(j_row + 1, 0)
                    cell.text = 'Cooke'

                    for l in range(3):

                        cell = table.cell(j_row + 1, l + 1)
                        cell.text = '%6.2f' % q_Cooke[j, l]

                    j_row += 1

                if ERF_flag > 0:

                    cell = table.cell(j_row + 1, 0)
                    cell.text = 'ERF'

                    for l in range(3):

                        cell = table.cell(j_row + 1, l + 1)
                        cell.text = '%6.2f' % q_erf[j, l]

                    j_row += 1

                if EW_flag > 0:

                    cell = table.cell(j_row + 1, 0)
                    cell.text = 'EW'

                    for l in range(3):

                        cell = table.cell(j_row + 1, l + 1)
                        cell.text = '%6.2f' % q_EW[j, l]

                    j_row += 1

                for cell in iter_cells(table):
                    for paragraph in cell.text_frame.paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(14)

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
                add_small_logo(slide, left, top)

            if (j >= n_SQ):

                slide = prs.slides.add_slide(title_slide_layout)

                for count, group in enumerate(group_list):

                    figname = output_dir + '/' + elicitation_name + \
                        '_PDF_group' + str(group) + '_' + \
                        str(j - n_SQ + 1).zfill(2) + '.png'
                    if group == 0:

                        width = Inches(6.0)
                        add_small_figure(slide, figname, Inches(5.5),
                                         Inches(2.0), width)

                    else:

                        width = Inches(4.22)
                        add_small_figure(
                            slide, figname, Inches(11.18),
                            Inches(2.2) + (group - 1) * Inches(3.02), width)

                text_box = TQ_LongQuestion[j - n_SQ]
                add_text_box(slide, left, top, text_box,18)

                add_date(slide)
                add_small_logo(slide, left, top)

                text_title = TQ_question[j - n_SQ]
                add_title(slide, text_title)

                n_rows = 4

                if EW_flag > 0:

                    n_rows += 3

                if ERF_flag > 0:

                    n_rows += 3

                x, y, cx, cy = Inches(1), Inches(5.5), Inches(3.5), Inches(3)
                shape = slide.shapes.add_table(n_rows, 4, x, y, cx,
                                               MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT)

                table = shape.table

                cell = table.cell(0, 1)
                cell.text = 'Q05'

                cell = table.cell(0, 2)
                cell.text = 'Q50'

                cell = table.cell(0, 3)
                cell.text = 'Q95'

                j_row = 0

                if EW_flag > 0:

                    for count, group in enumerate(group_list):

                        cell = table.cell(j_row + 1 + group, 0)

                        if group == 0:
                            cell.text = 'Cooke'

                        else:
                            cell.text = 'Cooke ' + str(group)

                        for l in range(3):

                            cell = table.cell(j_row + 1 + group, l + 1)

                            if global_units[j] == "%":
                                cell.text = '%6.2f' % q_Cooke_groups[j, l,
                                                                     count]
                            else:
                                cell.text = '%.2E' % q_Cooke_groups[j, l,
                                                                    count]

                    j_row += len(group_list)

                if ERF_flag > 0:

                    for count, group in enumerate(group_list):

                        cell = table.cell(j_row + 1 + group, 0)

                        if group == 0:
                            cell.text = 'ERF'

                        else:
                            cell.text = 'ERF ' + str(group)

                        for l in range(3):

                            cell = table.cell(j_row + 1 + group, l + 1)

                            if global_units[j] == "%":
                                cell.text = '%6.2f' % q_erf_groups[j, l, count]
                            else:
                                cell.text = '%.2E' % q_erf_groups[j, l, count]

                    j_row += len(group_list)

                if EW_flag > 0:

                    for count, group in enumerate(group_list):

                        cell = table.cell(j_row + 1 + group, 0)

                        if group == 0:
                            cell.text = 'EW'

                        else:
                            cell.text = 'EW ' + str(group)

                        for l in range(3):

                            cell = table.cell(j_row + 1 + group, l + 1)

                            if global_units[j] == "%":
                                cell.text = '%6.2f' % q_EW_groups[j, l, count]
                            else:
                                cell.text = '%.2E' % q_EW_groups[j, l, count]

                    j_row += len(group_list)

                for cell in iter_cells(table):
                    for paragraph in cell.text_frame.paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(12)

    prs.save(output_dir + "/" + elicitation_name + ".pptx")  # saving file


if __name__ == '__main__':
    main()
