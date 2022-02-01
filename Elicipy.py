import datetime
import copy
from pptx.parts.chart import ChartPart
from pptx.parts.embeddedpackage import EmbeddedXlsxPart
from difflib import SequenceMatcher
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE
from pptx import Presentation
import difflib
from scipy import stats
from matplotlib.ticker import PercentFormatter
from script_fromR import createDATA1
from item_weights import item_weights
from global_weights import global_weights
from scipy.stats import chi2
import numpy as np
import pkg_resources
import re
import os
import sys
from pdf2image import convert_from_path
from pathlib import Path
import openpyxl
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

elicitation_name = 'test'

analysis = True
target = True
log = 0
n_sample = 1000
n_bins = 10
hist_type = 'bar'
#hist_type = 'step'

DEF_scale = 'uni'

path = './OUTPUT'

# Check whether the specified path exists or not
isExist = os.path.exists(path)

if not isExist:

    # Create a new directory because it does not exist
    os.makedirs(path)
    print("The new directory OUTPUT is created!")

if os.path.isfile('scale_sd.txt') == True and os.path.isfile(
        'scale_tg.txt') == True:
    SQ_scalea = np.loadtxt('scale_sd.txt', dtype='str')
    SQ_scale = SQ_scalea.tolist()
    TQ_scalea = np.loadtxt('scale_tg.txt', dtype='str', delimiter=',')
    TQ_scale = TQ_scalea.tolist()
    ALL_scale = SQ_scale + TQ_scale


def iter_cells(table):
    for row in table.rows:
        for cell in row.cells:
            yield cell


required = {'easygui', 'tkinter'}
installed = {pkg.key for pkg in pkg_resources.working_set}

print("Select your *.csv file for seed")

if ('easygui' in installed):
    import easygui
    filename = easygui.fileopenbox(msg='select your *.csv file',
                                   filetypes=['*.csv'])

elif ('tkinter' in installed):
    from tkinter import *
    root = Tk()
    root.filename = filedialog.askopenfilename(title='select your *.csv file',
                                               filetypes=[("csv files",
                                                           "*.csv")])
    filename = root.filename
    root.destroy()

# Read a comma-separated values (csv) file into DataFrame df_SQ
df_SQ = pd.read_csv(filename)

# The second column (index 1) must contain the first name of the expert
firstname = df_SQ[df_SQ.columns[1]].astype(str).tolist()

# The third column (index 2) must contain the first name of the expert
surname = df_SQ[df_SQ.columns[2]].astype(str).tolist()

# create a list with firstname+surname
# this is needed to search for expert matches between seed and target questions
NS_SQ = []

for name, surname in zip(firstname, surname):

    NS_SQ.append(name + surname)

print('NS_SQ',NS_SQ)

# create a 2D numpy array with the answers to the seed questions
cols_as_np = df_SQ[df_SQ.columns[3:]].to_numpy()

# we want to work with a 3D array, with the following dimension:
# n_expert X n_pctl X n_SQ
n_experts = cols_as_np.shape[0]
n_pctl = 3
n_SQ = int(cols_as_np.shape[1] / n_pctl)

# reshaped numpy array with expert answers
SQ_array = np.reshape(cols_as_np, (n_experts, n_SQ, n_pctl))

# swap the array to have seed for the last index
SQ_array = np.swapaxes(SQ_array, 1, 2)

# sort according to the percentile values
# (sometimes the expert give the percentiles in the wrong order)
SQ_array = np.sort(SQ_array, axis=1)

# list with the "title" of the seed questions
SQ_title = []

# list with the units of the seed questions
SQ_units = []

# scale for seed question:
# - uni
# - log

SQ_scale = []

for i in range(3, 3 + n_SQ * n_pctl, 3):

    # the units must be written in the field with the question title,
    # between square brackets, and separated with a ";" from the scale
    string1 = df_SQ.columns[i]
    match1 = string1[string1.index("[") + 1:string1.index("]")]

    match1_splitted = match1.split(';')
    su = match1_splitted[0]
    SQ_units.append(su)

    # if the scale is defined, append it to the list
    if len(match1_splitted) == 2:

        sc = match1_splitted[0]

        # scale must be "uni" or "log"
        if sc == 'uni':

            # when "uni", it can be "uni%" or simply "uni"
            if su == '%':

                SQ_scale.append('uni%')

            else:

                SQ_scale.append('uni')

        elif sc == 'log':

            SQ_scale.append('log')

        else:

            # when different from "uni" or "log", append empy string
            SQ_scale.append(DEF_scale)

    else:

        # if it is not defined, append empty sting
        SQ_scale.append(DEF_scale)

    # for each seed question, we ask for three percentiles, so we have
    # three columns in the csv files, with three column names.
    # We take the longest common string among these names and we
    # assign the string to the title "SQ_title".
    string1 = df_SQ.columns[i].split("[")[0]
    string2 = df_SQ.columns[i + 1].split("[")[0]
    string3 = df_SQ.columns[i + 2].split("[")[0]
    match12 = SequenceMatcher(None, string1, string2).find_longest_match(
        0, len(string1), 0, len(string2))
    string12 = string1[match12.a:match12.a + match12.size]
    match = SequenceMatcher(None, string12, string3).find_longest_match(
        0, len(string12), 0, len(string3))

    SQ_title.append(string12[match.a:match.a + match.size])
    print('Seed question '+str(int(i/3-1)),string12[match.a:match.a + match.size])

# print on screen the units
print("Seed_units = ", SQ_units)

# print on screen the units
print("Seed_scales = ", SQ_scale)


for i in np.arange(n_SQ):

    """
    n_lin = 0
    n_log = 0

    check_lin = []
    check_log = []
    """

    for k in np.arange(n_experts):

        """
        if ( SQ_array[k,0,i] < SQ_array[k,1,i]) and ( SQ_array[k,2,i] > SQ_array[k,1,i] ):

            check_lin.append((SQ_array[k,2,i]-SQ_array[k,1,i])/(SQ_array[k,1,i]-SQ_array[k,0,i])) 
            check_log.append((np.log(SQ_array[k,2,i])-np.log(SQ_array[k,1,i]))/(np.log(SQ_array[k,1,i])-np.log(SQ_array[k,0,i]))) 
        """

        # if 5% and 50% percentiles are equal, reduce 5%
        if SQ_array[k, 0, i] == SQ_array[k, 1, i]:

            SQ_array[k, 0, i] = SQ_array[k, 1, i] * 0.99

        # if 50% and 95% percentiles are equal, increase 95%
        if SQ_array[k, 2, i] == SQ_array[k, 1, i]:

            SQ_array[k, 2, i] = SQ_array[k, 1, i] * 1.01

    print('')
    print('Seed question ', i)
    print(SQ_array[:, :, i])
    # print('mean(lin) =',np.nanmean(check_lin),'mean(log) =',np.nanmean(check_log))

if target:

    print("")
    print("Select your *.csv file for target")

    if ('easygui' in installed):
        import easygui
        filename = easygui.fileopenbox(msg='select your *.csv file',
                                       filetypes=['*.csv'])

    elif ('tkinter' in installed):
        from tkinter import *
        root = Tk()
        root.filename = filedialog.askopenfilename(
            title='select your *.csv file', filetypes=[("csv files", "*.csv")])
        filename = root.filename
        root.destroy()

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

    print('NS_TQ',NS_TQ)

    sorted_idx = []

    # loop to search for matches between experts in seed and target
    for TQ_name in NS_TQ:
    
        index = NS_SQ.index(difflib.get_close_matches(TQ_name, NS_SQ)[0])
        sorted_idx.append(index)
        
    print('Sorted list of experts to match the order of seeds:', sorted_idx)
    

    # create a 2D numpy array with the answers to the seed questions
    cols_as_np = df_TQ[df_TQ.columns[3:]].to_numpy()

    # sort for expert names
    cols_as_np = cols_as_np[sorted_idx, :]

    # we want to work with a 3D array, with the following dimension:
    # n_expert X n_pctl X n_TQ
    n_experts_TQ = cols_as_np.shape[0]

    # check if number of experts in seed and target is the same
    if (n_experts_TQ != n_experts):

        print('Error: number of experts in seeds and targets different')
        sys.exit()

    n_pctl = 3
    n_TQ = int(cols_as_np.shape[1] / n_pctl)

    # reshaped numpy array with expert answers
    TQ_array = np.reshape(cols_as_np, (n_experts, n_TQ, n_pctl))

    # swap the array to have pctls for the second index
    TQ_array = np.swapaxes(TQ_array, 1, 2)

    # sort according to the percentile values
    # (sometimes the expert give the percentiles in the wrong order)
    TQ_array = np.sort(TQ_array, axis=1)

    # list with the "title" of the target questions
    TQ_question = []

    # list with the units of the target questions
    TQ_units = []

    # scale for target question:
    # - uni
    # - log
    TQ_scale = []

    for i in range(3, 3 + n_TQ * n_pctl, 3):

        # the units must be written in the field with the question title,
        # between square brackets, and separated with a ";" from the scale
        string1 = df_TQ.columns[i]
        match1 = string1[string1.index("[") + 1:string1.index("]")]

        match1_splitted = match1.split(';')
        tg = match1_splitted[0]
        print('tg',tg)
        TQ_units.append(tg)

        # if the scale is defined, append it to the list
        if len(match1_splitted) == 2:

            sc = match1_splitted[0]

            # scale must be "uni" or "log"
            if sc == 'uni':

                # when "uni", it can be "uni%" or simply "uni"
                if su == '%':

                    TQ_scale.append('uni%')

                else:

                    TQ_scale.append('uni')

            elif sc == 'log':

                TQ_scale.append('log')

            else:

                # when different from "uni" or "log", append empy string
                TQ_scale.append(DEF_scale)

        else:

            # if it is not defined, append empty sting
            TQ_scale.append(DEF_scale)

        # for each seed question, we ask for three percentiles, so we have
        # three columns in the csv files, with three column names.
        # We take the longest common string among these names and we
        # assign the string to the title "TQ_question".
        string1 = df_TQ.columns[i].split("[")[0]
        string2 = df_TQ.columns[i + 1].split("[")[0]
        string3 = df_TQ.columns[i + 2].split("[")[0]
        match12 = SequenceMatcher(None, string1, string2).find_longest_match(
            0, len(string1), 0, len(string2))
        string12 = string1[match12.a:match12.a + match12.size]
        match = SequenceMatcher(None, string12, string3).find_longest_match(
            0, len(string12), 0, len(string3))

        TQ_question.append(string12[match.a:match.a + match.size])
        print('match', string12[match.a:match.a + match.size])

    # print on screen the units
    print("Target units = ", TQ_units)

    # print on screen the units
    print("Target scales = ", TQ_scale)

    ALL_scale = SQ_scale + TQ_scale

    for i in np.arange(n_TQ):

        for k in np.arange(n_experts):
            if TQ_array[k, 0, i] == TQ_array[k, 1, i]:
                TQ_array[k, 0, i] = TQ_array[k, 1, i] * 0.99
            if TQ_array[k, 2, i] == TQ_array[k, 1, i]:
                TQ_array[k, 2, i] = TQ_array[k, 1, i] * 1.01

        print('Target question ', i)
        print(TQ_array[:, :, i])

else:

    # if we do not read the target questions, set empty array
    n_TQ = 0
    n_pctl = 3

    TQ_array = np.zeros((n_experts, n_pctl, n_TQ))
    ALL_scale = SQ_scale

print("")
print("Select your *.xlsx file for realization")

if ('easygui' in installed):
    import easygui
    filename = easygui.fileopenbox(msg='select your *.xlsx file',
                                   filetypes=['*.xlsx'])

elif ('tkinter' in installed):
    from tkinter import *
    root = Tk()
    root.filename = filedialog.askopenfilename(title='select your *.xlsx file',
                                               filetypes=[("xlsx files",
                                                           "*.xlsx")])
    filename = root.filename
    root.destroy()

wb_obj = openpyxl.load_workbook(filename)

# Read the active sheet:
sheet = wb_obj.active

i = 0
a = []
for row in sheet.iter_rows(max_row=2):
    for cell in row:
        if i == 1:
            a.append(cell.value)

    i = i + 1

if target:

    nTot = TQ_array.shape[2] + SQ_array.shape[2]

else:

    nTot = SQ_array.shape[2]

realization = np.zeros(TQ_array.shape[2] + SQ_array.shape[2])
realization[0:SQ_array.shape[2]] = a[0:SQ_array.shape[2]]

print("")
print('Realization', realization)

back_measure = []

for i in np.arange(TQ_array.shape[2] + SQ_array.shape[2]):

    back_measure.append('uni')

# parameters for DM
alpha = 0.05  # significance level (this value cannot be higher than the
# highest calibration score of the pool of experts)
k = 0.1  # overshoot for intrinsic range

# global cal_power
cal_power = 1  # this value should be between [0.1, 1]. The default is 1.

optimization = 'no'  # choose from 'yes' or 'no'

weight_type = 'global'  # choose from 'equal', 'item', 'global', 'user'

N_max_it = 5  # maximum number of seed items to be removed at a time when

if analysis:

    if optimization == 'no':

        if weight_type == 'global':

            W = global_weights(SQ_array, TQ_array, realization, alpha, back_measure,
                               k, cal_power)

        elif weight_type == 'item':

            [unorm_w, W_itm,
             W_itm_tq] = item_weights(SQ_array, TQ_array, realization, alpha,
                                      back_measure, k, cal_power)

    Weq = np.ones(n_experts)
    Weqok = [x / n_experts for x in Weq]

    W_gt0_01 = []
    expin = []

    for x in W[:, 4]:
        if x > 0:
            W_gt0_01.append(x)

    k = 1
    for i in W[:, 4]:
        if i > 0:
            expin.append(k)
        k += 1

    W_gt0 = [round((x * 100), 1) for x in W_gt0_01]

    print("")
    print('W')
    print(W[:, -1])
    print("")
    print('Weq')
    print(Weqok)

DAT = np.zeros((n_experts * (n_SQ + n_TQ), n_pctl + 2))

DAT[:, 0] = np.repeat(np.arange(1, n_experts + 1), n_SQ + n_TQ)
DAT[:, 1] = np.tile(np.arange(1, n_SQ + n_TQ + 1), n_experts)

DAT[:, 2:] = np.append(SQ_array, TQ_array, axis=2).transpose(0, 2, 1).reshape(-1, 3)

q05 = []
q50 = []
q95 = []

q05_EW = []
q50_EW = []
q95_EW = []

figs_h = {}
axs_h = {}
axs_h2 = {}

plt.rcParams.update({'font.size': 8})

print("")
if analysis:
    print("j,quan05,quan50,qmean,quan95")

for j in np.arange(n_SQ + n_TQ):

    if analysis:

        if ALL_scale[j] == "uni%":

            quan05, quan50, qmean, quan95, C = createDATA1(
                DAT, j, W[:, 4].flatten(), n_sample, 'red', 10, 60, False, '',
                0, 0, [0, 100], 1)

            quan05_EW, quan50_EW, qmean_EW, quan95_EW, C_EW = createDATA1(
                DAT, j, Weqok, n_sample, 'green', 10, 60, False, '', 0, 0,
                [0, 100], 1)

        elif ALL_scale[j] == "uni":

            quan05, quan50, qmean, quan95, C = createDATA1(
                DAT, j, W[:, 4].flatten(), n_sample, 'red', 10, 60, False, '',
                0, 0, [0, np.inf], 1)

            quan05_EW, quan50_EW, qmean_EW, quan95_EW, C_EW = createDATA1(
                DAT, j, Weqok, n_sample, 'green', 10, 60, False, '', 0, 0,
                [0, np.inf], 1)

        else:

            quan05, quan50, qmean, quan95, C = createDATA1(
                DAT, j, W[:, 4].flatten(), n_sample, 'red', 10, 60, True, '',
                0, 0, [-np.inf, np.inf], 1)

            quan05_EW, quan50_EW, qmean_EW, quan95_EW, C_EW = createDATA1(
                DAT, j, Weqok, n_sample, 'green', 10, 60, True, '', 0, 0,
                [-np.inf, np.inf], 1)

        print(j, quan05, quan50, qmean, quan95)
        print(j, quan05_EW, quan50_EW, qmean_EW, quan95_EW)

        q05.append(quan05)
        q50.append(quan50)
        q95.append(quan95)

        q05_EW.append(quan05_EW)
        q50_EW.append(quan50_EW)
        q95_EW.append(quan95_EW)

        if (j >= n_SQ):

            ntarget = str(j - n_SQ + 1)

            figs_h[j] = plt.figure()
            axs_h[j] = figs_h[j].add_subplot(111)
            C_stack = np.stack((C, C_EW), axis=0)
            wg = np.ones_like(C_stack.T) / n_sample

            if hist_type == 'step':

                axs_h[j].hist(C_stack.T,
                              bins=n_bins,
                              weights=wg,
                              histtype='step',
                              fill=False,
                              rwidth=0.95,
                              color=['orange', 'springgreen'])

            elif hist_type == 'bar':

                axs_h[j].hist(C_stack.T,
                              bins=n_bins,
                              weights=wg,
                              histtype='bar',
                              rwidth=0.95,
                              ec="k",
                              color=['orange', 'springgreen'])

            axs_h[j].set_xlabel(TQ_units[j - n_SQ])
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

            xt = plt.xticks()[0]

            axs_h2[j] = axs_h[j].twinx()

            gkde = stats.gaussian_kde(C)
            gkde_EW = stats.gaussian_kde(C_EW)

            if ALL_scale[j] == "uni%":

                xmin = 0.0
                xmax = 100.0

            elif ALL_scale[j] == "uni":

                xmin = 0.0
                xmax = np.amax(C_stack)

            else:

                xmin = np.amin(C_stack)
                xmax = np.amax(C_stack)

            gkde_norm = gkde.integrate_box_1d(xmin, xmax)
            gkde_EW_norm = gkde_EW.integrate_box_1d(xmin, xmax)

            lnspc = np.linspace(xmin, xmax, 1000)
            kdepdf = gkde.evaluate(lnspc) / gkde_norm
            kdepdf_EW = gkde_EW.evaluate(lnspc) / gkde_EW_norm
            axs_h2[j].plot(lnspc, kdepdf, 'r--')
            axs_h2[j].plot(lnspc, kdepdf_EW, 'g--')

            axs_h[j].set_xlim(xmin, xmax)
            axs_h2[j].set_xlim(xmin, xmax)

            axs_h2[j].set_ylabel('PDF', color='b')

            axs_h2[j].set_ylim(bottom=0)
            plt.legend(['CM', 'EW'])
            plt.title('Target Question ' + str(j - n_SQ + 1))

            figname = path + '/' + elicitation_name + '_hist_' + str(j - n_SQ + 1).zfill(2) + '.pdf'
            figs_h[j].savefig(figname)

            images = convert_from_path(figname)
            figname = path + '/' + elicitation_name + '_hist_' + str(j - n_SQ + 1).zfill(2) + '.png'
            images[0].save(figname, 'PNG')

            plt.close()

h = 0
figs0 = {}
axs0 = {}

for j in np.arange(n_SQ):

    nseed = str(j + 1)

    x = SQ_array[:, 1, j]
    y = np.arange(n_experts) + 1

    # creating error
    x_errormax = SQ_array[:, 2, j] - SQ_array[:, 1, j]
    x_errormin = SQ_array[:, 1, j] - SQ_array[:, 0, j]

    x_error = [x_errormin, x_errormax]

    figs0[j] = plt.figure()
    axs0[j] = figs0[j].add_subplot(111)
    axs0[j].errorbar(x, y, xerr=x_error, fmt='bo')
    axs0[j].plot(x - x_errormin, y, 'bx')
    axs0[j].plot(x + x_errormax, y, 'bx')

    if analysis:

        axs0[j].errorbar([q50[h]],
                         n_experts + 1,
                         xerr=[[q50[h] - q05[h]], [q95[h] - q50[h]]],
                         fmt='ro')
        axs0[j].plot(q05[h], n_experts + 1, 'rx')
        axs0[j].plot(q95[h], n_experts + 1, 'rx')

        axs0[j].errorbar([q50_EW[h]],
                         n_experts + 2,
                         xerr=[[q50_EW[h] - q05_EW[h]],
                               [q95_EW[h] - q50_EW[h]]],
                         fmt='go')
        axs0[j].plot(q05_EW[h], n_experts + 2, 'gx')
        axs0[j].plot(q95_EW[h], n_experts + 2, 'gx')

        axs0[j].plot(realization[j], n_experts + 3, 'kx')

    else:

        axs0[j].plot(realization[j], n_experts + 1, 'kx')

    xt = plt.xticks()[0]
    xmin, xmax = min(xt), max(xt)

    if (realization[j] > 999):
        txt = '%5.2e' % realization[j]
    else:
        txt = '%6.2f' % realization[j]

    if analysis:

        b = np.amin([np.amin(SQ_array[:, 0, j]), q05[h], realization[j]])
        c = np.amin([np.amax(SQ_array[:, 0, j]), q95[h], realization[j]])
        axs0[j].annotate(txt, (realization[j], n_experts + 3 + 0.15))

    else:

        b = np.amin([np.amin(SQ_array[:, 0, j]), realization[j]])
        c = np.amin([np.amax(SQ_array[:, 0, j]), realization[j]])
        axs0[j].annotate(txt, (realization[j], n_experts + 1 + 0.15))

    ytick = []
    for i in y:
        ytick.append('Exp.' + str(int(i)))

    if analysis:

        ytick.append('DM-Cooke')

        ytick.append('DM-Equal')

    ytick.append('Realization')

    ytick_tuple = tuple(i for i in ytick)

    if analysis:

        y = np.arange(n_experts + 3) + 1

    else:

        y = np.arange(n_experts + 1) + 1

    axs0[j].set_yticks(y)
    axs0[j].set_yticklabels(ytick_tuple)
    axs0[j].set_xlabel(SQ_units[j])
    plt.title('Seed Question ' + str(j + 1))

    if (np.abs(c - b) >= 9.99e2):

        axs0[j].set_xscale('log')

    if analysis:
        axs0[j].set_ylim(0.5, n_experts + 4.0)
    else:
        axs0[j].set_ylim(0.5, n_experts + 2.0)

    axs0[j].grid()

    figname = path + '/' + elicitation_name + '_seed_' + str(j + 1).zfill(2) + '.pdf'
    figs0[j].savefig(figname)

    images = convert_from_path(figname)
    figname = path + '/' + elicitation_name + '_seed_' + str(j + 1).zfill(2) + '.png'
    images[0].save(figname, 'PNG')

    plt.close()

    h = h + 1

figs = {}
axs = {}

for j in np.arange(n_TQ):

    x = TQ_array[:, 1, j]
    y = np.arange(n_experts) + 1

    # creating error
    x_errormax = TQ_array[:, 2, j] - TQ_array[:, 1, j]
    x_errormin = TQ_array[:, 1, j] - TQ_array[:, 0, j]

    x_error = [x_errormin, x_errormax]

    figs[j] = plt.figure()
    axs[j] = figs[j].add_subplot(111)
    axs[j].errorbar(x, y, xerr=x_error, fmt='bo')
    axs[j].plot(x - x_errormin, y, 'bx')
    axs[j].plot(x + x_errormax, y, 'bx')

    if analysis:

        axs[j].errorbar(q50[h], [n_experts + 1],
                        xerr=[[q50[h] - q05[h]], [q95[h] - q50[h]]],
                        fmt='ro')
        axs[j].plot(q05[h], n_experts + 1, 'rx')
        axs[j].plot(q95[h], n_experts + 1, 'rx')

        axs[j].errorbar([q50_EW[h]],
                        n_experts + 2,
                        xerr=[[q50_EW[h] - q05_EW[h]],
                              [q95_EW[h] - q50_EW[h]]],
                        fmt='go')
        axs[j].plot(q05_EW[h], n_experts + 2, 'gx')
        axs[j].plot(q95_EW[h], n_experts + 2, 'gx')

    xt = plt.xticks()[0]
    xmin, xmax = min(xt), max(xt)

    if analysis:

        b = np.amin([np.amin(TQ_array[:, 0, j]), q05[h]])
        c = np.amin([np.amax(TQ_array[:, 0, j]), q95[h]])

    else:

        b = np.amin(TQ_array[:, 0, j])
        c = np.amax(TQ_array[:, 0, j])

    ytick = []
    for i in y:
        ytick.append('Exp.' + str(int(i)))

    ytick.append('DM-Cooke')

    ytick.append('DM-Equal')

    y = np.arange(n_experts + 2) + 1

    ytick_tuple = tuple(i for i in ytick)
    axs[j].set_yticks(y)
    axs[j].set_yticklabels(ytick_tuple)
    axs[j].set_xlabel(TQ_units[j])
    plt.title('Target Question ' + str(j + 1))

    if (np.abs(c - b) >= 9.99e2):

        axs[j].set_xscale('log')

    if analysis:
        axs[j].set_ylim(0.5, n_experts + 2.5)
    else:
        axs[j].set_ylim(0.5, n_experts + 0.5)

    axs[j].grid(linewidth=0.4)

    figname = path + '/' + elicitation_name + '_target_' + str(j + 1).zfill(2) + '.pdf'
    figs[j].savefig(figname)

    images = convert_from_path(figname)
    figname = path + '/' + elicitation_name + '_target_' + str(j + 1).zfill(2) + '.png'
    images[0].save(figname, 'PNG')

    plt.close()

    h = h + 1

prs = Presentation()
prs.slide_width = Inches(16)
prs.slide_height = Inches(9)

lyt = prs.slide_layouts[0]  # choosing a slide layout
slide = prs.slides.add_slide(lyt)  # adding a slide
title = slide.shapes.title  # assigning a title
subtitle = slide.placeholders[1]  # placeholder for subtitle
title.text = "Expert elicitation"  # title
Current_Date_Formatted = datetime.datetime.today().strftime('%d-%b-%Y')

subtitle.text = Current_Date_Formatted  # subtitle

if analysis:

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    title_shape = slide.shapes.title
    title_shape.text = "Experts' weights"
    title_para = slide.shapes.title.text_frame.paragraphs[0]
    title_para.font.name = "Helvetica"
    # ---add table weights to slide---
    x, y, cx, cy = Inches(2), Inches(2), Inches(8), Inches(4)
    #x, y, cx, cy = Inches(2), Inches(2), MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT, Inches(4)
    shape = slide.shapes.add_table(2,
                                   len(W_gt0) + 1, x, y, cx,
                                   MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT)
    #shape = slide.shapes.add_table(2, len(W_gt0)+1, x, y, cx, cy)
    table = shape.table

    cell = table.cell(0, 0)
    cell.text = 'Expert ID'

    cell = table.cell(1, 0)
    cell.text = 'Expert weight'

    for j in np.arange(len(W_gt0)):
        cell = table.cell(0, j + 1)
        cell.text = 'Exp' + str(expin[j])

        cell = table.cell(1, j + 1)
        cell.text = '%6.2f' % W_gt0[j]

    for cell in iter_cells(table):
        for paragraph in cell.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(12)

for j in np.arange(n_SQ):

    figname = path + '/' + elicitation_name + '_seed_' + str(j + 1).zfill(2) + '.png'

    title_slide_layout = prs.slide_layouts[5]
    left = Inches(2)
    top = Inches(1.5)

    slide = prs.slides.add_slide(title_slide_layout)

    title_shape = slide.shapes.title
    title_shape.text = SQ_title[j]
    title_shape.width = Inches(15)
    title_shape.height = Inches(2)
    
    title_para = slide.shapes.title.text_frame.paragraphs[0]

    title_para.font.name = "Helvetica"

    img = slide.shapes.add_picture('./' + figname, left, top, width=Inches(10))

    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(0.2),
                                   Inches(16), Inches(0.3))
    shape.shadow.inherit = False
    fill = shape.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(0, 0, 255)
    shape.text = "Expert elicitation " + \
        datetime.datetime.today().strftime('%d-%b-%Y')
    shape_para = shape.text_frame.paragraphs[0]
    shape_para.font.name = "Helvetica"

for j in np.arange(n_TQ):

    figname = path + '/' + elicitation_name + '_target_' + str(j + 1).zfill(2) + '.png'

    blank_slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(title_slide_layout)

    title_shape = slide.shapes.title
    title_shape.text = TQ_question[j]
    title_shape.width = Inches(15)
    title_shape.height = Inches(2)
    title_para = slide.shapes.title.text_frame.paragraphs[0]
    title_para.font.name = "Helvetica"

    img = slide.shapes.add_picture('./' + figname, left, top, width=Inches(10))

    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(0.2),
                                   Inches(16), Inches(0.3))
    shape.shadow.inherit = False
    fill = shape.fill
    fill.solid()
    fill.fore_color.rgb = RGBColor(0, 0, 255)
    shape.text = "Expert elicitation " + \
        datetime.datetime.today().strftime('%d-%b-%Y')
    shape_para = shape.text_frame.paragraphs[0]
    shape_para.font.name = "Helvetica"

if analysis and target:

    slide = prs.slides.add_slide(prs.slide_layouts[5])
    title_shape = slide.shapes.title
    title_shape.text = "Percentiles of target questions"
    title_para = slide.shapes.title.text_frame.paragraphs[0]
    title_para.font.name = "Helvetica"
    # ---add table to slide---
    x, y, cx, cy = Inches(2), Inches(2), Inches(12), Inches(4)
    shape = slide.shapes.add_table(n_TQ + 1, 4, x, y, cx,
                                   MSO_AUTO_SIZE.SHAPE_TO_FIT_TEXT)
    table = shape.table

    cell = table.cell(0, 1)
    cell.text = 'Q05'

    cell = table.cell(0, 2)
    cell.text = 'Q50'

    cell = table.cell(0, 3)
    cell.text = 'Q95'

    for j in np.arange(n_TQ):

        cell = table.cell(j + 1, 0)
        cell.text = 'Target Question ' + str(j + 1)

        cell = table.cell(j + 1, 1)
        cell.text = '%6.2f' % q05[n_SQ + j]

        cell = table.cell(j + 1, 2)
        cell.text = '%6.2f' % q50[n_SQ + j]

        cell = table.cell(j + 1, 3)
        cell.text = '%6.2f' % q95[n_SQ + j]

    for cell in iter_cells(table):
        for paragraph in cell.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(14)

for j in np.arange(n_SQ + n_TQ):

    if analysis:

        if (j >= n_SQ):

            figname = path + '/' + elicitation_name + '_hist_' + str(j - n_SQ + 1).zfill(2) + '.png'

            blank_slide_layout = prs.slide_layouts[6]
            title_slide_layout = prs.slide_layouts[5]
            slide = prs.slides.add_slide(title_slide_layout)
            left = Inches(2)
            top = Inches(1.5)

            title_shape = slide.shapes.title
            title_shape.text = TQ_question[j - n_SQ]
            title_para = slide.shapes.title.text_frame.paragraphs[0]
            title_para.font.name = "Helvetica"

            img = slide.shapes.add_picture('./' + figname,
                                           left,
                                           top,
                                           width=Inches(10))
            shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(0.2),
                                           Inches(16), Inches(0.3))
            shape.shadow.inherit = False
            fill = shape.fill
            fill.solid()
            fill.fore_color.rgb = RGBColor(0, 0, 255)
            shape.text = "Expert elicitation " + \
                datetime.datetime.today().strftime('%d-%b-%Y')
            shape_para = shape.text_frame.paragraphs[0]
            shape_para.font.name = "Helvetica"

prs.save(path + "/"+elicitation_name+".pptx")  # saving file
