import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import PercentFormatter

from tools import printProgressBar

plt.rcParams.update({"font.size": 8})

rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]

# matplotlib.use("TkAgg")


def create_fig_hist(
    group,
    j,
    n_sample,
    n_SQ,
    hist_type,
    C,
    C_erf,
    C_EW,
    colors,
    legends,
    legendsPDF,
    global_units,
    Cooke_flag,
    ERF_flag,
    EW_flag,
    global_log,
    global_minVal,
    global_maxVal,
    output_dir,
    elicitation_name,
    del_rows,
    TQ_units,
    label_indexes,
    minval_all,
    maxval_all,
    n_bins,
):
    """Create figure with histrogram and fit and small figure
       with PDFs only

    Parameters
    ----------
    group : int
        index of group of experts
    j : int
        index of the question
    n_sample : int
        number of samples used
    n_SQ : int
        number of seed questions
    hist_type : string
        string to select plot type ('step' or 'bar')
    C : float numpy array (size n_sample)
        array with samples for Cooke method
    C_erf : float numpy array (size n_sample)
        array with samples for ERF method
    C_EW : float numpy array (size n_sample)
        array with samples for equal weight
    colors : list of strings
        list with color names for plots
    legends : list of strings
        list with legend strings for histrograms
    legendsPDF : list of strings
        list with legend strings for PDFs
    global_units : list of strings
        list of strings for units of answers to add as xlabel
    Cooke_flag : int
        Cooke_flag > 0 => plot the Cooke method results
    ERF_flag : int
        ERF_flag > 0 => plot the ERF method results
    EW_flag : int
        EW_flag > 0 => plot the equal weights results
    global_log : list of int
        1: log scale; 0: linear scale
    global_minVal : list of float
        minimum allowed value for answer to question j
    global_maxVal : list of float
        maximum allowed value for answer to question j
    output_dir : string
        name of output folder
    elicitation_name : string
        name of the elicitation
    del_rows : list of int
        list of methods to skip (0:cooke, 1:ERF, 2:EW)
    TQ_units : list of strings
        list of strings for units of target answers to add as xlabel
    label_indexes : list of int
        index to use as label for the figure title
    minval_all : list of float
        minimum value for answers to question j among experts
    maxval_all : list of float
        maximum value for answers to question j among experts


    Returns
    -------
    none

    """

    from scipy import stats

    if not os.path.exists(output_dir + "/" + "Barplots_PNGPDF"):
        os.makedirs(output_dir + "/" + "Barplots_PNGPDF")

    fig = plt.figure()
    axs_h = fig.add_subplot(111)
    C_stack = np.stack((C, C_erf, C_EW), axis=0)
    C_stack = np.delete(C_stack, del_rows, 0)
    wg = np.ones_like(C_stack.T) / n_sample

    if global_units[j] == "%" and global_log[j] == 0:

        xmin = 0.0

    else:

        xmin = minval_all[j]

    if global_units[j] == "%":

        xmax = 100.0

    else:

        xmax = maxval_all[j]

    if global_log[j] == 1:

        bins = np.geomspace(minval_all[j], maxval_all[j], n_bins + 1)

    else:

        bins = np.linspace(minval_all[j], maxval_all[j], n_bins + 1)

    if hist_type == "step":

        (n, bins, patches) = axs_h.hist(
            C_stack.T,
            bins=bins,
            weights=wg,
            histtype="step",
            fill=False,
            rwidth=0.95,
            color=colors,
        )

    elif hist_type == "bar":

        (n, bins, patches) = axs_h.hist(
            C_stack.T,
            bins=bins,
            weights=wg,
            histtype="bar",
            rwidth=0.50,
            ec="k",
            color=colors,
        )

    axs_h.set_xlabel(TQ_units[j - n_SQ])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.xticks()[0]

    axs_h2 = axs_h.twinx()

    if global_log[j] == 1:

        pdf_min = np.log10(minval_all[j])
        pdf_max = np.log10(maxval_all[j])
        pdf_C = np.log10(C)
        pdf_C_erf = np.log10(C_erf)
        pdf_C_EW = np.log10(C_EW)
        pdf_spc = np.linspace(pdf_min, pdf_max, 1000)
        pdf_scale = (pdf_max - pdf_min) / (maxval_all[j] - minval_all[j])
        lnspc = 10**pdf_spc

    else:

        pdf_min = global_minVal[j]
        if pdf_min == float("-inf"):
            pdf_min = minval_all[j]

        pdf_max = global_maxVal[j]
        if pdf_max == float("inf"):
            pdf_max = maxval_all[j]

        pdf_C = C
        pdf_C_erf = C_erf
        pdf_C_EW = C_EW
        pdf_spc = np.linspace(pdf_min, pdf_max, 1000)
        pdf_scale = 1.0
        lnspc = pdf_spc

    if Cooke_flag > 0:

        gkde = stats.gaussian_kde(pdf_C)
        gkde_norm = gkde.integrate_box_1d(pdf_min, pdf_max)
        kdepdf = gkde.evaluate(pdf_spc) / gkde_norm * pdf_scale
        axs_h2.plot(lnspc, kdepdf, "r--")

    if ERF_flag > 0:
        gkde_erf = stats.gaussian_kde(pdf_C_erf)
        gkde_erf_norm = gkde_erf.integrate_box_1d(pdf_min, pdf_max)
        kdepdf_erf = gkde_erf.evaluate(pdf_spc) / gkde_erf_norm * pdf_scale
        axs_h2.plot(lnspc, kdepdf_erf, "--", color="tab:purple")

    if EW_flag > 0:
        gkde_EW = stats.gaussian_kde(pdf_C_EW)
        gkde_EW_norm = gkde_EW.integrate_box_1d(pdf_min, pdf_max)
        kdepdf_EW = gkde_EW.evaluate(pdf_spc) / gkde_EW_norm * pdf_scale
        axs_h2.plot(lnspc, kdepdf_EW, "g--")

    axs_h.set_xlim(xmin, xmax)
    axs_h2.set_xlim(xmin, xmax)
    axs_h2.set_ylabel("PDF", color="b")
    axs_h2.set_ylim(bottom=0)

    [ymin, ymax] = axs_h2.get_ylim()
    for xbin in bins:

        axs_h2.plot([xbin, xbin], [ymin, ymax], "k-", linewidth=0.2, alpha=0.1)

    if global_log[j] == 1:

        axs_h.set_xscale("log")
        axs_h2.set_xscale("log")

    plt.legend(legends)
    plt.title("Target Question " + label_indexes[j])

    figname = (output_dir + "/" + "Barplots_PNGPDF" + "/" + elicitation_name +
               "_hist_" + str(j - n_SQ + 1).zfill(2) + ".pdf")
    fig.savefig(figname)

    # images = convert_from_path(figname)
    figname = (output_dir + "/" + "Barplots_PNGPDF" + "/" + elicitation_name +
               "_hist_" + str(j - n_SQ + 1).zfill(2) + ".png")
    # images[0].save(figname, "PNG")
    fig.savefig(figname, dpi=300)

    plt.close()

    # CUMULATIVE PLOTS

    if not os.path.exists(output_dir + "/" + "Groups_PNGPDF"):
        os.makedirs(output_dir + "/" + "Groups_PNGPDF")

    fig_cum = plt.figure()
    axs_cum = fig_cum.add_subplot(111)
    axs_cum.set_xlabel(TQ_units[j - n_SQ])
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

    plt.xticks()[0]

    if Cooke_flag > 0:

        kde_cum = np.cumsum(kdepdf)
        kde_cum /= np.amax(kde_cum)
        axs_cum.plot(lnspc, kde_cum, "r--")

    if ERF_flag > 0:

        kde_cum_erf = np.cumsum(kdepdf_erf)
        kde_cum_erf /= np.amax(kde_cum_erf)
        axs_cum.plot(lnspc, kde_cum_erf, "--", color="tab:purple")

    if EW_flag > 0:

        kde_cum_EW = np.cumsum(kdepdf_EW)
        kde_cum_EW /= np.amax(kde_cum_EW)
        axs_cum.plot(lnspc, kde_cum_EW, "g--")

    axs_cum.set_xlim(xmin, xmax)
    axs_cum.set_ylabel("Cumulative", color="b")
    axs_cum.set_ylim(bottom=0)

    if global_log[j] == 1:

        axs_cum.set_xscale("log")

    plt.legend(legends, prop={"size": 18})

    if group == 0:

        plt.title("Target Question " + label_indexes[j], fontsize=18)

    else:

        plt.title("Target Question " + label_indexes[j] + " Group " +
                  str(group),
                  fontsize=18)

    figname = (output_dir + "/" + "Groups_PNGPDF" + "/" + elicitation_name +
               "_cum_group" + str(group) + "_" + str(j - n_SQ + 1).zfill(2) +
               ".pdf")
    fig_cum.savefig(figname)

    # images = convert_from_path(figname)
    figname = (output_dir + "/" + "Groups_PNGPDF" + "/" + elicitation_name +
               "_cum_group" + str(group) + "_" + str(j - n_SQ + 1).zfill(2) +
               ".png")
    # images[0].save(figname, "PNG")
    fig_cum.savefig(figname, dpi=300)

    plt.close()

    # create figure with PDFs only for small inset and groups
    fig2 = plt.figure()
    axs_h2 = fig2.add_subplot(111)
    axs_h2.set_xlabel(TQ_units[j - n_SQ])

    if Cooke_flag > 0:

        axs_h2.plot(lnspc, kdepdf, "r--", linewidth=2)

    if ERF_flag > 0:

        axs_h2.plot(lnspc, kdepdf_erf, "--", color="tab:purple", linewidth=2)

    if EW_flag > 0:

        axs_h2.plot(lnspc, kdepdf_EW, "g--", linewidth=2)

    axs_h2.set_xlim(xmin, xmax)

    if global_log[j] == 1:

        axs_h2.set_xscale("log")

    axs_h2.set_ylabel("PDF", color="b")

    axs_h2.set_ylim(bottom=0)
    plt.legend(legends, prop={"size": 18})

    if group == 0:

        plt.title("Target Question " + label_indexes[j], fontsize=18)

    else:

        plt.title("Target Question " + label_indexes[j] + " Group " +
                  str(group),
                  fontsize=18)

    figname = (output_dir + "/" + "Groups_PNGPDF" + "/" + elicitation_name +
               "_PDF_group" + str(group) + "_" + str(j - n_SQ + 1).zfill(2) +
               ".pdf")
    fig2.savefig(figname)

    # images = convert_from_path(figname)
    figname = (output_dir + "/" + "Groups_PNGPDF" + "/" + elicitation_name +
               "_PDF_group" + str(group) + "_" + str(j - n_SQ + 1).zfill(2) +
               ".png")
    # images[0].save(figname, "PNG")
    fig2.savefig(figname, dpi=300)

    plt.close()

    return


def create_figure_violin(
    count,
    violin_group,
    n_SQ,
    label_indexes,
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
):
    """Create figure for violin group (subset of target qustions)

    Parameters
    ----------
    count : int
        index of the group of questions
    violin_group : list of int
        indexes of questions for group
    n_SQ : int
        number of seed questions
    q_EW : numpy array of floats
        percentiles computed with equal weight
    q_Cooke : numpy array of floats
        percentiles computed with Cooke method
    q_erf : numpy array of floats
        percentiles computed with ERF method
    global_units : list of strings
        list of strings for units of answers to add as xlabel
    Cooke_flag : int
        Cooke_flag > 0 => plot the Cooke method results
    ERF_flag : int
        ERF_flag > 0 => plot the ERF method results
    EW_flag : int
        EW_flag > 0 => plot the equal weights results
    global_log : list of int
        1: log scale; 0: linear scale
    TQ_minVals : list of float
        minimum allowed value for answer to target questions
    TQ_maxVals
        maximum allowed value for answer to target questions
    output_dir : string
        name of output folder

    Returns
    -------
    none

    """

    if not os.path.exists(output_dir + "/" + "Violin_PNGPDF"):
        os.makedirs(output_dir + "/" + "Violin_PNGPDF")

    x = np.arange(len(violin_group)) + 1
    xmin = 0.5
    xmax = len(violin_group) + 0.5

    violin_group = np.array(violin_group)

    ncols = 0
    if EW_flag:
        EW_col = ncols
        ncols += 1
    if Cooke_flag:
        Cooke_col = ncols
        ncols += 1
    if ERF_flag:
        ERF_col = ncols
        ncols += 1

    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(8.0, 4.0))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    if EW_flag:

        y = samples_EW[:, violin_group]

        if (ncols > 1):

            vp_EW = axes[EW_col].violinplot(y,
                                            showmeans=False,
                                            showmedians=False,
                                            widths=0.25)

            axes[EW_col].set_title('EW')
            axes[EW_col].set_xticks(x)

        else:

            vp_EW = axes.violinplot(y,
                                    showmeans=False,
                                    showmedians=False,
                                    widths=0.25)
            axes.set_title('EW')
            axes.set_xticks(x)

        xtick = []
        for i in violin_group:
            xtick.append("TQ" + label_indexes[i])

        if (ncols > 1):

            axes[EW_col].set_xticklabels(xtick)

            axes[EW_col].set_xlim(xmin, xmax)
            axes[EW_col].set_ylim(TQ_minVals[violin_group[0] - n_SQ],
                                  TQ_maxVals[violin_group[0] - n_SQ])

        else:

            axes.set_xticklabels(xtick)

            axes.set_xlim(xmin, xmax)
            axes.set_ylim(TQ_minVals[violin_group[0] - n_SQ],
                          TQ_maxVals[violin_group[0] - n_SQ])

        # Set the color of the violin patches
        for pc in vp_EW['bodies']:
            pc.set_facecolor('g')
            pc.set_edgecolor('g')
            pc.set_linewidth(0.25)

        # Make all the violin statistics marks green:
        for partname in ('cbars', 'cmins', 'cmaxes'):
            vp = vp_EW[partname]
            vp.set_edgecolor('g')

        y = q_EW[violin_group, 1]
        lower_error = q_EW[violin_group, 1] - \
            q_EW[violin_group, 0]
        upper_error = q_EW[violin_group, 2] - \
            q_EW[violin_group, 1]
        # asymmetric_error = [lower_error, upper_error]

        # line1 = axes[EW_col].errorbar(x,
        #                             y,
        #                             yerr=asymmetric_error,
        #                             fmt="go",
        #                             markersize='4',
        #                             label="EW")

        if (ncols > 1):

            axes[EW_col].plot(x, y - lower_error, "gx", markersize='4')
            axes[EW_col].plot(x, y + upper_error, "gx", markersize='4')

        else:

            axes.plot(x, y - lower_error, "gx", markersize='4')
            axes.plot(x, y + upper_error, "gx", markersize='4')

    if Cooke_flag > 0:

        y = samples[:, violin_group]

        vp_Cooke = axes[Cooke_col].violinplot(y,
                                              showmeans=False,
                                              showmedians=False,
                                              widths=0.25)

        axes[Cooke_col].set_title('Cooke')

        axes[Cooke_col].set_xticks(x)

        xtick = []
        for i in violin_group:
            xtick.append("TQ" + label_indexes[i])

        axes[Cooke_col].set_xticklabels(xtick)

        axes[Cooke_col].set_xlim(xmin, xmax)
        axes[Cooke_col].set_ylim(TQ_minVals[violin_group[0] - n_SQ],
                                 TQ_maxVals[violin_group[0] - n_SQ])

        # Set the color of the violin patches
        for pc in vp_Cooke['bodies']:
            pc.set_facecolor('r')
            pc.set_edgecolor('r')
            pc.set_linewidth(0.25)

        # Make all the violin statistics marks red:
        for partname in ('cbars', 'cmins', 'cmaxes'):
            vp = vp_Cooke[partname]
            vp.set_edgecolor('r')

        y = q_Cooke[violin_group, 1]
        lower_error = q_Cooke[violin_group, 1] - \
            q_Cooke[violin_group, 0]
        upper_error = q_Cooke[violin_group, 2] - \
            q_Cooke[violin_group, 1]
        # asymmetric_error = [lower_error, upper_error]

        # line1 = axes[Cooke_col].errorbar(x,
        #                                y,
        #                                yerr=asymmetric_error,
        #                                fmt="ro",
        #                                markersize='4',
        #                                label="EW")
        axes[Cooke_col].plot(x, y - lower_error, "rx", markersize='4')
        axes[Cooke_col].plot(x, y + upper_error, "rx", markersize='4')

    if ERF_flag > 0:

        y = samples_erf[:, violin_group]

        vp_ERF = axes[ERF_col].violinplot(y,
                                          showmeans=False,
                                          showmedians=False,
                                          widths=0.25)

        axes[ERF_col].set_title('ERF')

        axes[ERF_col].set_xticks(x)

        xtick = []
        for i in violin_group:
            xtick.append("TQ" + label_indexes[i])

        axes[ERF_col].set_xticklabels(xtick)

        axes[ERF_col].set_xlim(xmin, xmax)
        axes[ERF_col].set_ylim(TQ_minVals[violin_group[0] - n_SQ],
                               TQ_maxVals[violin_group[0] - n_SQ])

        # Set the color of the violin patches
        for pc in vp_ERF['bodies']:
            pc.set_facecolor('b')
            pc.set_edgecolor('b')
            pc.set_linewidth(0.25)

        # Make all the violin statistics marks blue:
        for partname in ('cbars', 'cmins', 'cmaxes'):
            vp = vp_ERF[partname]
            vp.set_edgecolor('b')

        y = q_erf[violin_group, 1]
        lower_error = q_erf[violin_group, 1] - \
            q_erf[violin_group, 0]
        upper_error = q_erf[violin_group, 2] - \
            q_erf[violin_group, 1]
        # asymmetric_error = [lower_error, upper_error]

        # line1 = axes[ERF_col].errorbar(x,
        #                              y,
        #                              yerr=asymmetric_error,
        #                              fmt="bo",
        #                              markersize='4',
        #                              label="EW")
        axes[ERF_col].plot(x, y - lower_error, "bx", markersize='4')
        axes[ERF_col].plot(x, y + upper_error, "bx", markersize='4')

    # axs.grid(linewidth=0.4)

    figname = (output_dir + "/" + "Violin_PNGPDF" + "/" + elicitation_name +
               "_violin_" + str(count + 1).zfill(2) + ".pdf")
    fig.savefig(figname)

    # images = convert_from_path(figname)
    figname = (output_dir + "/" + "Violin_PNGPDF" + "/" + elicitation_name +
               "_violin_" + str(count + 1).zfill(2) + ".png")
    # images[0].save(figname, "PNG")
    fig.savefig(figname, dpi=300)
    plt.close()

    return


def create_figure_index(
    count,
    index_group,
    n_SQ,
    label_indexes,
    indexMean_EW,
    indexStd_EW,
    indexMean_Cooke,
    indexStd_Cooke,
    indexMean_erf,
    indexStd_erf,
    indexQuantiles_EW,
    indexQuantiles_Cooke,
    indexQuantiles_erf,
    global_units,
    Cooke_flag,
    ERF_flag,
    EW_flag,
    global_log,
    output_dir,
    elicitation_name,
):
    """Create figure for trend group (subset of target qustions)

    Parameters
    ----------
    count : int
        index of the group of questions
    trend_group : list of int
        indexes of questions for group
    n_SQ : int
        number of seed questions
    q_EW : numpy array of floats
        percentiles computed with equal weight
    q_Cooke : numpy array of floats
        percentiles computed with Cooke method
    q_erf : numpy array of floats
        percentiles computed with ERF method
    global_units : list of strings
        list of strings for units of answers to add as xlabel
    Cooke_flag : int
        Cooke_flag > 0 => plot the Cooke method results
    ERF_flag : int
        ERF_flag > 0 => plot the ERF method results
    EW_flag : int
        EW_flag > 0 => plot the equal weights results
    global_log : list of int
        1: log scale; 0: linear scale
    TQ_minVals : list of float
        minimum allowed value for answer to target questions
    TQ_maxVals
        maximum allowed value for answer to target questions
    output_dir : string
        name of output folder

    Returns
    -------
    none

    """

    if not os.path.exists(output_dir + "/" + "Index_PNGPDF"):
        os.makedirs(output_dir + "/" + "Index_PNGPDF")

    fig = plt.figure()
    axs = fig.add_subplot(111)

    y = -np.arange(len(index_group))

    handles = []

    index_group = np.array(index_group)

    if EW_flag:

        x = indexMean_EW[index_group-n_SQ]
        lower_error = indexStd_EW[index_group-n_SQ]
        upper_error = indexStd_EW[index_group-n_SQ]
        error = [lower_error, upper_error]

        """
        x = indexQuantiles_EW[index_group - n_SQ, 1]
        lower_error = x - indexQuantiles_EW[index_group - n_SQ, 0]
        upper_error = indexQuantiles_EW[index_group - n_SQ, 2] - x
        error = [lower_error, upper_error]
        """

        # line1 = axs.errorbar(x, y - 0.1, xerr=error, fmt="go", label="EW")
        line1 = axs.errorbar(x, y - 0.1, xerr=error, fmt="g*", label="EW")
        axs.plot(x - lower_error, y - 0.1, "gx")
        axs.plot(x + upper_error, y - 0.1, "gx")
        # axs.plot(indexMean_EW[index_group - n_SQ], y - 0.1, "g*")
        # axs.plot(indexQuantiles_EW[index_group - n_SQ, 1], y - 0.1, "go")

        handles.append(line1)

    if Cooke_flag > 0:

        x = indexMean_Cooke[index_group-n_SQ]
        lower_error = indexStd_Cooke[index_group-n_SQ]
        upper_error = indexStd_Cooke[index_group-n_SQ]
        error = [lower_error, upper_error]

        """
        x = indexQuantiles_Cooke[index_group - n_SQ, 1]
        lower_error = x - indexQuantiles_Cooke[index_group - n_SQ, 0]
        upper_error = indexQuantiles_Cooke[index_group - n_SQ, 2] - x
        error = [lower_error, upper_error]
        """

        # line2 = axs.errorbar(x, y, xerr=error, fmt="go", label="CM")
        line2 = axs.errorbar(x, y, xerr=error, fmt="r*", label="CM")
        axs.plot(x - lower_error, y, "rx")
        axs.plot(x + upper_error, y, "rx")
        # axs.plot(indexMean_Cooke[index_group - n_SQ], y, "r*")
        # axs.plot(indexQuantiles_Cooke[index_group - n_SQ, 1], y, "go")

        handles.append(line2)

    if ERF_flag > 0:

        x = indexMean_erf[index_group-n_SQ]
        lower_error = indexStd_erf[index_group-n_SQ]
        upper_error = indexStd_erf[index_group-n_SQ]
        error = [lower_error, upper_error]

        """
        x = indexQuantiles_erf[index_group - n_SQ, 1]
        lower_error = x - indexQuantiles_erf[index_group - n_SQ, 0]
        upper_error = indexQuantiles_erf[index_group - n_SQ, 2] - x
        error = [lower_error, upper_error]
        """

        # line3 = axs.errorbar(x, y + 0.1, xerr=error, fmt="bo", label="ERF")
        line3 = axs.errorbar(x, y + 0.1, xerr=error, fmt="b*", label="ERF")
        axs.plot(x - lower_error, y + 0.1, "bx")
        axs.plot(x + upper_error, y + 0.1, "bx")
        # axs.plot(indexMean_erf[index_group - n_SQ], y + 0.1, "b*")
        # axs.plot(indexQuantiles_erf[index_group - n_SQ, 1], y + 0.1, "bo")

        handles.append(line3)

    axs.set_yticks(y)
    axs.grid(axis='x')

    ytick = []
    for i in index_group:
        ytick.append("TQ" + label_indexes[i])

    axs.set_yticklabels(ytick)

    axs.set_xlim(-1.0, 1.0)

    plt.title("Question Group " + str(count + 1))

    axs.legend(handles=handles)

    figname = (output_dir + "/" + "Index_PNGPDF" + "/" + elicitation_name +
               "_index_" + str(count + 1).zfill(2) + ".pdf")
    fig.savefig(figname)

    # images = convert_from_path(figname)
    figname = (output_dir + "/" + "Index_PNGPDF" + "/" + elicitation_name +
               "_index_" + str(count + 1).zfill(2) + ".png")
    # images[0].save(figname, "PNG")
    fig.savefig(figname, dpi=300)
    plt.close()

    return


def create_figure_trend(
    count,
    trend_group,
    n_SQ,
    label_indexes,
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
):
    """Create figure for trend group (subset of target qustions)

    Parameters
    ----------
    count : int
        index of the group of questions
    trend_group : list of int
        indexes of questions for group
    n_SQ : int
        number of seed questions
    q_EW : numpy array of floats
        percentiles computed with equal weight
    q_Cooke : numpy array of floats
        percentiles computed with Cooke method
    q_erf : numpy array of floats
        percentiles computed with ERF method
    global_units : list of strings
        list of strings for units of answers to add as xlabel
    Cooke_flag : int
        Cooke_flag > 0 => plot the Cooke method results
    ERF_flag : int
        ERF_flag > 0 => plot the ERF method results
    EW_flag : int
        EW_flag > 0 => plot the equal weights results
    global_log : list of int
        1: log scale; 0: linear scale
    TQ_minVals : list of float
        minimum allowed value for answer to target questions
    TQ_maxVals
        maximum allowed value for answer to target questions
    output_dir : string
        name of output folder

    Returns
    -------
    none

    """

    if not os.path.exists(output_dir + "/" + "Trend_PNGPDF"):
        os.makedirs(output_dir + "/" + "Trend_PNGPDF")

    fig = plt.figure()
    axs = fig.add_subplot(111)

    x = np.arange(len(trend_group)) + 1

    handles = []

    trend_group = np.array(trend_group)

    if EW_flag:

        y = q_EW[trend_group, 1]
        lower_error = q_EW[trend_group, 1] - \
            q_EW[trend_group, 0]
        upper_error = q_EW[trend_group, 2] - \
            q_EW[trend_group, 1]
        asymmetric_error = [lower_error, upper_error]

        line1 = axs.errorbar(x - 0.1,
                             y,
                             yerr=asymmetric_error,
                             fmt="go",
                             label="EW")
        axs.plot(x - 0.1, y - lower_error, "gx")
        axs.plot(x - 0.1, y + upper_error, "gx")
        handles.append(line1)

    if Cooke_flag > 0:

        y = q_Cooke[trend_group, 1]
        lower_error = (q_Cooke[trend_group, 1] - q_Cooke[trend_group, 0])
        upper_error = (q_Cooke[trend_group, 2] - q_Cooke[trend_group, 1])
        asymmetric_error = [lower_error, upper_error]

        line2 = axs.errorbar(x, y, yerr=asymmetric_error, fmt="ro", label="CM")
        axs.plot(x, y - lower_error, "rx")
        axs.plot(x, y + upper_error, "rx")
        handles.append(line2)

    if ERF_flag > 0:

        y = q_erf[trend_group, 1]
        lower_error = (q_erf[trend_group, 1] - q_erf[trend_group, 0])
        upper_error = (q_erf[trend_group, 2] - q_erf[trend_group, 1])
        asymmetric_error = [lower_error, upper_error]

        line3 = axs.errorbar(x + 0.1,
                             y,
                             yerr=asymmetric_error,
                             fmt="bo",
                             label="ERF")
        axs.plot(x + 0.1, y - lower_error, "bx")
        axs.plot(x + 0.1, y + upper_error, "bx")
        handles.append(line3)

    axs.set_xticks(x)

    xtick = []
    for i in trend_group:
        xtick.append("TQ" + label_indexes[i])

    axs.set_xticklabels(xtick)

    # ax1.set_yscale('log')

    axs.set_ylim(TQ_minVals[trend_group[0] - n_SQ],
                 TQ_maxVals[trend_group[0] - n_SQ])

    # axs.grid(linewidth=0.4)

    plt.title("Question Group " + str(count + 1))

    axs.legend(handles=handles)

    figname = (output_dir + "/" + "Trend_PNGPDF" + "/" + elicitation_name +
               "_trend_" + str(count + 1).zfill(2) + ".pdf")
    fig.savefig(figname)

    # images = convert_from_path(figname)
    figname = (output_dir + "/" + "Trend_PNGPDF" + "/" + elicitation_name +
               "_trend_" + str(count + 1).zfill(2) + ".png")
    # images[0].save(figname, "PNG")
    fig.savefig(figname, dpi=300)
    plt.close()

    return


def create_figure_pie(count, pie_group, n_SQ, label_indexes, q_EW, q_Cooke,
                      q_erf, Cooke_flag, ERF_flag, EW_flag, output_dir,
                      elicitation_name):
    """Create figure for trend group (subset of target qustions)

    Parameters
    ----------
    trend_group : list of int
        indexes of questions for group
    n_SQ : int
        number of seed questions
    q_EW : numpy array of floats
        percentiles computed with equal weight
    q_Cooke : numpy array of floats
        percentiles computed with Cooke method
    q_erf : numpy array of floats
        percentiles computed with ERF method
    Cooke_flag : int
        Cooke_flag > 0 => plot the Cooke method results
    ERF_flag : int
        ERF_flag > 0 => plot the ERF method results
    EW_flag : int
        EW_flag > 0 => plot the equal weights results
    output_dir : string
        name of output folder

    Returns
    -------
    none

    """

    if not os.path.exists(output_dir + "/" + "Piecharts_PNGPDF"):
        os.makedirs(output_dir + "/" + "Piecharts_PNGPDF")

    ncols = 0
    if EW_flag:
        EW_col = ncols
        ncols += 1
    if Cooke_flag:
        Cooke_col = ncols
        ncols += 1
    if ERF_flag:
        ERF_col = ncols
        ncols += 1

    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(8.0, 4.0))
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    # x = np.arange(len(pie_group)) + 1

    labels = []
    for i in pie_group:
        labels.append("TQ" + label_indexes[i])

    pie_group = np.array(pie_group)

    if EW_flag:

        y = q_EW[pie_group, 3]

        sizes = y

        if ncols > 1:

            axes[EW_col].pie(sizes, labels=labels, autopct='%1.1f%%')
            axes[EW_col].set_title('EW')

        else:

            axes.pie(sizes, labels=labels, autopct='%1.1f%%')
            axes.set_title('EW')

    if Cooke_flag > 0:

        y = q_Cooke[pie_group, 3]

        sizes = y

        if ncols > 1:

            axes[Cooke_col].pie(sizes, labels=labels, autopct='%1.1f%%')
            axes[Cooke_col].set_title('Cooke')

        else:

            axes.pie(sizes, labels=labels, autopct='%1.1f%%')
            axes.set_title('Cooke')

    if ERF_flag > 0:

        y = q_erf[pie_group, 3]

        sizes = y

        if ncols > 1:

            axes[ERF_col].pie(sizes, labels=labels, autopct='%1.1f%%')
            axes[ERF_col].set_title('ERF')

        else:

            axes.pie(sizes, labels=labels, autopct='%1.1f%%')
            axes.set_title('ERF')

    figname = (output_dir + "/" + "Piecharts_PNGPDF" + "/" + elicitation_name +
               "_pie_" + str(count + 1).zfill(2) + ".pdf")
    fig.savefig(figname)

    # images = convert_from_path(figname)
    figname = (output_dir + "/" + "Piecharts_PNGPDF" + "/" + elicitation_name +
               "_pie_" + str(count + 1).zfill(2) + ".png")
    # images[0].save(figname, "PNG")
    fig.savefig(figname, dpi=300)
    plt.close()

    return


def create_figure_answers(h, k, n_experts, max_len_plot, n_SQ, SQ_array,
                          TQ_array, realization, analysis, Cooke_flag,
                          ERF_flag, EW_flag, global_units, output_dir, q_Cooke,
                          q_erf, q_EW, elicitation_name, global_log,
                          label_indexes, nolabel_flag):

    if not os.path.exists(output_dir + "/" + "Itemwise_PNGPDF"):
        os.makedirs(output_dir + "/" + "Itemwise_PNGPDF")
    idx0 = k * max_len_plot
    idx1 = min((k + 1) * max_len_plot, n_experts)

    if h >= n_SQ:

        j = h - n_SQ
        Q_array = TQ_array[idx0:idx1, :, j]
        string = "Target"
        string_title = string
        # string_title = "%.2E, " % indexMean[j] + "%.2E, " % indexStd[j] + \
        # "Target"

        xmin = np.amin(TQ_array[:, 0, j])
        xmax = np.amax(TQ_array[:, 2, j])

    else:

        j = h
        Q_array = SQ_array[idx0:idx1, :, j]
        string = "Seed"
        string_title = string

        xmin = np.amin(SQ_array[:, 0, j])
        xmin = np.minimum(xmin, realization[h])
        xmax = np.amax(SQ_array[:, 2, j])
        xmax = np.maximum(xmax, realization[h])

    if global_log[h] == 1:

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
    axs.errorbar(x, y, xerr=x_error, fmt="bo")
    axs.plot(x - x_errormin, y, "bx")
    axs.plot(x + x_errormax, y, "bx")

    ytick = []
    for i in y:
        if nolabel_flag:
            ytick.append("")
        else:
            ytick.append("Exp." + str(int(i + idx0)))

    yerror = idx1 - idx0
    if analysis:

        if Cooke_flag > 0:

            yerror = yerror + 1
            axs.errorbar(
                q_Cooke[h, 1],
                yerror,
                xerr=[[q_Cooke[h, 1] - q_Cooke[h, 0]],
                      [q_Cooke[h, 2] - q_Cooke[h, 1]]],
                fmt="ro",
            )
            axs.plot(q_Cooke[h, 0], yerror, "rx")
            axs.plot(q_Cooke[h, 2], yerror, "rx")
            axs.plot(q_Cooke[h, 3], yerror, "r*")

            ytick.append("DM-Cooke")

            if global_log[h] == 1:

                txt_Cooke = "%.2E" % q_Cooke[h, 1]

            elif q_Cooke[h, 2] < 0.01:
                txt_Cooke = "%5.3e" % q_Cooke[h, 1]
            elif q_Cooke[h, 2] > 999:
                txt_Cooke = "%5.2e" % q_Cooke[h, 1]
            else:
                txt_Cooke = "%6.2f" % q_Cooke[h, 1]

            if h >= n_SQ:
                axs.annotate(txt_Cooke, (q_Cooke[h, 1] * 0.99, yerror + 0.15))

        if ERF_flag > 0:

            yerror = yerror + 1
            axs.errorbar(
                q_erf[h, 1],
                [yerror],
                xerr=[[q_erf[h, 1] - q_erf[h, 0]],
                      [q_erf[h, 2] - q_erf[h, 1]]],
                fmt="o",
                color="tab:purple",
            )
            axs.plot(q_erf[h, 0], yerror, "x", color="tab:purple")
            axs.plot(q_erf[h, 2], yerror, "x", color="tab:purple")
            axs.plot(q_erf[h, 3], yerror, "*", color="tab:purple")

            ytick.append("DM-ERF")

            if global_log[h] == 1:

                txt_erf = "%.2E" % q_erf[h, 1]

            elif q_erf[h, 2] < 0.01:
                txt_erf = "%5.3e" % q_erf[h, 1]
            elif q_erf[h, 2] > 999:
                txt_erf = "%5.2e" % q_erf[h, 1]
            else:
                txt_erf = "%6.2f" % q_erf[h, 1]

            if h >= n_SQ:
                axs.annotate(txt_erf, (q_erf[h, 1] * 0.99, yerror + 0.15))

        if EW_flag > 0:

            yerror = yerror + 1
            axs.errorbar(
                [q_EW[h, 1]],
                [yerror],
                xerr=[[q_EW[h, 1] - q_EW[h, 0]], [q_EW[h, 2] - q_EW[h, 1]]],
                fmt="go",
            )

            axs.plot(q_EW[h, 0], yerror, "gx")
            axs.plot(q_EW[h, 2], yerror, "gx")
            axs.plot(q_EW[h, 3], yerror, "g*")

            ytick.append("DM-Equal")

            if global_log[h] == 1:

                txt_EW = "%.2E" % q_EW[h, 1]

            elif q_EW[h, 2] < 0.01:
                txt_EW = "%5.3e" % q_EW[h, 1]
            elif q_EW[h, 2] > 999:
                txt_EW = "%5.2e" % q_EW[h, 1]
            else:
                txt_EW = "%6.2f" % q_EW[h, 1]

            if h >= n_SQ:
                axs.annotate(txt_EW, (q_EW[h, 1] * 0.99, yerror + 0.15))

        if h < n_SQ:

            if realization[j] > 999:
                txt = "%5.2e" % realization[h]
            else:
                txt = "%6.2f" % realization[h]

            yerror = yerror + 1
            axs.plot(realization[h], yerror, "kx")
            axs.annotate(txt, (realization[h] * 0.99, yerror + 0.15))

            ytick.append("Realization")

    else:

        if h < n_SQ:

            if realization[j] > 999:
                txt = "%5.2e" % realization[h]
            else:
                txt = "%6.2f" % realization[h]

            axs.plot(realization[h], idx1 - idx0 + 1, "kx")
            axs.annotate(txt, (realization[j] * 0.99, idx1 - idx0 + 1 + 0.15))
            ytick.append("Realization")

    y = np.arange(len(ytick)) + 1

    ytick_tuple = tuple(i for i in ytick)
    axs.set_yticks(y)

    axs.set_yticklabels(ytick_tuple)
    axs.set_xlabel(global_units[h])

    if global_log[h] == 1:

        axs.set_xscale("log")

    axs.set_ylim(0.5, len(ytick) + 1.0)
    axs.set_xlim(xmin, xmax)

    axs.grid(linewidth=0.4)

    plt.title(string_title + " Question " + label_indexes[h])
    figname = (output_dir + "/" + "Itemwise_PNGPDF" + "/" + elicitation_name +
               "_" + string + "_" + str(j + 1).zfill(2) + "_" +
               str(k + 1).zfill(2) + ".pdf")
    fig.savefig(figname)

    # images = convert_from_path(figname)
    figname = (output_dir + "/" + "Itemwise_PNGPDF" + "/" + elicitation_name +
               "_" + string + "_" + str(j + 1).zfill(2) + "_" +
               str(k + 1).zfill(2) + ".png")
    # images[0].save(figname, "PNG")
    fig.savefig(figname, dpi=300)
    plt.close()


def create_barplot(group, n_SQ, n_TQ, n_sample, global_log, global_minVal,
                   global_maxVal, global_units, TQ_units, label_indexes,
                   minval_all, maxval_all, ERF_flag, Cooke_flag, EW_flag,
                   hist_type, output_dir, elicitation_name, n_bins, q_Cooke,
                   q_erf, q_EW, samples, samples_erf, samples_EW):

    if not os.path.exists(output_dir + "/" + "Barplots_PNGPDF"):
        os.makedirs(output_dir + "/" + "Barplots_PNGPDF")

    del_rows = []
    keep_rows = []

    if Cooke_flag == 0:
        del_rows.append(int(0))
    else:
        keep_rows.append(int(0))

    if ERF_flag == 0:
        del_rows.append(int(1))
    else:
        keep_rows.append(int(1))

    if EW_flag == 0:
        del_rows.append(int(2))
    else:
        keep_rows.append(int(2))

    colors = ["tomato", "purple", "springgreen"]
    colors = [colors[index] for index in keep_rows]

    legends = ["CM", "ERF", "EW"]
    legends = [legends[index] for index in keep_rows]

    for j in np.arange(n_SQ, n_SQ + n_TQ):

        if (n_TQ > 1):

            printProgressBar(j - n_SQ, n_TQ - 1, prefix='      ')

        legendsPDF = []

        if Cooke_flag > 0:

            legendsPDF.append("CM" + "%9.2f" % q_Cooke[j, 0] +
                              "%9.2f" % q_Cooke[j, 1] +
                              "%9.2f" % q_Cooke[j, 2])
        if ERF_flag > 0:

            legendsPDF.append("ERF" + "%9.2f" % q_erf[j, 0] +
                              "%9.2f" % q_erf[j, 1] + "%9.2f" % q_erf[j, 2])

        legendsPDF.append("EW" + "%9.2f" % q_EW[j, 0] + "%9.2f" % q_EW[j, 1] +
                          "%9.2f" % q_EW[j, 2])

        create_fig_hist(group, j, n_sample, n_SQ, hist_type, samples[:, j],
                        samples_erf[:, j], samples_EW[:, j], colors, legends,
                        legendsPDF, global_units, Cooke_flag, ERF_flag,
                        EW_flag, global_log, global_minVal, global_maxVal,
                        output_dir, elicitation_name, del_rows, TQ_units,
                        label_indexes, minval_all, maxval_all, n_bins)

    return
