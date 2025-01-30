def COOKEweights(SQ_array, TQ_array, realization, alpha, background_measure,
                 overshoot, cal_power, Cooke_flag):
    """Compute the weights with Cooke formulation

    Parameters
    ----------
    SQ_array : numpy array
        Array with answers to seed questions
        Size: n_exp * n_pctl * n_seed
    TQ_array : numpy array
        Array with answers to target questions
        Size: n_exp * n_pctl * n_target
    realization : list
        Python list with realization of the seed questions
    alpha : float
        Significance
    background_measure:
        Python list
    overshoot : float
    cal_power : float
    Cooke_flag : integer

    Returns
    -------
    W : numpy array
        an array with weights

    This function is based on the Matlab package
    ANDURIL (Authors:  Georgios Leontaris and
    Oswaldo Morales-Napoles)
    """

    import numpy as np

    # number of Seed questions
    N = SQ_array.shape[2]
    # number of experts
    E = SQ_array.shape[0]
    # number of realization in every bin
    M = np.zeros((E, 4))
    # score
    C = np.zeros((E))
    # unNormalized weight
    w = np.zeros((E))
    # array with information, score and normalized weight for each expert
    W = np.zeros((E, 5))

    # create numpy array M with the number of realizations captured in every
    # expert's bin that is formed by the provided quantiles
    if Cooke_flag == 1:

        for ex in np.arange(E):
            for i in np.arange(N):
                if realization[i] <= SQ_array[ex, 0, i]:

                    M[ex, 0] += 1

                elif realization[i] <= SQ_array[ex, 1, i]:

                    M[ex, 1] += 1

                elif realization[i] <= SQ_array[ex, 2, i]:

                    M[ex, 2] += 1

                else:

                    M[ex, 3] += 1

    elif Cooke_flag == 2:

        for ex in np.arange(E):
            for i in np.arange(N):
                if realization[i] < SQ_array[ex, 0, i]:

                    M[ex, 0] += 1

                elif realization[i] == SQ_array[ex, 0, i]:

                    M[ex, 0] += 0.5
                    M[ex, 1] += 0.5

                elif realization[i] < SQ_array[ex, 1, i]:

                    M[ex, 1] += 1

                elif realization[i] == SQ_array[ex, 1, i]:

                    M[ex, 1] += 0.5
                    M[ex, 2] += 0.5

                elif realization[i] < SQ_array[ex, 2, i]:

                    M[ex, 2] += 1

                elif realization[i] == SQ_array[ex, 2, i]:

                    M[ex, 2] += 0.5
                    M[ex, 3] += 0.5

                else:

                    M[ex, 3] += 1

    elif Cooke_flag == 3:

        for ex in np.arange(E):

            for i in np.arange(N):

                if background_measure[i] == "uni":

                    val05 = SQ_array[ex, 0, i]
                    val50 = SQ_array[ex, 1, i]
                    val95 = SQ_array[ex, 2, i]
                    real = realization[i]

                else:

                    val05 = np.log(SQ_array[ex, 0, i])
                    val50 = np.log(SQ_array[ex, 1, i])
                    val95 = np.log(SQ_array[ex, 2, i])
                    real = np.log(realization[i])

                total_span = (val95 - val05) / 0.90
                interval_width = total_span / N

                real_min = real - 0.5 * interval_width
                real_max = real + 0.5 * interval_width

                M[ex, 0] += max(
                    0,
                    (min(val05, real_max) - real_min)) / (real_max - real_min)
                M[ex, 1] += max(0,
                                (min(val50, real_max) -
                                 max(val05, real_min))) / (real_max - real_min)
                M[ex, 2] += max(0,
                                (min(val95, real_max) -
                                 max(val50, real_min))) / (real_max - real_min)
                M[ex, 3] += max(
                    0,
                    (real_max - max(val95, real_min))) / (real_max - real_min)

    else:

        raise ValueError(
            "ERROR: Cooke_flag should be 1, 2 or 3",
            Cooke_flag,
        )

    # calculate calibration and information score for every expert
    [I_real, I_tot] = calculate_information(SQ_array, TQ_array, realization,
                                            overshoot, background_measure)

    for ex in np.arange(E):

        C[ex] = calscore(M[ex, :], cal_power)

        W[ex, 0] = C[ex]
        W[ex, 1] = I_tot[ex]
        W[ex, 2] = I_real[ex]

        w[ex] = C[ex] * I_real[ex]

    # unNormalized weight
    W[:, 3] = w.T

    # set to zero based on calibration score
    w[C < alpha] = 0.0

    # set to zero based on weight
    # w[ (w/np.sum(w)) < alpha] = 0.0

    # Normalized weight
    W[:, 4] = w / np.sum(w)

    return W, C, I_real, M


def calscore(M, cal_power):
    """Compute single expert score

    Parameters
    ----------
    M : numpy array
        Number of realizations captured in every expert's bin
        that is formed by the provided quantiles
    cal_power : float

    Returns
    -------
    CS : float
         Score

    This function is based on the Matlab package
    ANDURIL (Authors:  Georgios Leontaris and
    Oswaldo Morales-Napoles)
    """

    import numpy as np
    from scipy.stats import chi2

    N = np.sum(M)  # number of seed items
    S = M / N

    if np.sum(np.isnan(S)) == len(S):

        CS = np.nan

    else:
        if len(S) == 4:
            P = np.array([0.05, 0.45, 0.45, 0.05])
        elif len(S) == 6:
            P = np.array([0.05, 0.20, 0.25, 0.25, 0.20, 0.05])

        # E1 = S * np.log(S / P)
        # MI = np.sum(E1[~np.isnan(E1)])

        # New lines: 2023/12/12
        E1 = np.zeros_like(P)
        for i, Si in enumerate(S):
            if Si > 0:
                E1[i] = Si * np.log(Si / P[i])
        MI = np.sum(E1)

        E = 2.0 * N * MI * cal_power
        CS = chi2.sf(E, len(S) - 1)

    return CS


def calculate_information(SQ_array, TQ_array, realization, overshoot,
                          background_measure):
    """
    This function is based on the Matlab package
    ANDURIL (Authors:  Georgios Leontaris and
    Oswaldo Morales-Napoles)
    """

    import numpy as np

    p = [0.05, 0.45, 0.45, 0.05]
    N = SQ_array.shape[2]
    E = SQ_array.shape[0]
    N_P = len(p)
    suma = np.zeros((N, E))
    x = np.zeros((E, N_P + 1))

    Nq = TQ_array.shape[2]

    lowval = np.zeros(N + Nq)
    highval = np.zeros(N + Nq)

    x_o = np.zeros(N + Nq)
    x_n = np.zeros(N + Nq)

    info_per_variable = np.zeros((N + Nq, E))
    Info_score_real = np.zeros((E))
    Info_score_tot = np.zeros((E))

    # loop over seed questions
    for i in np.arange(N):

        # print('seed index', i)

        lowval[i] = np.minimum(np.amin(SQ_array[:, :, i]), realization[i])
        highval[i] = np.maximum(np.amax(SQ_array[:, :, i]), realization[i])

        # print('lowval,highval', lowval[i], highval[i])

        if background_measure[i] == "uni":

            x_o[i] = lowval[i] - overshoot * (highval[i] - lowval[i])
            x_n[i] = highval[i] + overshoot * (highval[i] - lowval[i])

        elif background_measure[i] == "log":

            if lowval[i] < 0 or highval[i] < 0:

                raise ValueError(
                    "Log-Uniform background measure cannot be used "
                    "for item %d because an expert provided "
                    "negative values",
                    i,
                )

            x_o[i] = np.log(lowval[i]) - overshoot * (np.log(highval[i]) -
                                                      np.log(lowval[i]))
            x_n[i] = np.log(highval[i]) + overshoot * (np.log(highval[i]) -
                                                       np.log(lowval[i]))

        else:

            raise ValueError(
                "ERROR: scale not uni or log",
                i,
            )

        # print('x_o,x_n', x_o[i], x_n[i])

        # loop over experts
        for e in np.arange(E):

            if background_measure[i] == "uni":

                tmp = np.insert(SQ_array[e, :, i], 0, x_o[i])
                tmp2 = np.append(tmp, x_n[i])
                x[e, :] = tmp2

            elif background_measure[i] == "log":

                tmp = np.insert(np.log(SQ_array[e, :, i]), 0, x_o[i])
                tmp2 = np.append(tmp, x_n[i])
                x[e, :] = tmp2

            for j in np.arange(N_P):

                suma[i, e] = suma[i, e] + p[j] * \
                    np.log(p[j] / (x[e, j+1] - x[e, j]))

            info_per_variable[i, e] = np.log(x_n[i] - x_o[i]) + suma[i, e]

    for e in np.arange(E):
        # for the overall information of every expert take the average of each
        # expert e for all calibration variables N
        Info_score_real[e] = np.sum(info_per_variable[:, e]) / N

    N_items_tot = SQ_array.shape[2] + TQ_array.shape[2]
    sumb = np.zeros((N_items_tot, E))

    for i in np.arange(SQ_array.shape[2], N_items_tot):

        if not realization[i]:  # information of Target Questions

            lowval[i] = np.amin(TQ_array[:, :, i - SQ_array.shape[2]])
            highval[i] = np.amax(TQ_array[:, :, i - SQ_array.shape[2]])

            if background_measure[i] == "uni":

                x_o[i] = lowval[i] - overshoot * (highval[i] - lowval[i])
                x_n[i] = highval[i] + overshoot * (highval[i] - lowval[i])

            elif background_measure[i] == "log":

                if lowval[i] < 0 or highval[i] < 0:

                    raise ValueError(
                        "Log-Uniform background measure cannot be used "
                        "for item %d because an expert provided "
                        "negative values",
                        i,
                    )

                x_o[i] = np.log(lowval[i]) - overshoot * (np.log(highval[i]) -
                                                          np.log(lowval[i]))
                x_n[i] = np.log(highval[i]) + overshoot * (np.log(highval[i]) -
                                                           np.log(lowval[i]))

            else:

                raise ValueError(
                    "ERROR: scale not uni or log",
                    i,
                )

            for e in np.arange(E):

                if background_measure[i] == "uni":

                    tmp = np.insert(TQ_array[e, :, i - SQ_array.shape[2]], 0,
                                    x_o[i])
                    tmp2 = np.append(tmp, x_n[i])
                    x[e, :] = tmp2

                elif background_measure[i] == "log":

                    tmp = np.insert(
                        np.log(TQ_array[e, :, i - SQ_array.shape[2]]), 0,
                        x_o[i])
                    tmp2 = np.append(tmp, x_n[i])
                    x[e, :] = tmp2

                for j in np.arange(N_P):

                    sumb[i - SQ_array.shape[2],
                         e] = sumb[i - SQ_array.shape[2],
                                   e] + p[j] * np.log(p[j] /
                                                      (x[e, j + 1] - x[e, j]))

                info_per_variable[i, e] = (np.log(x_n[i] - x_o[i]) +
                                           sumb[i - SQ_array.shape[2], e])

    # compute total information score
    for e in np.arange(E):

        # for the overall information of every expert take the average of each
        # expert e for all calibration variables N_items_tot
        Info_score_tot[e] = np.sum(info_per_variable[:, e]) / N_items_tot

    return Info_score_real, Info_score_tot
