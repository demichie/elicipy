def calculate_information(SQ_array, TQ_array, realization, k, background_measure):

    """
    This function is based on the Matlab package ANDURIL
    Authors:  Georgios Leontaris and Oswaldo Morales-Napoles
    email:    G.Leontaris@tudelft.nl & O.MoralesNapoles@tudelft.nl
    """

    import numpy as np
    from calscore import calscore

    per = [0.05, 0.5, 0.95]
    p = [0.05, 0.45, 0.45, 0.05]
    N = SQ_array.shape[2]
    E = SQ_array.shape[0]
    N_P = len(p)
    suma = np.zeros((N, E))
    x = np.zeros((E, N_P + 1))

    Nq = TQ_array.shape[2]

    l = np.zeros(N + Nq)
    h = np.zeros(N + Nq)

    x_o = np.zeros(N + Nq)
    x_n = np.zeros(N + Nq)

    info_per_variable = np.zeros((N + Nq, E))
    Info_score_real = np.zeros((E, 1))
    Info_score_tot = np.zeros((E, 1))

    for i in np.arange(N):

        l[i] = np.minimum(np.amin(SQ_array[:, :, i]), realization[i])
        h[i] = np.maximum(np.amax(SQ_array[:, :, i]), realization[i])

        for e in np.arange(E):

            if background_measure[i] == 'uni':

                x_o[i] = l[i] - k * (h[i] - l[i])
                x_n[i] = h[i] + k * (h[i] - l[i])

                tmp = np.insert(SQ_array[e, :, i], 0, x_o[i])
                tmp2 = np.append(tmp, x_n[i])
                x[e, :] = tmp2

            elif background_measure[i] == 'log':

                if l[i] < 0 or h[i] < 0:

                    raise ValueError(
                        'Log-Uniform background measure cannot be used for item %d because an expert provided negative values',
                        i)

                x_o[i] = np.log(l[i]) - k * (np.log(h[i]) - np.log(l[i]))
                x_n[i] = np.log(h[i]) + k * (np.log(h[i]) - np.log(l[i]))
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
        Info_score_real[e, 0] = np.sum(info_per_variable[:, e]) / N

    N_items_tot = SQ_array.shape[2] + TQ_array.shape[2]
    sumb = np.zeros((N_items_tot, E))

    for i in np.arange(SQ_array.shape[2], N_items_tot):
        if not realization[i]:  # information of Target Questions

            # print(i,i-SQ_array.shape[2])
            l[i] = np.amin(TQ_array[:, :, i - SQ_array.shape[2]])
            h[i] = np.amax(TQ_array[:, :, i - SQ_array.shape[2]])

            for e in np.arange(E):
                if background_measure[i] == 'uni':
                    x_o[i] = l[i] - k * (h[i] - l[i])
                    x_n[i] = h[i] + k * (h[i] - l[i])

                    tmp = np.insert(
                        TQ_array[e, :, i - SQ_array.shape[2]], 0, x_o[i])
                    tmp2 = np.append(tmp, x_n[i])
                    x[e, :] = tmp2

                elif background_measure[i] == 'log':
                    if l(i) < 0 or h(i) < 0:
                        raise ValueError(
                            'Log-Uniform background measure cannot be used for item %d because an expert provided negative values',
                            i)

                    x_o[i] = np.log(l[i]) - k * (np.log(h[i]) - np.log(l[i]))
                    x_n[i] = np.log(h(i)) + k * (np.log(h[i]) - np.log(l[i]))

                    tmp = np.insert(np.log(TQ_array[e, :, i - SQ_array.shape[2]]), 0,
                                    x_o[i])
                    tmp2 = np.append(tmp, x_n[i])
                    x[e, :] = tmp2

                for j in np.arange(N_P):
                    sumb[i - SQ_array.shape[2],
                         e] = sumb[i - SQ_array.shape[2],
                                   e] + p[j] * np.log(p[j] /
                                                      (x[e, j + 1] - x[e, j]))

                info_per_variable[i, e] = np.log(x_n[i] - x_o[i]) + sumb[
                    i - SQ_array.shape[2], e]

    # compute total information score
    for e in np.arange(E):

        # print('info',info_per_variable[:,e])
        # for the overall information of every expert take the average of each
        # expert e for all calibration variables N_items_tot
        Info_score_tot[e, 0] = np.sum(info_per_variable[:, e]) / N_items_tot

    return Info_score_real, Info_score_tot
