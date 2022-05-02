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
        elif length(S) == 6:
            P = np.array([0.05, 0.20, 0.25, 0.25, 0.20, 0.05]) 

        E1 = S * np.log(S / P)
        MI = np.sum(E1[~np.isnan(E1)])
        E = 2.0 * N * MI * cal_power
        CS = chi2.sf(E, len(S) - 1)
        
    return CS

def calculate_information(SQ_array, TQ_array, realization, k, background_measure):

    """
    This function is based on the Matlab package 
    ANDURIL (Authors:  Georgios Leontaris and 
    Oswaldo Morales-Napoles)
    """

    import numpy as np

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
    Info_score_real = np.zeros((E))
    Info_score_tot = np.zeros((E))

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
        Info_score_real[e] = np.sum(info_per_variable[:, e]) / N

    N_items_tot = SQ_array.shape[2] + TQ_array.shape[2]
    sumb = np.zeros((N_items_tot, E))

    for i in np.arange(SQ_array.shape[2], N_items_tot):
    
        if not realization[i]:  # information of Target Questions

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

        # for the overall information of every expert take the average of each
        # expert e for all calibration variables N_items_tot
        Info_score_tot[e] = np.sum(info_per_variable[:, e]) / N_items_tot

    return Info_score_real, Info_score_tot

    
def global_weights(SQ_array, TQ_array, realization, alpha, background_measure,
                   overshoot, cal_power):

    """Compute the weights with Cooke formulation
    
    Parameters
    ----------
    SQ_array : numpy array
        Array with answers to seed questions
    TQ_array : numpy array
        Array with answers to seed questions
    realization : list
        Python list with realization of the seed questions
    alpha : float
        Significance
    background_measure:
        Python list
    overshoot : float
    cal_power : float
    
    Returns
    -------
    W : numpy array
        an array with weights
         
    This function is based on the Matlab package 
    ANDURIL (Authors:  Georgios Leontaris and 
    Oswaldo Morales-Napoles)
    """

    import numpy as np
    
    N = SQ_array.shape[2]
    E = SQ_array.shape[0]
    M = np.zeros((E, 4))
    C = np.zeros((E))
    w = np.zeros((E))

    W = np.zeros((E, 5))

    # create numpy array M with the number of realizations captured in every
    # expert's bin that is formed by the provided quantiles
    for ex in np.arange(E):
        for i in np.arange(N):
            if realization[i] <= SQ_array[ex, 0, i]:
                M[ex, 0] = M[ex, 0] + 1
            elif realization[i] <= SQ_array[ex, 1, i]:
                M[ex, 1] = M[ex, 1] + 1
            elif realization[i] <= SQ_array[ex, 2, i]:
                M[ex, 2] = M[ex, 2] + 1
            else:
                M[ex, 3] = M[ex, 3] + 1

    # calculate calibration and information score for every expert
    [I_real, I_tot] = calculate_information(SQ_array, TQ_array, realization, overshoot,
                                            background_measure)
    
    for ex in np.arange(E):
    
        C[ex] = calscore(M[ex, :], cal_power)

        W[ex, 0] = C[ex]
        W[ex, 1] = I_tot[ex]
        W[ex, 2] = I_real[ex]

        if C[ex] < alpha:
        
            w[ex] = 0
            
        else:
        
            w[ex] = C[ex] * I_real[ex]

    # unNormalized weight
    W[:, 3] = w.T

    # Normalized weight
    W[:, 4] = w / np.sum(w)

    return W
