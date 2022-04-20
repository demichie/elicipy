def global_weights(SQ_array, TQ_array, realization, alpha, background_measure, k,
                   cal_power):

    import numpy as np
    from calculate_information import calculate_information
    from calscore import calscore

    """
    This function is based on the Matlab package ANDURIL
    Authors:  Georgios Leontaris and Oswaldo Morales-Napoles
    email:    G.Leontaris@tudelft.nl & O.MoralesNapoles@tudelft.nl
    """

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
    [I_real, I_tot] = calculate_information(SQ_array, TQ_array, realization, k,
                                            background_measure)
    # print('I_real, I_tot',I_real, I_tot)

    for ex in np.arange(E):
        C[ex] = calscore(M[ex, :], cal_power)

        W[ex, 0] = C[ex]
        W[ex, 1] = I_tot[ex, 0]
        W[ex, 2] = I_real[ex, 0]

        if C[ex] < alpha:
            ind = 0
        else:
            ind = 1

        w[ex] = C[ex] * I_real[ex] * ind

    # unNormalized weight
    W[:, 3] = w.T

    # Normalized weight
    norm_w = w / np.sum(w)

    W[:, 4] = norm_w.T

    return W
