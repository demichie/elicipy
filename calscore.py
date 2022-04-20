def calscore(M, cal_power):

    """
    This function is based on the Matlab package ANDURIL
    Authors:  Georgios Leontaris and Oswaldo Morales-Napoles
    email:    G.Leontaris@tudelft.nl & O.MoralesNapoles@tudelft.nl
    """

    import numpy as np
    from scipy.stats import chi2
    
    N = np.sum(M)  # number of seed items
    S = M / N

    if np.sum(np.isnan(S)) == len(S):
        CS = np.nan

    else:
        if len(S) == 4:
            P = np.array([5, 45, 45, 5]) / 100
        elif length(S) == 6:
            P = np.array([5, 20, 25, 25, 20, 5]) / 100

        E1 = S * np.log(S / P)
        MI = np.sum(E1[~np.isnan(E1)])
        E = 2 * N * MI * cal_power
        CS = chi2.sf(E, len(S) - 1)

    return CS
