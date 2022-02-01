def calscore(M, cal_power):

    import numpy as np
    from scipy.stats import chi2
    """
    %--------------------------------------------------------------------------
    % Description
    %--------------------------------------------------------------------------
    % This function calculates the statistical accuracy (or calibration score) 
    % of expert over the set of seed items.
    %--------------------------------------------------------------------------
    % Input(s)
    %--------------------------------------------------------------------------
    % M: An EΧB matrix that contains the number of the realizations captured in 
    % every bin that is formed by the quantiles provided by every expert e. 
    % Where E is the number of the experts and B the number of the bins.
    % 
    % Note: If this function is used to compute the calibration score of a DM 
    % that was obtained from a case where only one expert had a non-zero weight 
    % and one of the quantiles is exactly equal to the realization, attention 
    % should be paid to the calculation of matrix M. Due to precision of the 
    % calculating engine, it might occur that the resulted quantile from 
    % integrating the density has a minor difference with the initial 
    % assessment that will result in a different matrix M and subsequently a 
    % different calibration score. To solve this, the user could use digits or 
    % roundn MATLAB functions to set the precision of the obtained quantiles 
    % such that it is relevant for the values of the variables under 
    % consideration.
    %--------------------------------------------------------------------------
    % Output(s)
    %--------------------------------------------------------------------------
    % CS: A scalar CS with the statistical accuracy (or calibration score) of 
    % the expert over the set of seed items.
    %--------------------------------------------------------------------------
    % Last Update:  8-June-2018
    % Authors:  Georgios Leontaris and Oswaldo Morales-Napoles
    % email:    G.Leontaris@tudelft.nl & O.MoralesNapoles@tudelft.nl
    %--------------------------------------------------------------------------
    """

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
