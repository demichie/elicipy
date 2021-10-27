def global_weights(Cal_var, TQs, realization, alpha, background_measure, k,cal_power):

    import numpy as np
    from calculate_information import calculate_information
    from calscore import calscore
    
    """
    %--------------------------------------------------------------------------
    % Description
    %--------------------------------------------------------------------------
    % This function calculates the calibration score, the information score 
    % over the seed items and subsequently the global weight of every expert e.
    %--------------------------------------------------------------------------
    % Input(s)
    %--------------------------------------------------------------------------
    % Cal_var:  A three-dimensional array that contains the assessments of the 
    % experts for every seed item.
    % TQs:  A three-dimensional array that contains the assessments of the 
    % experts for every target variable.
    % realization:  A cell array  that contains the realization of every seed 
    % question and as many empty cells ([]) as target variables
    % alpha: Significance level. 
    % back_measure: A cell array with the background measure of every item.
    % k: overshoot.
    %--------------------------------------------------------------------------
    % Output(s)
    %--------------------------------------------------------------------------
    % A table W with the calibration score (first column) the information score 
    % over all the items (second column), the information score over the seed 
    % items (third column), unnormalized weight (fourth column), normalized 
    % weight (fifth column) for every expert (in a different row of the table).
    %--------------------------------------------------------------------------
    % Last Update:  8-Dec-2017 
    % Authors:  Georgios Leontaris and Oswaldo Morales-Napoles
    % email:    G.Leontaris@tudelft.nl & O.MoralesNapoles@tudelft.nl
    %--------------------------------------------------------------------------
    """

    N = Cal_var.shape[2]
    E = Cal_var.shape[0]
    M = np.zeros((E,4))
    C = np.zeros((E))
    w = np.zeros((E))

    W = np.zeros((E,5))

    # create table M with the number of realizations captured in every expert's
    # bin that is formed by the provided quantiles
    for ex in np.arange(E):
        for i in np.arange(N):
            if realization[i] <= Cal_var[ex,0,i]:
                M[ex,0] = M[ex,0] + 1
            elif realization[i] <= Cal_var[ex,1,i]:
                M[ex,1] = M[ex,1] + 1
            elif realization[i] <= Cal_var[ex,2,i]:
                M[ex,2] = M[ex,2] + 1
            else:
                M[ex,3] = M[ex,3] + 1

    # calculate calibration and information score for every expert
    [I_real, I_tot] = calculate_information(Cal_var, TQs, realization, k, background_measure)
    # print('I_real, I_tot',I_real, I_tot)
    
    for ex in np.arange(E):
        C[ex] = calscore(M[ex,:],cal_power)
        
        W[ex,0] = C[ex]
        W[ex,1] = I_tot[ex,0]
        W[ex,2] = I_real[ex,0]
        
        if C[ex] < alpha:
            ind = 0
        else:
            ind = 1
        
        w[ex] = C[ex]*I_real[ex]*ind
    
    # unNormalized weight
    W[:,3] = w.T
        
    # Normalized weight
    norm_w = w/np.sum(w)
    
    W[:,4] = norm_w.T
    
    return W
