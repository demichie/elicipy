def  item_weights(Cal_var, TQs, realization, alpha, back_measure, k,cal_power):

    import numpy as np
    from calscore import calscore
    
    """
    %--------------------------------------------------------------------------
    % Description
    %--------------------------------------------------------------------------
    % This function calculates the item weights of every expert e for every 
    % item. The main difference with the global weights weighting scheme is 
    % that the weights are different for every item. In this way the opinion of 
    % every expert has a different weight for every item. This is achieved by 
    % using the relative information of every particular item.
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
    % w_unorm: Unormalized weights. This EΧN matrix contains the weights of 
    % every expert for every seed item. Where E is the number of the experts
    % and N the number of the seed items.
    % W_item: A EΧN matrix with the normalized weights of every expert for 
    % every seed item.
    % W_item_tq: A EΧNtq matrix with the normalized weights of every expert 
    % for every target item. Where Ntq the number of target items.
    %--------------------------------------------------------------------------
    % Last Update:  8-June-2018 
    % Authors:  Georgios Leontaris and Oswaldo Morales-Napoles
    % email:    G.Leontaris@tudelft.nl & O.MoralesNapoles@tudelft.nl
    %--------------------------------------------------------------------------
    """

    p = [0.05, 0.45, 0.45, 0.05]
    N = Cal_var.shape[2]
    N_tq = TQs.shape[2]
    E = Cal_var.shape[0]

    N_P = len(p)

    suma = np.zeros((N,E))
    suma_tq = np.zeros((N_tq,E))
    x = np.zeros((E, N_P + 1))
    M = np.zeros((E,4))
    
    l = np.zeros(N)
    h = np.zeros(N)

    x_o = np.zeros(N)
    x_n = np.zeros(N)
    
    info_per_variable = np.zeros((N,E))
    C = np.zeros((E))
    w = np.zeros((E,N))
    W = np.zeros((E,N))

    
    for i in np.arange(N):
    
        l[i] = np.minimum(np.amin(Cal_var[:,:,i]),realization[i])
        h[i] = np.maximum(np.amax(Cal_var[:,:,i]),realization[i])
        
        for e in np.arange(E):
            
            if back_measure[i]=='uni':
            
                # only for the case of background measure 'uni'
                x_o[i] = l[i] - k*(h[i]-l[i])
                x_n[i] = h[i] + k*(h[i]-l[i])
                tmp = np.insert(Cal_var[e,:,i],0,x_o[i])
                tmp2 = np.append(tmp,x_n[i])
                x[e,:] = tmp2
                
            elif back_measure[i]=='log_uni':
            
                x_o[i] = np.log(l[i]) - k*(np.log(h[i])-np.log(l[i]))
                x_n[i] = np.log(h[i]) + k*(np.log(h[i])-np.log(l[i]))
                tmp = np.insert(np.log(Cal_var[e,:,i]),0,x_o[i])
                tmp2 = np.append(tmp,x_n[i])
                x[e,:] = tmp2
                
            else:
                print('Background measure is not correct! Choose from uni or log_uni')
            
            for j in np.arange(N_P):
                suma[i,e] = suma[i,e] + p[j] * np.log( p[j] / (x[e,j+1] - x[e,j]) )
            
            
            info_per_variable[i,e] = np.log(x_n[i] - x_o[i]) + suma[i,e]    
    
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


    for ex in np.arange(E):
        C[ex] = calscore(M[ex,:], cal_power)
    
        for it in np.arange(N):
            
            if C[ex] < alpha:
                ind = 0
            else:
                ind = 1
            
            w[ex,it] = C[ex]*info_per_variable[it,ex]*ind


    w_unorm = w

    #Normalized weight
    for g in np.arange(w.shape[1]):
        for z in np.arange(w.shape[0]):
            W[z,g] = w[z,g]/np.sum(w[:,g])


    # Calculate weights for target questions
    x_tq = np.zeros((E, N_P + 1))
    l_tq = np.zeros(N_tq)
    h_tq = np.zeros(N_tq)

    x_o_tq = np.zeros(N_tq)
    x_n_tq = np.zeros(N_tq)
    
    info_per_variable_tq = np.zeros((N_tq,E))
    w_tq = np.zeros((E,N_tq))
    W_tq = np.zeros((E,N_tq))
    
    
    for i in np.arange(N_tq):
    
        l_tq[i] = np.amin(TQs[:,:,i])
        h_tq[i] = np.amax(TQs[:,:,i])
    
        for e in np.arange(E):
        
            if back_measure[N+i]=='uni':
                # only for the case of background measure 'uni'
                x_o_tq[i] = l_tq[i] - k*(h_tq[i]-l_tq[i])
                x_n_tq[i] = h_tq[i] + k*(h_tq[i]-l_tq[i])
                tmp = np.insert(TQs[e,:,i],0,x_o_tq[i])
                tmp2 = np.append(tmp,x_n_tq[i])
                x_tq[e,:] = tmp2
                
            
            elif back_measure[N+i]=='log_uni':
                x_o_tq[i] = np.log(l_tq[i]) - k*(np.log(h_tq[i])-np.log(l_tq[i]))
                x_n_tq[i] = np.log(h_tq[i]) + k*(np.log(h_tq[i])-np.log(l_tq[i]))
                tmp = np.insert(np.log(TQs[e,:,i]),0,x_o_tq[i])
                tmp2 = np.append(tmp,x_n_tq[i])
                x_tq[e,:] = tmp2
                
            else:
                disp('Background measure is not correct! Choose from uni or log_uni')
        
            for j in np.arange(N_P):
                suma_tq[i,e] = suma_tq[i,e] + p[j] * np.log( p[j] / (x_tq[e,j+1] - x_tq[e,j]) )
        
            info_per_variable_tq[i,e] = np.log(x_n_tq[i] - x_o_tq[i]) + suma_tq[i,e] 



    for ex in np.arange(E):
        C[ex] = calscore(M[ex,:],cal_power)
    
        for it in np.arange(N_tq):
            
            if C[ex] < alpha:
                ind = 0
            else:
                ind = 1
        
            w_tq[ex,it] = C[ex]*info_per_variable_tq[it,ex]*ind


    w_unorm_tq = w_tq

    #Normalized weight
    for g in np.arange(w_tq.shape[1]):
        for z in np.arange(w_tq.shape[0]):
            W_tq[z,g] = w_tq[z,g]/np.sum(w_tq[:,g])

    return [w_unorm, W, W_tq]

