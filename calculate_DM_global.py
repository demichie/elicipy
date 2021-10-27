def calculate_DM_global(Cal_var,TQs, realization, w, k, back_measure, alpha):

    """
    %--------------------------------------------------------------------------
    % Description
    %--------------------------------------------------------------------------
    % This function calculates the distribution of the DM for every item, using 
    % the global weights or equal weights weighting schemes. In order to obtain
    % equal weight DM, a vector with equal weights for every expert should be 
    % given as an argument.
    %--------------------------------------------------------------------------
    % Input(s)
    %--------------------------------------------------------------------------
    % Cal_var:  A three-dimensional array that contains the assessments of the 
    % experts for every seed item.
    % TQs:  A three-dimensional array that contains the assessments of the 
    % experts for every target variable.
    % realization:  A cell array  that contains the realization of every seed 
    % question and as many empty cells ([]) as target variables
    % k: overshoot.
    % back_measure: A cell array with the background measure of every item.
    % alpha: Significance level. 
    %--------------------------------------------------------------------------
    % Output(s)
    %--------------------------------------------------------------------------
    % f_DM_out: A cell array that contains the density of the DM for values X 
    % concerning every item 
    % F_DM_out: A cell array that contains the cumulative probability of the DM
    % for values X_out of every item
    % X_out: A cell array that contains all the unique values provided by the
    % experts with non-zero weights
    % DM: A matrix DM with the quantiles of the obtained DM. This matrix has 
    % the ql, 5%, 50%, 95% and qh quantiles of the DMs distribution for every
    % item.
    % W_incl_VE: The table W_incl_VE. This is actually the table W updated with
    % the obtained DM (in the last row).
    %--------------------------------------------------------------------------
    % Last Update:  8-Dec-2017 
    % Authors:  Georgios Leontaris and Oswaldo Morales-Napoles
    % email:    G.Leontaris@tudelft.nl & O.MoralesNapoles@tudelft.nl
    %--------------------------------------------------------------------------
    """
    
    N_Cal_var = Cal_var.shape[2]
    N_tot_var = N_Cal_var + TQs.shape[2]
    DM = np.zeros((size(N_tot_var,3),size(Cal_var,2)+2))

    # calculate the DMs quantiles of the Seed Variables
    for i in np.arange(Cal_var.shape[2]):
    
        #     exp_tbc = find(w~=0); % expert(s) with non-zero weights
        #     l(i) = min(min(min(Cal_var(exp_tbc,:,i))),realization{i});
        #     h(i) = max(max(max(Cal_var(exp_tbc,:,i))),realization{i});
        
        l[i] = np.minimum(np.amin(Cal_var[:,:,i]),realization[i])
        h[i] = np.maximum(np.amax(Cal_var[:,:,i]),realization[i])
        
        
        if back_measure[i]=='uni':
        
            ql[i] = l[i] - k*(h[i]-l[i])
            qh[i] = h[i] + k*(h[i]-l[i])
            
            
            for x in np.arange(len(w)):
            
                str[x] = " {:d}*unif_dnes_v1(x,[{:d} {:d} {:d} {:d} {:d} ])".format(w[x], ql[i], Cal_var[x,0,i], Cal_var[x,1,i], Cal_var[x,2,i],qh[i])
            
            
            fin_str[0] = str[0]
            
            for y = 2:length(str)
            
                fin_str[y] = "{} + {}".format(fin_str[y-1], str[y])
                
            
            # the density of the decision maker:
            f_DM = str2func(sprintf('@(x) %s',fin_str{length(str)}));
            
            # Vector of experts' quantiles for every item
            quant = reshape(Cal_var(:,:,i),1,size(Cal_var,1)*size(Cal_var,2));
            
            X = unique(sort([ql(i) quant qh(i)]));
            
            # the integral of the DM density
            for z = 2:length(X)
                F_DM(z) = integral(f_DM,X(1),X(z));
            end
            
            q5_DM = interp1(F_DM,X,0.05);
            q50_DM = interp1(F_DM,X,0.5);
            q95_DM = interp1(F_DM,X,0.95);
            
            DM(i,:) = [ql(i); q5_DM; q50_DM; q95_DM; qh(i)];
            
            f_DM_out{i,:} = feval(f_DM,X);
            F_DM_out{i,:} = F_DM;
            X_out{i,:} = X;
            
            clear X F_DM f_DM;
            
        elseif strcmp(back_measure{i}, 'log_uni')
            ql(i) = log(l(i)) - k*(log(h(i))-log(l(i)));
            qh(i) = log(h(i)) + k*(log(h(i))-log(l(i)));
            
            
            for x = 1:length(w)
                str{x,1} = sprintf(' %d*unif_dnes_v1(x,[%d %d %d %d %d ])',w(x),ql(i),...
                    log(Cal_var(x,1,i)),log(Cal_var(x,2,i)),log(Cal_var(x,3,i)),qh(i));
            end
        
            fin_str{1} = str{1};
            
            for y = 2:length(str)
                fin_str{y} = sprintf('%s + %s',fin_str{y-1}, str{y});
            end
            
            % the density of the decision maker:
            
            f_DM = str2func(sprintf('@(x) %s',fin_str{length(str)}));
            
            % Vector of experts' quantiles for every item
            quant = reshape(Cal_var(:,:,i),1,size(Cal_var,1)*size(Cal_var,2));
            
            X = unique(sort([ql(i) log(quant) qh(i)]));
            
            % the integral of the DM density
            for z = 2:length(X)
                F_DM(z) = integral(f_DM,X(1),X(z));
            end
            
            q5_DM = interp1(F_DM,X,0.05);
            q50_DM = interp1(F_DM,X,0.5);
            q95_DM = interp1(F_DM,X,0.95);
            
            DM(i,:) = [exp(ql(i)); exp(q5_DM); exp(q50_DM); exp(q95_DM); exp(qh(i))];
            
            f_DM_out{i,:} = feval(f_DM,X);
            F_DM_out{i,:} = F_DM;
            X_out{i,:} = exp(X);
            clear X F_DM f_DM;
            
        else
            disp('Wrong background measure of item %d',i);
        end
        
    end
    
    % calculate the DMs quantiles of the Target Variables
    for i = 1:size(TQs,3)
        %     exp_tbc = find(w~=0); % expert(s) with non-zero weights
        %     l(i) = min(min(min(Cal_var(exp_tbc,:,i))),realization{i});
        %     h(i) = max(max(max(Cal_var(exp_tbc,:,i))),realization{i});
        l_tq(i) = min(min(TQs(:,:,i)));
        h_tq(i) = max(max(TQs(:,:,i)));
        
        if strcmp(back_measure{N_Cal_var+i}, 'uni')
            ql_tq(i) = l_tq(i) - k*(h_tq(i)-l_tq(i));
            qh_tq(i) = h_tq(i) + k*(h_tq(i)-l_tq(i));
            
            
            for x = 1:length(w)
                str_tq{x,1} = sprintf(' %d*unif_dnes_v1(x,[%d %d %d %d %d ])',w(x),ql_tq(i),...
                    TQs(x,1,i),TQs(x,2,i),TQs(x,3,i),qh_tq(i));
            end
            
            fin_str_tq{1} = str_tq{1};
            
            for y = 2:length(str_tq)
                fin_str_tq{y} = sprintf('%s + %s',fin_str_tq{y-1}, str_tq{y});
            end
            
            % the density of the decision maker:
            
            f_DM = str2func(sprintf('@(x) %s',fin_str_tq{length(str_tq)}));
            
            % Vector of experts' quantiles for every item
            quant_tq = reshape(TQs(:,:,i),1,size(TQs,1)*size(TQs,2));
            
            X = unique(sort([ql_tq(i) quant_tq qh_tq(i)]));
            
            % the integral of the DM density
            for z = 2:length(X)
                F_DM(z) = integral(f_DM,X(1),X(z));
            end
            
            q5_DM = interp1(F_DM,X,0.05);
            q50_DM = interp1(F_DM,X,0.5);
            q95_DM = interp1(F_DM,X,0.95);
            
            DM(N_Cal_var+i,:) = [ql_tq(i); q5_DM; q50_DM; q95_DM; qh_tq(i)];
            
            f_DM_out{N_Cal_var+i,:} = feval(f_DM,X);
            F_DM_out{N_Cal_var+i,:} = F_DM;
            X_out{N_Cal_var+i,:} = X;
            clear X F_DM f_DM;
            
        elseif strcmp(back_measure{N_Cal_var+i}, 'log_uni')
            ql_tq(i) = log(l_tq(i)) - k*(log(h_tq(i))-log(l_tq(i)));
            qh_tq(i) = log(h_tq(i)) + k*(log(h_tq(i))-log(l_tq(i)));
            
            
            for x = 1:length(w)
                str_tq{x,1} = sprintf(' %d*unif_dnes_v1(x,[%d %d %d %d %d ])',w(x),ql_tq(i),...
                    log(TQs(x,1,i)),log(TQs(x,2,i)),log(TQs(x,3,i)),qh_tq(i));
            end
            
            fin_str_tq{1} = str_tq{1};
            
            for y = 2:length(str)
                fin_str_tq{y} = sprintf('%s + %s',fin_str_tq{y-1}, str_tq{y});
            end
            
            % the density of the decision maker:
            
            f_DM = str2func(sprintf('@(x) %s',fin_str_tq{length(str)}));
            
            % Vector of experts' quantiles for every item
            quant_tq = reshape(TQs(:,:,i),1,size(TQs,1)*size(TQs,2));
            
            X = unique(sort([ql_tq(i) log(quant_tq) qh_tq(i)]));
            
            % the integral of the DM density
            for z = 2:length(X)
                F_DM(z) = integral(f_DM,X(1),X(z));
            end
            
            q5_DM = interp1(F_DM,X,0.05);
            q50_DM = interp1(F_DM,X,0.5);
            q95_DM = interp1(F_DM,X,0.95);
            
            DM(N_Cal_var+i,:) = [exp(ql_tq(i)); exp(q5_DM); exp(q50_DM); exp(q95_DM); exp(qh_tq(i))];
            
            f_DM_out{N_Cal_var+i,:} = feval(f_DM,X);
            F_DM_out{N_Cal_var+i,:} = F_DM;
            X_out{N_Cal_var+i,:} = exp(X);
            clear X F_DM f_DM;
            
        else
            disp('Wrong background measure of item %d',i);
        end
    end

    % Update table W with the Calibration and Info of the virtual expert (DM)
    Cal_var_upd = zeros(size(Cal_var,1)+1,size(Cal_var,2),size(Cal_var,3));
    TQs_upd = zeros(size(TQs,1)+1,size(TQs,2),size(TQs,3));
    for i = 1:size(Cal_var,3)
        Cal_var_upd(:,:,i) = [Cal_var(:,:,i); DM(i,2:4)];  % add the virtual expert (DM)
    end
    for i = 1:size(TQs,3)
        TQs_upd(:,:,i) = [TQs(:,:,i); DM(i+size(Cal_var,3),2:4)];
    end
    W_incl_VE = global_weights(Cal_var_upd, TQs_upd, realization, alpha, back_measure, k)


    return [f_DM_out, F_DM_out, X_out, DM, W_incl_VE]
