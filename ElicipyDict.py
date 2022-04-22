elicitation_name = 'test'

input_dir = 'DATA'
csv_file = 'questionnaire.csv'

output_dir = 'OUTPUT2'

analysis = True
target = True

n_sample = 1000
n_bins = 10

# hist_type: 'bar' or 'step'
hist_type = 'bar'

EW_flag = True
ERF_flag = True
Cooke_flag = True

# parameters for Cooke

# significance level (this value cannot be higher than the
# highest calibration score of the pool of experts)
alpha = 0.05  
# overshoot for intrinsic rangek = 0.1  

# global cal_power
# this value should be between [0.1, 1]. The default is 1.
cal_power = 1  


