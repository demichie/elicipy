elicitation_name = 'test'

input_dir = 'DATA3'
csv_file = 'questionnaire.csv'

output_dir = 'OUTPUT3'

analysis = True
target = False

n_sample = 5000
n_bins = 10

# hist_type: 'bar' or 'step'
hist_type = 'bar'

# EW_flag (equal weights):
# 0 - no EW
# 1 - EW 
EW_flag = 1

# ERF_flag: 
# 0 - no ERF
# 1 - ERF original 
# 2 - ERF modified
ERF_flag = 1

# 
Cooke_flag = 1

# parameters for Cooke

# significance level (this value cannot be higher than the
# highest calibration score of the pool of experts)
alpha = 0.05  

# overshoot for intrinsic range
overshoot = 0.1  

# global cal_power
# this value should be between [0.1, 1]. The default is 1.
cal_power = 1  


