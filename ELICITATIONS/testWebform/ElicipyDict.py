elicitation_name = 'test'

# datarepo can be 'local' or 'github'
datarepo = 'local'

# if datarepo is 'github' you need user and token to access it
# user = ''
# github_token = ''

# folder or repository with data
Repository = 'testWebform'

output_dir = 'OUTPUT'

# comment if there is ony one language
language = 'ENG'

analysis = True
target = True
postprocessing = True

# target_list = [2,3]
# seed_list = [1,2]

group_list = [0]

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

# Cooke flag:
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

# tree parameters
first_node_list = [1]
first_node_str_list = ['']

# groups for trend plots
# trend_groups = [ [2,3] ]
