elicitation_name = 'Cotopaxi'

# datarepo can be 'local' or 'github'
datarepo = 'local'

# if datarepo is 'github' you need user and token to access it
# user = ''
# github_token = ''

# Repository with data (used only when datarepo='github')
# Repository = ''

output_dir = 'OUTPUT'

# comment if there is ony one language
language = 'ENG'

analysis = True
target = True
postprocessing = True

# Do not show the index of the experts in the bar plots
nolabel_flag = False

# target_list = [1,2,3,5]
# seed_list = [1,2,3,4,5,6,7,8]

# group_list = [5,6]

n_sample = 10000
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

# flag for Cooke
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
trend_groups = [ [6,7,8,9,10] ]
