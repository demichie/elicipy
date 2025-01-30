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
seed = False
target = True
postprocessing = True

# flag to use the label column of the questionnaire instead of the index
label_flag = True

# Do not show the index of the experts in the bar plots
nolabel_flag = False

# target questions to be analyzed (comment if use all target questions)
# target_list = [3,4,5,11,12]

# seed questions to be analyzed (comment if use all seed questions)
# seed_list = [1,2,3,4,5,6,7,8]

# sub-groups of experts to be analyzed (comment if not used)
# group_list = [5,6]

# groups of target questions for trend plots
trend_groups = [ [6,7,8,9,10] ]

# groups of target questions for violin plots
violin_groups = [ [6,7,8,9,10] ]

# groups of target questions for pie charts
pie_groups = [ [3,4,5],[6,7,8,9,10] ]

# groups of target questions for ELICIPY index plots
index_groups = [[1,2,3,4],[5,6,7,8]]

n_sample = 10000
n_bins = 10

# hist_type: 'bar' or 'step'
hist_type = 'bar'

# flag for normalization of group of target questions which has to sum to 1
normalizeSum = False

# EW_flag (equal weights):
# 0 - no EW
# 1 - EW
EW_flag = 1

# ERF_flag:
# 0 - no ERF
# 1 - ERF original
# 2 - ERF modified
ERF_flag = 0

# flag for Cooke (weights are computed when >0, and read from file
# when <0)
# 0 - no Cooke
# 1 - Cooke original
# 2 - Cooke balanced
# 3 - Cooke continuous
Cooke_flag = 1

# remove comment to read from file (only when Cooke_flag<0)
# weights_file = "weights.csv"

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



