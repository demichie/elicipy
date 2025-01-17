# Add login page requesting password
password_protected = False


# select 'github', 'local' or 'local_github'
# datarepo = 'local'
datarepo = 'local'

# Github repository for answers, used only when datarepo='github' or 'local_github'
RepositoryData = 'test_answers'

# Parameters used only when datarepo='local' or 'local_github'
# user = ''
# github_token = ''

# this can be 'seed' or 'target'
quest_type = 'seed'

group_list = ['SG1 (Clermont)', 'SG2 (Quito)', 'SG3 (Geol.)', 'SG4 (MathMod)',
              'SG5 (Junior)', 'SG6 (Senior)']

# target_list = [2,3]
# seed_list = [1,2]
absolute_indexing = True

pctls = [5, 50, 95]
# companion_document = ""
# supplementary_documents = [""]

confirmation_email = False

# if confirmation email is True and datarepo is 'local' or 'local_github'
# fill the following fields, if 'github' add to streamlit
# secrets

# SENDER_ADDRESS
# SENDER_PASSWORD
# SENDER_NAME
# SMTP_SERVER_ADDRESS
# PORT
