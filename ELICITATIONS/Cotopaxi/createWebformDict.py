# select 'github' or 'local'
datarepo = 'local'

# datarepo = 'github'
# RepositoryData = 'createWebformAnswers'

Repository = 'testWebform'

# this can be 'seed' or 'target'
quest_type = 'target'

group_list = ['SG1 (Clermont)', 'SG2 (Quito)', 'SG3 (Geol.)', 'SG4 (MathMod)',
              'SG5 (Junior)', 'SG6 (Senior)']

# target_list = [2,3]
# seed_list = [1,2]
absolute_indexing = True

# encrypted = False

# user = 'username'
# github_token = "token"

pctls = [5, 50, 95]
# companion_document = ""
# supplementary_documents = [""]

confirmation_email = False

# if confirmation email is True and datarepo is 'local'
# fill the following fields, if 'github' add to streamlit
# secrets

# SENDER_ADDRESS
# SENDER_PASSWORD
# SENDER_NAME
# SMTP_SERVER_ADDRESS
# PORT
