[![SQAaaS badge](https://github.com/EOSC-synergy/SQAaaS/raw/master/badges/badges_150x116/badge_software_gold.png)](https://api.eu.badgr.io/public/assertions/egeaQ6tgQkWEtV3myK3SKQ "SQAaaS gold badge achieved")

[![SQAaaS badge shields.io](https://img.shields.io/badge/sqaaas%20software-gold-yellow)](https://api.eu.badgr.io/public/assertions/egeaQ6tgQkWEtV3myK3SKQ "SQAaaS gold badge achieved")

[![DOI](https://zenodo.org/badge/421933730.svg)](https://zenodo.org/badge/latestdoi/421933730)



# Elicipy

Elicipy is an Expert elicitation Python tool, aimed at both the collection of
anwers from the epxerts with a webform and their analysis.

The answers (for both seed and target questions) are collected with a
Streamlit webapp, that can run locally or from streamlit.io (read below).

The analysis is always done locally.

All the elicitation data, included the input files for the creation of the webapp and for analysis, are in subfolders of the ELICITATIONS folder. 
The files required to run an elicitation are:
1. createWebformDict.py
1. elicipyDict.py
1. DATA/questionnaire.csv 

When there are multiple elicitations folders inside ELICITATION, the elicitation for the Webapp is specified in the file ElicitationCase.py

___

## Webform

### Webapp hosted on streamlit.io

Here there are steps to install and run the webapp from streamlit.io.

On Github, you need two repositories, one to collect the answers and on hosting the webapp streamlit_app.py.



### GITHUB STREAMLIT REPOSITORY

1. To create the new repository for the answers, go to the you github main page, click on the tab "Repositories", and then on the green button "New", on the top-right above the list of your repositories. Write the "Repository name" for your asnwer repository, set this repository to "Private" if you don't want to show the answers, and finally click on "Create repository" at the bottom.
1. Go to the Elicipy repository (https://github.com/demichie/elicipy) and create a copy of the repository by clicking on the green button "Use this template" on the top-right and selecting "Create a new repository". Input the name of the new repository and click on the green button "Create repository" at bottom of the page. 
1. Edit in the input file createWebformDict.py the fields datarepo (set to 'github') and the RepositoryData (with the new github repository
   for the answers).
1. Set in the input file createWebformDict.py the quest_type variable to
   "seed" of "target".
1. Edit the csv file with your questions in the DATA folder of the github
   repository for the webapp (see the questionnaire.csv file for an example of
   the format).
1. In the ELICITATIONS folder, edit the file "ElicitationCase.py" with the elicitation folder name.
1. Click on the top-right (on your github user icon), and from the menu click
   on "Settings".
1. At the bottom of the left panel, click on "Developer settings".
1. On the left, click on "Personal access tokens".
1. Click on "Tokens (classic)".
1. Click on top on "Generate new token" and select (Generate new token (classic).
1. Give a name.
1. Select scope "repo".
1. At the end of the page click on "Generate token".
1. Copy the newly generated token. You will use it as Streamlit Secret.

### STREAMLIT

1. Login with github account.
1. On the top-right, click on "Create app".
1. Select "Yup, I have an app" to deploy the app from the github repository.
1. In the "Repository" field, select the github repository for the webapp (i.e. the repository you generated from the template).
1. Click on "Advanced settings".
1. Select Python version 3.9 or 3.10.
1. In the Secrets textbox write.

   github_token = "insert_here_your_token"

1. If you want to send a confirmation email after the answers are submitted, in the file createWebformDict.py set confirmation_mail = True and add the following lines in the Secrets texbox (fill with your email data). 

 SENDER_ADDRESS = ''

 SENDER_NAME = ''
 
 SENDER_PASSWORD = ''

 SMTP_SERVER_ADDRESS = ''
 
 PORT =

1. Click on "Save".
1. Click on "Deploy".

Now you should see your webform, and on the top-right you can click on "Share" to get the link.

### Webapp running locally with data saved locally

1. Edit in the input file createWebformDict.py the fields datarepo (set to 'local').
1. If you want to send a confirmation email after the answers are submitted, in the file createWebformDict.py set confirmation_mail = True and add the following lines (fill with your email data). 

 SENDER_ADDRESS = ''

 SENDER_NAME = ''
 
 SENDER_PASSWORD = ''

 SMTP_SERVER_ADDRESS = ''
 
 PORT =
1. Edit the file "ElicitationCase.py" with the elicitation folder name.
1. Start the webapp with:

> streamlit run streamlit_app.py

On your screen you will see these lines (with different web addresses):

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  
  Network URL: http://10.246.1.121:8501
```

The "Network URL" is the link to share. 



### Webapp running locally with data saved on Github

1. On Github, you need a repository to collect the answers (create a new
repository).
1. Click on the top-right (on your github user icon), and from the menu click
   on "Settings".
1. At the bottom of the left panel, click on "Developer settings".
1. On the left, click on "Personal access tokens".
1. Click on "Tokens (classic)".
1. Click on top on "Generate new token" and select (Generate new token (classic).
1. Give a name.
1. Select scope "Repo".
1. At the end of the page click on "Generate token".
1. Copy the newly generated token. 
1. On your computer, edit in the input file createWebformDict.py the fields datarepo (set to 'local_github') and fill the RepositoryData field (with the new github repository
   for the answers), the user filed and the github_token_field.
1. If you want to send a confirmation email after the answers are submitted, in the file createWebformDict.py set confirmation_mail = True and add the following lines (fill with your email data). 

 SENDER_ADDRESS = ''

 SENDER_NAME = ''
 
 SENDER_PASSWORD = ''

 SMTP_SERVER_ADDRESS = ''
 
 PORT =
   
1. Edit the file "ElicitationCase.py" with the elicitation folder name
1. Start the webapp with:

> streamlit run streamlit_app.py
 
On your screen you will see these lines (with different web addresses):

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://10.246.1.121:8501
```

The "Network URL" is the link to share. 

___

## Analysis

### Analysis with Webapp hosted on streamlit.io

1. Copy the elicitation folder containing DATA from the ELICITATIONS folder of the Webapp repository to the ELICITATIONS folder on your computer 
1. On your computer, in the file ElicipyDict.py, set datarepo = 'github' and RepositoryData = the name of the Github repository with the answers
1. Set the analysis parameter as desired.
1. run the Python analysis script:

> python elicipy.py

### Analysis with Webapp hosted locally and answers saved on Github
1. On your computer, in the file ElicipyDict.py, set datarepo = 'github' and RepositoryData = the name of the Github repository with the answers
1. Set the analysis parameter as desired.
1. run the Python analysis script:

> python elicipy.py


### Analysis with Webapp hosted locally and answers saved locally
1. In ElicipyDict.py set datarepo = 'local'
1. Set the analysis parameter as desired.
1. run the Python analysis script:

> python elicipy.py


The analysis results are saved in the folder output_dir (set in ElicipyDict.py)

___

Authors:

* Mattia de' Michieli Vitturi.
* Andrea Bevilacqua.
* Alessandro Tadini.
* Augusto Neri.

Some of the functions are based on the scripts of the Matlab package Anduril
(authors:  Georgios Leontaris and Oswaldo Morales-Napoles). The development of the code was supported by the project PIANETA DINAMICO, Istituto Nazionale di Geofisica e Vulcanologia, Italy.


