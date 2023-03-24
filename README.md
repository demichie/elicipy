# Elicipy

Elicipy is an Expert elicitation Python tool, aimed at both the collection of
anwers from the epxerts with a webform and their analysis.

The answers (for both seed and target questions) are collected with a
Streamlit webapp, that can run locally or from streamlit.io (read below).

Once the seed and target questions have been answered by the expert, you can
run the Python analysis script:

> python elicipy.py

The script will create a new folder with .csv files and a .pptx presentation
of the results of the elicitation.

Authors:
* Mattia de' Michieli Vitturi.
* Andrea Bevilacqua.
* Alessandro Tadini.

Some of the functions are based on the scripts of the Matlab package Anduril
(authors:  Georgios Leontaris and Oswaldo Morales-Napoles).

## Webform

The webform can run locally or at streamlit.io.

Here there are steps to install and run the webapp from streamlit.io.

On Github, you need two repositories, one hosting the webapp streamlit_app.py
(fork this repository) and the other to collect the answers (create a new
repository).

### GITHUB STREAMLIT REPOSITORY

1. Fork this repository.
1. Edit in the input file createWebformDict.py the fields datarepo (with the
   forked repository) and the RepositoryData (with the new github repository
   for the answers).
1. Set in the input file createWebformDict.py the quest_type variable to
   "seed" of "target".
1. Edit the csv file with your questions in the DATA folder of the github
   repository for the webapp (see the questionnaire.csv file for an example of
   the format).
1. Click on the top-right (on your github user icon), and from the menu click
   on "Settings".
1. At the bottom of the left panel, click on "Developer settings".
1. On the left, click on "Personal access tokens".
1. Click on "Generate new token".
1. Give a name and copy your token.
1. Select scope "Repo".

### STREAMLIT

2. login with github account.
2. Open the drop-down menu next to "New app".
2. Select "From existing repo".
2. Select the github repository for the webapp.
2. Click on "Advanced settings".
2. Select Python version 3.7.
2. In the Secrets textbox write.

   github_token = "insert_here_your_token"

2. Click on "Save".
2. Click on "Deploy".

You can share this link for the form:

<https://share.streamlit.io/YOUR_GITHUB_PAGE/createwebform/main>

