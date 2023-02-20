# Elicipy

Elicipy is an Expert elicitation Python code.

The input files are created with a Streamlit webapp, that can run locally or from streamlit.io (read below)

Once the seed and target questions have been answered by the expert, you can copy the full folder here.

Then, you can run the Python script:

> python elicipy.py

The script will create a new folder with .csv files and a .pptx presentation of the results of the elicitation.

Authors:
- Mattia de' Michieli Vitturi
- Andrea Bevilacqua
- Alessandro Tadini

Some of the functions are based on the scripts of the Matlab package Anduril (authors:  Georgios Leontaris and Oswaldo Morales-Napoles).


# createWebform

GITHUB

1) Fork the repository
2) Edit the csv file with your questions in the DATA folder (see the questionnaire.csv file for an example of the format)
3) Set the quest_type variable to "seed" of "target"
4) Click on the top-right (on your github user icon), and from the menu click on "Settings"
5) At the bottom of the left panel, click on "Developer settings"
6) On the left, click on "Personal access tokens"
7) Click on "Generate new token"
8) Give a name and copy your token
9) Select scope "Repo"


STREAMLIT

1) login with github account
2) Open the drop-down menu next to "New app"
3) Select "From existing repo"
4) Select the github repository for the webform
5) Click on "Advanced settings"
6) Select Python version 3.7
7) In the Secrets textbox write
   
   github_token = "insert_here_your_token"

8) Click on "Save"
9) Click on "Deploy"


You can share this link for the form:

https://share.streamlit.io/YOUR_GITHUB_PAGE/createwebform/main

