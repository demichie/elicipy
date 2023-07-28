import os
import sys
import urllib
import shutil
import base64

from github import Github
from github import InputGitTreeElement
from datetime import datetime
from github import Github, GithubException
from github.ContentFile import ContentFile


def github_file_to_bytes(Repository, repo, filename, branch="main"):
    content_encoded = repo.get_contents(
        urllib.parse.quote(filename), ref=branch
    ).content
    content = base64.b64decode(content_encoded)
    dict_file = open(Repository + "/" + filename, "wb")
    dict_file.write(content)
    dict_file.close()
    return


def download_directory(repository, branch, server_path) -> None:
    """
    Download all contents at server_path with commit tag sha in
    the repository.
    """
    if os.path.exists(server_path):
        shutil.rmtree(server_path)

    os.makedirs(server_path)
    contents = repository.get_contents(server_path, branch)

    for content in contents:
        print("Processing %s" % content.path)
        if content.type == "dir":
            os.makedirs(content.path)
            download_directory(repository, branch, content.path)
        else:
            try:
                path = content.path
                if path[-3:] == "csv":
                    content_encoded = repository.get_contents(
                        urllib.parse.quote(path), ref=branch
                    )
                    repo_file = open(path, "w")
                    repo_file.write(content_encoded.decoded_content.decode())
                    repo_file.close()

            except (GithubException, IOError, ValueError) as exc:
                print("Error processing %s: %s", content.path, exc)

    return


def saveDataFromGithub(datarepo, Repository, RepositoryData, user, github_token):

    g = Github(user, github_token)
    repo = g.get_user().get_repo(RepositoryData)

    current_path = os.getcwd()

    filename = "createWebformDict.py"

    if datarepo == "github":

        if os.path.exists("./DATA"):
            print("Deleting old DATA folder")
            shutil.rmtree("./DATA")

        os.makedirs("./DATA")
        os.chdir('./DATA')
        csv_file = 'questionnaire.csv'

        github_file_to_bytes(Repository, repo, csv_file, branch="main")

    elif datarepo == "local_github":
    
        os.chdir('./DATA')
        print('Current path:',os.getcwd())

        if os.path.exists("./seed"):
            print("Deleting old seed folder")
            shutil.rmtree("./seed")
        else:
            print("No seed folder found")    
        if os.path.exists("./target"):
            print("Deleting old seed folder")
            shutil.rmtree("./target")
        else:
            print("No target folder found")    


    print(os.getcwd())

    download_directory(repo, "main", "seed")
    download_directory(repo, "main", "target")
    os.chdir(current_path)

    return
