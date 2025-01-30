import os
import sys
import urllib
import shutil
import base64

from github import Github, GithubException


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


def saveDataFromGithub(Repository, user, github_token):

    g = Github(user, github_token)
    repo = g.get_user().get_repo(Repository)

    filename = "createWebformDict.py"

    if os.path.exists(Repository):
        shutil.rmtree(Repository)

    os.makedirs(Repository)

    github_file_to_bytes(Repository, repo, filename, branch="main")

    current_path = os.getcwd()

    os.chdir(Repository)

    print(os.getcwd())

    sys.path.insert(0, os.getcwd())
    from createWebformDict import input_dir

    download_directory(repo, "main", input_dir)
    os.chdir(current_path)

    return
