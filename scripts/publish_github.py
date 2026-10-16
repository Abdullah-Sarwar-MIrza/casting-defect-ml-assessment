"""Publish only this independent project. Uses Git Credential Manager in memory."""
import argparse
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
NAME="casting-defect-ml-assessment"
assert ROOT.name==NAME

def git(*args, capture=False):
    return subprocess.run(["git","-c",f"safe.directory={ROOT.as_posix()}",*args],cwd=ROOT,
                          check=True,text=True,capture_output=capture)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--create",action="store_true"); parser.add_argument("--push",action="store_true"); parser.add_argument("--status",action="store_true"); parser.add_argument("--enable-pages",action="store_true"); args=parser.parse_args()
    credential=subprocess.run(["git","credential","fill"],input="protocol=https\nhost=github.com\n\n",capture_output=True,text=True,check=True,timeout=30)
    values=dict(line.split("=",1) for line in credential.stdout.splitlines() if "=" in line)
    token=values.get("password")
    if not token: raise RuntimeError("No GitHub credential available")
    def api(path,method="GET",payload=None):
        request=urllib.request.Request("https://api.github.com"+path,method=method,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Authorization":"Bearer "+token,"Accept":"application/vnd.github+json","User-Agent":NAME,
                     "X-GitHub-Api-Version":"2022-11-28","Content-Type":"application/json"})
        with urllib.request.urlopen(request,timeout=90) as response:
            data=response.read(); return json.loads(data) if data else {}
    user=api("/user"); owner=user["login"]; repo_path=f"/repos/{owner}/{NAME}"
    if owner.lower()!="abdullah-sarwar-mirza": raise RuntimeError("Unexpected GitHub account")
    if args.create:
        try:
            api(repo_path)
        except urllib.error.HTTPError as exc:
            if exc.code!=404: raise
        else:
            raise RuntimeError("Repository already exists; refusing to mix with existing work")
        if (ROOT/".git").exists(): raise RuntimeError("New project already has Git history")
        repo=api("/user/repos","POST",dict(name=NAME,description="CPU-trained casting defect classification: baseline, transfer learning, evaluation and live ONNX inference",private=False,auto_init=False))
        git("init","-b","main")
        git("config","user.name",user.get("name") or owner)
        git("config","user.email",f'{user["id"]}+{owner}@users.noreply.github.com')
        git("remote","add","origin",repo["clone_url"])
        git("add",".")
        git("diff","--cached","--check")
        tracked=git("ls-files",capture=True).stdout.splitlines()
        assert not any(name.startswith((".venv/","cache/",".publishing/","data/")) or name==".env" for name in tracked)
        git("commit","-m","Build independent casting ML assessment with trained models and public inference")
        git("push","-u","origin","main")
        pages=api(repo_path+"/pages","POST",dict(source=dict(branch="main",path="/docs")))
        api(repo_path,"PATCH",dict(homepage=pages["html_url"]))
        print(json.dumps(dict(repository=repo["html_url"],deployment=pages["html_url"]),indent=2))
    if args.push:
        assert (ROOT/".git").is_dir()
        remote=git("remote","get-url","origin",capture=True).stdout.strip()
        assert remote.lower()==f"https://github.com/{owner}/{NAME}.git".lower()
        git("add","."); git("diff","--cached","--check")
        changes=git("diff","--cached","--name-only",capture=True).stdout.strip()
        if changes: git("commit","-m","Record verified live deployment and container inference evidence")
        git("push","origin","main")
    if args.enable_pages:
        try:
            pages=api(repo_path+"/pages","POST",dict(source=dict(branch="main",path="/docs")))
        except urllib.error.HTTPError as exc:
            if exc.code!=409: raise
            pages=api(repo_path+"/pages")
        api(repo_path,"PATCH",dict(homepage=pages["html_url"]))
        print(json.dumps(dict(deployment=pages["html_url"]),indent=2))
    if args.status:
        for suffix in ["/pages","/pages/builds/latest","/actions/runs?per_page=3"]:
            value=api(repo_path+suffix)
            if "workflow_runs" in value:
                print(json.dumps([dict(name=r["name"],status=r["status"],conclusion=r["conclusion"],url=r["html_url"]) for r in value["workflow_runs"]],indent=2))
            else:
                print(json.dumps({k:value.get(k) for k in ["status","html_url","error","updated_at"]},indent=2))

if __name__=="__main__": main()
