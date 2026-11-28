"""Sync data/ with a private Hugging Face dataset repo, so the Mac and Colab share files.

    python -m src.hub push   # upload data/ (creates the private repo on first use)
    python -m src.hub pull   # download it into data/
"""

import sys

from huggingface_hub import HfApi, snapshot_download

from src.data import DATA

REPO_NAME = "trigger-or-defection-data"


def repo_id(api: HfApi | None = None) -> str:
    api = api or HfApi()
    return f"{api.whoami()['name']}/{REPO_NAME}"


def push(paths: list[str] | None = None, message: str = "Update data") -> str:
    """Upload all of data/, or only the given paths (relative to data/)."""
    api = HfApi()
    rid = repo_id(api)
    api.create_repo(rid, repo_type="dataset", private=True, exist_ok=True)
    if paths is None:
        api.upload_folder(folder_path=DATA, repo_id=rid, repo_type="dataset",
                          ignore_patterns=[".gitkeep"], commit_message=message)
    else:
        for p in paths:
            api.upload_file(path_or_fileobj=DATA / p, path_in_repo=p, repo_id=rid,
                            repo_type="dataset", commit_message=message)
    return rid


def pull() -> str:
    rid = repo_id()
    snapshot_download(rid, repo_type="dataset", local_dir=DATA)
    return rid


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "push":
        print("pushed to", push(message="Push data/ from local"))
    elif cmd == "pull":
        print("pulled from", pull())
    else:
        sys.exit("usage: python -m src.hub [push|pull]")
