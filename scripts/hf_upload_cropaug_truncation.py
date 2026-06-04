"""dope_cropaug_ft_s2 (truncation best) 를 HF Hub 에 업로드.

토큰은 환경변수 HF_TOKEN 으로만 전달 (스크립트에 박지 않음):
    HF_TOKEN=hf_xxx python scripts/hf_upload_cropaug_truncation.py
"""
import os
from huggingface_hub import HfApi

REPO_ID = "CanelE452/pallet-dope-cropaug-truncation"
SRC = os.path.join(os.path.dirname(__file__), "..", "weights", "dope_cropaug_ft_s2")

token = os.environ["HF_TOKEN"]
api = HfApi(token=token)

api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)

uploads = [
    ("final_net_epoch_0180.pth", "final_net_epoch_0180.pth"),
    ("header.txt", "header.txt"),
    ("README.md", "README.md"),
    ("eval_results/eval_summary.json", "eval_summary.json"),
]

for local_rel, repo_rel in uploads:
    local = os.path.join(SRC, local_rel)
    print(f"uploading {repo_rel} ...", flush=True)
    api.upload_file(
        path_or_fileobj=local,
        path_in_repo=repo_rel,
        repo_id=REPO_ID,
        repo_type="model",
    )

print("done:", f"https://huggingface.co/{REPO_ID}")
print("files:", api.list_repo_files(repo_id=REPO_ID))
