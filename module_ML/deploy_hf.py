"""Upload model fine-tuned dan app Space ke Hugging Face Hub."""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def ensure_repo(api: HfApi, repo_id: str, repo_type: str) -> None:
    if repo_type == "space":
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, space_sdk="gradio")
        return
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)


def upload_model(api: HfApi, model_dir: Path, model_repo: str) -> None:
    if not model_dir.exists():
        raise FileNotFoundError(f"Folder model tidak ditemukan: {model_dir}")

    ensure_repo(api, model_repo, repo_type="model")
    api.upload_folder(
        repo_id=model_repo,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message="Upload fine-tuned IndoBERT model",
    )


def upload_space(api: HfApi, space_dir: Path, space_repo: str) -> None:
    if not space_dir.exists():
        raise FileNotFoundError(f"Folder Space tidak ditemukan: {space_dir}")

    ensure_repo(api, space_repo, repo_type="space")
    api.upload_folder(
        repo_id=space_repo,
        repo_type="space",
        folder_path=str(space_dir),
        commit_message="Update Space app",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-repo", type=str, required=True, help="Repo model HF, contoh: username/indobert-tokopedia-sentiment")
    parser.add_argument("--model-dir", type=str, default="module_ML/models/transformer/final_model")
    parser.add_argument("--space-repo", type=str, default=None, help="Repo Space HF, contoh: username/tokopedia-sentiment-space")
    parser.add_argument("--space-dir", type=str, default="module_ML/hf_space")
    args = parser.parse_args()

    api = HfApi()
    model_dir = Path(args.model_dir)
    upload_model(api, model_dir, args.model_repo)
    print(f"Model berhasil di-upload ke: https://huggingface.co/{args.model_repo}")

    if args.space_repo:
        upload_space(api, Path(args.space_dir), args.space_repo)
        print(f"Space berhasil di-update: https://huggingface.co/spaces/{args.space_repo}")
        print("Ingat set environment variable MODEL_REPO pada Space ke repo model yang baru.")


if __name__ == "__main__":
    main()
