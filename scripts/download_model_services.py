"""Download the two pinned GGUFs; requires the optional huggingface_hub package."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

MANIFEST = Path(__file__).with_name('model_services_manifest.json')


def download_models(model_dir, download=None):
    if download is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError('Install huggingface_hub with your virtual environment Python first.') from exc
        download = hf_hub_download
    model_dir = Path(model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for model in json.loads(MANIFEST.read_text(encoding='utf-8')):
        filename = model['filename']
        if Path(filename).name != filename or not filename.endswith('.gguf'):
            raise ValueError('Invalid model manifest filename.')
        print(f"Downloading {model['repo']}: {filename}", flush=True)
        paths.append(download(repo_id=model['repo'], revision=model['revision'],
                              filename=filename, local_dir=str(model_dir)))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    args = parser.parse_args()
    for path in download_models(args.model_dir):
        print(f'Ready: {path}')


if __name__ == '__main__':
    main()
