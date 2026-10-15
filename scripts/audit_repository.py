"""Read-only inventory of tracked binary/privacy artifacts. Never unpickle data.

Run from the repository root: python scripts/audit_repository.py
Ignoring a file does not untrack it or erase it from Git history.
"""
import json
from pathlib import Path
import subprocess


def audit(root):
    root = Path(root)
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode('utf-8').split('\0')
    findings = []
    for name in filter(None, names):
        path = Path(name)
        category = None
        if path.suffix.lower() == '.pkl':
            category = 'pickle_or_biometric_review'
        elif path.suffix.lower() in {'.m4a', '.wav', '.mp3', '.flac'}:
            category = 'recording_review'
        elif path.suffix.lower() in {'.docx', '.pdf'}:
            category = 'binary_document_retained'
        elif name.startswith('ai/wakeword/openWakeWord/'):
            category = 'vendored_dependency_review'
        if category:
            findings.append({'path': name, 'category': category})
    return findings


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    print(json.dumps({'read_only': True, 'findings': audit(root)}, indent=2))
