"""Builds lambda.zip (Python 3.12, x86_64 Lambda). Works on Windows, macOS and Linux.

Run from the rag-platform folder:  python build.py
Needs internet: pip packages plus a one-time model download from Hugging Face.
"""
import os
import shutil
import stat
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build")
ZIP_PATH = os.path.join(ROOT, "lambda.zip")


def run(cmd):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)


def folder_mb(path):
    total = 0
    for base, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(base, f)) for f in files)
    return round(total / 1024 / 1024, 1)


def _force_remove(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


if os.path.exists(BUILD):
    try:
        shutil.rmtree(BUILD, onerror=_force_remove)
    except OSError as err:
        raise SystemExit(f"Could not delete the old build folder ({err}). Delete the 'build' folder by hand in File Explorer, then run again.")
if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)
os.makedirs(os.path.join(BUILD, "models"), exist_ok=True)

# Lambda-compatible wheels, even when building on Windows or macOS
run([
    sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-t", BUILD,
    "--platform", "manylinux_2_28_x86_64", "--platform", "manylinux_2_17_x86_64", "--platform", "manylinux2014_x86_64",
    "--python-version", "3.12", "--implementation", "cp", "--only-binary=:all:", "--upgrade",
])

for name in os.listdir(os.path.join(ROOT, "src")):
    if name.endswith(".py"):
        shutil.copy(os.path.join(ROOT, "src", name), BUILD)

# Bundle all-MiniLM-L6-v2 (quantized ONNX, about 23 MB) so Lambda never downloads anything
run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402

repo = "sentence-transformers/all-MiniLM-L6-v2"


def pick_onnx(files):
    """Prefer a small quantized build; fall back to the full model if none exists."""
    for needle in ("quint8_avx2", "qint8_avx2", "qint8_avx512", "model_O3", "model.onnx"):
        for f in sorted(files):
            if needle in f:
                return f
    raise SystemExit("No .onnx file found in the repo. Files seen: " + str(files))


onnx_files = [f for f in list_repo_files(repo) if f.startswith("onnx/") and f.endswith(".onnx")]
print("ONNX files in the repo:", onnx_files)
chosen = pick_onnx(onnx_files)
print("using:", chosen)
shutil.copy(hf_hub_download(repo, chosen), os.path.join(BUILD, "models", "model.onnx"))
shutil.copy(hf_hub_download(repo, "tokenizer.json"), os.path.join(BUILD, "models", "tokenizer.json"))

for base, dirs, _ in os.walk(BUILD):
    for d in list(dirs):
        if d in ("__pycache__", "tests"):
            shutil.rmtree(os.path.join(base, d), ignore_errors=True)
            dirs.remove(d)

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for base, _, files in os.walk(BUILD):
        for f in files:
            full = os.path.join(base, f)
            z.write(full, os.path.relpath(full, BUILD))

print(f"\nunzipped size: {folder_mb(BUILD)} MB (Lambda limit is 250)")
print(f"zip size: {round(os.path.getsize(ZIP_PATH) / 1024 / 1024, 1)} MB")
print(f"created: {ZIP_PATH}")
