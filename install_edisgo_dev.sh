#!/bin/bash

# Script to install the venv in the ./venv dir, install pre-commit hooks.

set -euo pipefail
#set -x

workflow_dir="$(dirname "$(realpath -s "$0")")"

cd "$workflow_dir"

if [ -d "venv" ]
then
  rm -rf "venv"
else
  echo No dir venv.
fi

virtualenv "venv" --python python3.8

source "$workflow_dir/venv/bin/activate"
python --version
echo "Upgrade pip."
python -m pip -q install --upgrade pip

for repo in "[dev]"
do
  echo "Install $workflow_dir$repo."
  python -m pip install -e "$workflow_dir$repo"
done

echo "Save pip freeze to freeze.txt."
python --version > "$workflow_dir/venv/freeze.txt"
python -m pip -q freeze >> "$workflow_dir/venv/freeze.txt"


pre-commit install
