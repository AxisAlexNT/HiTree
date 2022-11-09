#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
HICT_DIR="$SCRIPT_DIR/../HiCT_Library/"
export PYTHONPATH="$PYTHONPATH:$HICT_DIR"
mypy -p hict
rm -rf "$HICT_DIR/build" "$HICT_DIR/dist" && python3 "$HICT_DIR/setup.py" bdist_wheel && pip3 install --no-deps --force-reinstall "${HICT_DIR}/dist"/*.whl
