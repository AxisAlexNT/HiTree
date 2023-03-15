Set-Variable "HICT_DIR" "${PSScriptRoot}/../HiCT_Library/"
Set-Variable "TESTS_DIR" "${HICT_DIR}/tests"
$env:PYTHONPATH += ";${HICT_DIR}"
echo "Setting HICT_DIR = ${HICT_DIR} and PYTHONPATH = ${env:PYTHONPATH}"
pytest -vvv -x -n16 ${TESTS_DIR}
