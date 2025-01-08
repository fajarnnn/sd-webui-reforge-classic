@echo off

:: set PYTHON=
:: set GIT=
:: set VENV_DIR=
:: set UV=1
set COMMANDLINE_ARGS= --no-download-sd-model

:: --xformers
:: --pin-shared-memory --cuda-malloc --cuda-stream
:: --skip-python-version-check --skip-torch-cuda-test --skip-version-check --skip-prepare-environment --skip-install

call webui.bat
