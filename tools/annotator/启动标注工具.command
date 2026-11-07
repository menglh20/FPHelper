#!/bin/zsh
set -eu
FP_ANNOTATOR_DIR="${0:A:h}"
FP_ANNOTATOR_PROJECT="${FP_ANNOTATOR_DIR:h:h}"
FP_ANNOTATOR_PYTHON="${FP_ANNOTATOR_PROJECT}/.venv/bin/python"
if [[ ! -x "$FP_ANNOTATOR_PYTHON" ]]; then
  FP_ANNOTATOR_PYTHON="$(command -v python3 || true)"
fi
if [[ -z "$FP_ANNOTATOR_PYTHON" ]]; then
  print "未找到 Python 3，请安装后重试。"
  exit 1
fi
exec "$FP_ANNOTATOR_PYTHON" "$FP_ANNOTATOR_DIR/server.py" --open "$@"
