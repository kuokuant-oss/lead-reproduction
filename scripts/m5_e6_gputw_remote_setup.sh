#!/usr/bin/env bash
# 在 GPUtw instance 上建立固定環境並解壓 bundle。
#
# 不修改系統全域 Python:一律建立獨立 venv。不使用 latest、不允許自動升級,
# 版本全部釘死在 environment_contract.json 記錄的值,那份值來自正式 E6
# gpu-host 的已驗證環境,而不是從執行中的 gpu-host 讀來的。
set -eu
STAGE="${1:?用法: m5_e6_gputw_remote_setup.sh <stage-dir> <bundle-file>}"
BUNDLE="${2:?缺少 bundle 檔名}"

cd "$STAGE"
if command -v unzstd >/dev/null 2>&1 && [ "${BUNDLE##*.}" = "zst" ]; then
  tar --use-compress-program=unzstd -xf "$BUNDLE"
else
  tar -xzf "$BUNDLE" 2>/dev/null || tar -xf "$BUNDLE"
fi
printf 'extracted %s files\n' "$(ls -1 | wc -l)"

VENV="$STAGE/.venv"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
. "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip

# 版本全部釘死。不使用 latest,不允許 pip 自由解析。
python -m pip install --quiet --no-input \
  "numpy==2.4.6" "pandas==3.0.3" "scikit-learn==1.8.0" "joblib==1.5.3" \
  "scipy>=1.11.1" "psutil>=5.9" "tabpfn==8.0.8"

# torch 依契約釘版;若該 build 不存在就直接失敗,不退而求其次裝別的版本。
python -m pip install --quiet --no-input \
  "torch==2.12.1" --index-url https://download.pytorch.org/whl/cu130 || {
    printf 'torch 2.12.1+cu130 安裝失敗;不接受替代版本,停止\n' >&2
    exit 5
  }

python - <<'PY'
import platform, torch, tabpfn, numpy, pandas, sklearn, joblib
print("python", platform.python_version())
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("tabpfn", tabpfn.__version__)
print("numpy", numpy.__version__, "pandas", pandas.__version__)
print("sklearn", sklearn.__version__, "joblib", joblib.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
printf 'remote setup 完成\n'
