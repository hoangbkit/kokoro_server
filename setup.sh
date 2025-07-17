#!/bin/bash

# Activate python venv
source venv/bin/activate

# Install dependency
pip install -r requirements.txt

# Locate all required paths
KOKORO_CONFIG=$(python -c "import kokoro_onnx, os; print(os.path.join(os.path.dirname(kokoro_onnx.__file__), 'config.json'))")
LANGUAGE_TAGS_BASE=$(python -c "import language_tags.data, os; print(os.path.join(os.path.dirname(language_tags.data.__file__), 'json'))")
INDEX_JSON="$LANGUAGE_TAGS_BASE/index.json"
REGISTRY_JSON="$LANGUAGE_TAGS_BASE/registry.json"
ESPEAKNG_DATA_DIR=$(python -c "import espeakng_loader, os; print(os.path.join(os.path.dirname(espeakng_loader.__file__), 'espeak-ng-data'))")
ESPEAKNG_LIB=$(python -c "import espeakng_loader, os; print(os.path.join(os.path.dirname(espeakng_loader.__file__), 'libespeak-ng.dylib'))")

# Check everything exists
[[ ! -f "$KOKORO_CONFIG" ]] && echo "Missing config.json" && exit 1
[[ ! -f "$INDEX_JSON" ]] && echo "Missing index.json" && exit 1
[[ ! -f "$REGISTRY_JSON" ]] && echo "Missing registry.json" && exit 1
[[ ! -d "$ESPEAKNG_DATA_DIR" ]] && echo "Missing espeak-ng-data dir" && exit 1
[[ ! -f "$ESPEAKNG_LIB" ]] && echo "Missing libespeak-ng.dylib" && exit 1

# Build frozen binary
pyinstaller --onedir \
  --add-data "$KOKORO_CONFIG:kokoro_onnx" \
  --add-data "$INDEX_JSON:language_tags/data/json" \
  --add-data "$REGISTRY_JSON:language_tags/data/json" \
  --add-data "$ESPEAKNG_DATA_DIR:espeakng_loader/espeak-ng-data" \
  --add-binary "$ESPEAKNG_LIB:espeakng_loader" \
  --noconfirm \
  kokoro_server.py

# Exit python venv
deactivate