set -x 

python scripts/data/preprocess_codecontests.py
python scripts/data/preprocess_lcb.py
python scripts/data/preprocess_mbppplus.py
python scripts/data/preprocess_rlvr.py
python scripts/data/preprocess_taco.py
python scripts/data/preprocess_deepscaler.py

pip install lighteval
python scripts/data/preprocess_aime24.py
python scripts/data/preprocess_aime25.py
python scripts/data/preprocess_gpqa.py