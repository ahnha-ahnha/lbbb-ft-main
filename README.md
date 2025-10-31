# FT-Transformer Feature Selection

Main-only, reproducible pipeline to compute feature-wise p-values from FT-Transformer embeddings via Hotelling's T^2.

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./vendor/tab_transformer_pytorch

python -m src.run_main \
  --data_path "your_dataset.csv" \
  --out_dir "./your_output_dir" --epochs 5000 --lr 1e-4 --dim 32 --depth 1 --heads 8
