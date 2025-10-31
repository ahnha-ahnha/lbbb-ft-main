# src/run_main.py
import argparse, os, pandas as pd
from ft_utils import seed_everything
from ft_model import run_main as _run_main

ROIS = [
False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,
False,False,False,True,True,True,False,False,True,True,True,False,True,True,False,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,True,
True,True,True,True,True,True,True,False,False
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="./hsw_2")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--heads", type=int, default=8)
    args = p.parse_args()

    seed_everything(21)
    df_raw = pd.read_csv(args.data_path)
    if df_raw.columns[0].lower() in {"", "index", "unnamed: 0"}:
        df_raw = df_raw.iloc[:, 1:]

    out = _run_main(df_raw, ROIS, args.out_dir, args.epochs, args.lr, args.dim, args.depth, args.heads)
    print("Saved:", out["csv_path"])

if __name__ == "__main__":
    main()
