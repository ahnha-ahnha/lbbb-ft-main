# src/ft_model.py
import os, time
from typing import List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import f

# --- 유연한 import: 수정본 Encoder2 / 공식 FTTransformer 둘 다 커버 ---
_TT_CLASS = None
try:
    from tab_transformer_pytorch.ft_transformer import FTTransformerEncoder2 as _TT
    _TT_CLASS = "encoder2"
except Exception:
    try:
        from tab_transformer_pytorch import FTTransformer as _TT
        _TT_CLASS = "ft"
    except Exception as e:
        raise ImportError(
            "tab_transformer_pytorch import 실패. vendor/tab_transformer_pytorch를 editable로 설치했는지 확인하세요."
        ) from e

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_ft_transformer(categories, num_continuous, dim, depth, heads):
    if _TT_CLASS == "encoder2":
        return _TT(
            categories=categories,
            num_continuous=num_continuous,
            dim=dim, depth=depth, heads=heads,
            dim_head=16, dim_out=1, num_special_tokens=2,
            attn_dropout=0., ff_dropout=0.
        )
    else:
        return _TT(
            categories=categories,
            num_continuous=num_continuous,
            dim=dim, depth=depth, heads=heads,
            dim_head=16, dim_out=1,
            attn_dropout=0., ff_dropout=0.
        )

def hostelling_T_sq(data1: torch.Tensor, data2: torch.Tensor):
    n1, p1 = data1.shape
    n2, p2 = data2.shape
    mean1 = data1.mean(dim=0); mean2 = data2.mean(dim=0)
    cov1 = torch.cov(data1.T); cov2 = torch.cov(data2.T)
    pooled = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)
    diff = mean1 - mean2
    T2 = (n1 * n2) / (n1 + n2) * (diff @ torch.inverse(pooled) @ diff)
    p = p1
    F_stat = (n1 + n2 - p - 1) / (p * (n1 + n2 - 2)) * T2
    p_value = 1 - f.cdf(float(F_stat.detach().cpu().item()), p, n1 + n2 - p - 1)
    return T2, float(p_value)

class TFT(nn.Module):
    def __init__(self, ft):
        super().__init__()
        self.ft = ft
        self.to_logits = nn.Sequential(
            nn.LayerNorm(ft.dim),
            nn.ReLU(),
            nn.Linear(ft.dim, 1),
        )

    def forward(self, x_categ, x_numer, label, return_attn=False):
        if _TT_CLASS == "encoder2":
            if return_attn:
                embeddings, attn, cls_embedding = self.ft(x_categ, x_numer, return_attn=True)
            else:
                embeddings, cls_embedding = self.ft(x_categ, x_numer, return_attn)
        else:
            # FTTransformer (0.4.x)는 forward(x_categ, x_numer) → (embeddings, cls)
            out = self.ft(x_categ, x_numer, return_attn=return_attn) if hasattr(self.ft, 'forward') else self.ft(x_categ, x_numer)
            if isinstance(out, tuple) and len(out) == 3:   # (emb, attn, cls)
                embeddings, _, cls_embedding = out
            elif isinstance(out, tuple) and len(out) == 2: # (emb, cls)
                embeddings, cls_embedding = out
            else:
                raise RuntimeError("FTTransformer forward 반환 형태를 확인하세요.")
        logits = self.to_logits(cls_embedding)

        # (B,F,D) -> (F,B,D)
        embeddings = torch.permute(embeddings, (1,0,2))
        g0 = (label == 0).flatten(); g1 = (label == 1).flatten()

        p_values, t_values = [], []
        for emb in embeddings:
            emb0, emb1 = emb[g0], emb[g1]
            t, p_val = hostelling_T_sq(emb0, emb1)
            t_values.append(float(t.detach().cpu().item()) if torch.is_tensor(t) else float(t))
            p_values.append(float(p_val))
        t_values = torch.tensor(t_values, device=logits.device)[None,:]
        return {'t_value': t_values, 'p_value': p_values, 'logits': logits}

def run_main(df_raw: pd.DataFrame, rois: List[bool], out_dir: str, epochs: int, lr: float, dim: int, depth: int, heads: int):
    from ft_utils import convert_column_types, preprocess, COLUMN_TYPES
    os.makedirs(out_dir, exist_ok=True)
    df_raw = convert_column_types(df_raw, COLUMN_TYPES)
    X_df, y_np, _, col_names = preprocess(df_raw)

    # tensors
    cat_cols = X_df.select_dtypes(['int64']).columns
    num_cols = X_df.select_dtypes(['float32']).columns
    X_num = torch.tensor(X_df[num_cols].values, dtype=torch.float32, device=DEVICE)
    X_cat = torch.tensor(X_df[cat_cols].values, dtype=torch.long, device=DEVICE)
    y_t   = torch.tensor(y_np, dtype=torch.float32, device=DEVICE)

    # model
    ft = build_ft_transformer(
        categories=[X_df[c].nunique() for c in cat_cols],
        num_continuous=len(num_cols),
        dim=dim, depth=depth, heads=heads
    ).to(DEVICE)
    model = TFT(ft).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_loss = float('inf'); best_p_values = []
    for ep in range(epochs):
        model.train(); optimizer.zero_grad()
        out = model(X_cat, X_num, y_t, return_attn=False)
        t_value, p_value, logits = out['t_value'], out['p_value'], out['logits']

        l2_loss = sum((p**2).sum() for p in model.parameters()) / sum(p.numel() for p in model.parameters())
        t_loss = 1.0 / (torch.log(t_value.sum() + 1) + 1e-8)
        ce_loss = criterion(logits, y_t)
        loss = t_loss + 1e-5 * l2_loss + 1e-5 * ce_loss
        loss.backward(); optimizer.step()

        cur = float(loss.item())
        if cur < best_loss:
            best_loss = cur; best_p_values = p_value
            torch.save(model.state_dict(), os.path.join(out_dir, f'FTT_best_main.pth'))

    ts = time.strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_dir, f'pvalues_main_dim{dim}_depth{depth}_heads{heads}_{ts}.csv')
    pd.DataFrame({
        'columns': col_names,
        'p-values': best_p_values,
        'rois': (rois[:len(col_names)] if rois is not None else [False]*len(col_names))
    }).to_csv(csv_path, index=False)

    return {"csv_path": csv_path, "best_loss": best_loss, "epochs": epochs}
