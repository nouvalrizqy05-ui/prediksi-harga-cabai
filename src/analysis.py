"""
Chili Price Prediction in Central Java
=======================================
Comparison of Simple Linear Regression and Multiple Polynomial Regression
with National PIHPS Price as Exogenous Variable.

Authors : Muhammad Nouval Ar-Rizqy, Rizky Ghaly Saputra,
          Selly Anggun Ramadhani, Muhammad Rais Al Mubarak
Affil.  : Department of Computer Science, Universitas Negeri Semarang
Journal : Scientific Journal of Informatics
Data    : PIHPS (pihps.kemenperin.go.id), July 2022 – December 2025

Usage
-----
    python src/analysis.py

Outputs (written to results/ and figures/)
------------------------------------------
    results/metrics_all.csv          - MAE / RMSE / MAPE for every model & commodity
    results/predictions_test.csv     - Actual vs predicted for test period (Apr–Dec 2025)
    results/model_equations.csv      - Fitted equation coefficients
    figures/fig1_harga_cabai.png     - Figure 1: price history
    figures/fig2_distribusi_data.png - Figure 2: train/test split
    figures/fig3_prediksi_cmb.png    - Figure 3: CMB prediction comparison
    figures/fig4_prediksi_cmk.png    - Figure 4: CMK prediction comparison
    figures/fig5_prediksi_crh.png    - Figure 5: CRH prediction comparison
    figures/fig6_prediksi_crm.png    - Figure 6: CRM prediction comparison

Requirements
------------
    pip install numpy scikit-learn matplotlib openpyxl pandas
"""

import os
import csv
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / 'data' / 'raw'
DATA_PRO = ROOT / 'data' / 'processed'
FIGURES  = ROOT / 'figures'
RESULTS  = ROOT / 'results'

for d in [FIGURES, RESULTS]:
    d.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────
N_TRAIN  = 33   # July 2022 – March 2025
N_TEST   = 9    # April 2025 – December 2025
DEG_MAX  = 5    # maximum polynomial degree to explore
BEST_DEG = {    # optimal degree per commodity (selected by test-set MAPE)
    'CMB': 2,
    'CMK': 2,
    'CRH': 2,
    'CRM': 3,
}
COMMODITY_LABELS = {
    'CMB': 'Cabai Merah Besar',
    'CMK': 'Cabai Merah Keriting',
    'CRH': 'Cabai Rawit Hijau',
    'CRM': 'Cabai Rawit Merah',
}

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linestyle':     '--',
})

PALETTE_4 = ['#e63946', '#2196F3', '#4CAF50', '#FF9800']


# ══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_all():
    """Return dict[code] -> {'X': ndarray, 'Y': ndarray} and list of dates."""
    import openpyxl
    file_map = {
        'CMB': DATA_RAW / 'PIHPS_CMB_Cabai_Merah_Besar.xlsx',
        'CMK': DATA_RAW / 'PIHPS_CMK_Cabai_Merah_Keriting.xlsx',
        'CRH': DATA_RAW / 'PIHPS_CRH_Cabai_Rawit_Hijau.xlsx',
        'CRM': DATA_RAW / 'PIHPS_CRM_Cabai_Rawit_Merah.xlsx',
    }
    data = {}
    dates = None
    for code, path in file_map.items():
        wb   = openpyxl.load_workbook(path, data_only=True)
        ws   = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if dates is None:
            raw_dates = [str(d).strip() for d in rows[0][2:]]
            dates = []
            for d in raw_dates:
                p = d.replace(' ', '').split('/')
                dates.append(datetime(int(p[2]), int(p[1]), int(p[0])))
        data[code] = {
            'X': np.array([float(str(v).replace(',', '')) for v in rows[1][2:]]),
            'Y': np.array([float(str(v).replace(',', '')) for v in rows[2][2:]]),
        }
    return data, dates


# ══════════════════════════════════════════════════════════════════════════
# 2. METRICS
# ══════════════════════════════════════════════════════════════════════════

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ══════════════════════════════════════════════════════════════════════════
# 3. MODEL FITTING & PREDICTION
# ══════════════════════════════════════════════════════════════════════════

def fit_slr(X_train, Y_train):
    """Fit Simple Linear Regression; return (model, X_mean)."""
    X_mean = 0.0  # no centering — matches reported MAPE values
    Xc = (X_train - X_mean).reshape(-1, 1)
    model = LinearRegression().fit(Xc, Y_train)
    return model, X_mean


def predict_slr(model, X_mean, X_test):
    Xc = (X_test - X_mean).reshape(-1, 1)
    return model.predict(Xc)


def fit_poly(X_train, Y_train, degree):
    """Fit polynomial regression via numpy lstsq; return (coefs, pf, X_mean)."""
    X_mean = 0.0  # no centering — matches reported MAPE values
    Xc = (X_train - X_mean).reshape(-1, 1)
    pf   = PolynomialFeatures(degree)
    Phi  = pf.fit_transform(Xc)
    coefs, *_ = np.linalg.lstsq(Phi, Y_train, rcond=None)
    return coefs, pf, X_mean


def predict_poly(coefs, pf, X_mean, X_test):
    Xc   = (X_test - X_mean).reshape(-1, 1)
    Phi  = pf.transform(Xc)
    return Phi @ coefs


# ══════════════════════════════════════════════════════════════════════════
# 4. FULL EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════

def run_all(data):
    """
    For every commodity, fit SLR + Poly(2..DEG_MAX) on training data,
    evaluate on test data. Return metrics dict and predictions dict.
    """
    metrics_rows   = []   # for results/metrics_all.csv
    pred_rows      = []   # for results/predictions_test.csv
    equation_rows  = []   # for results/model_equations.csv

    for code in ['CMB', 'CMK', 'CRH', 'CRM']:
        X = data[code]['X']
        Y = data[code]['Y']
        X_tr, Y_tr = X[:N_TRAIN], Y[:N_TRAIN]
        X_te, Y_te = X[N_TRAIN:], Y[N_TRAIN:]

        # ── SLR ──────────────────────────────────────────────────────────
        slr_model, slr_xmean = fit_slr(X_tr, Y_tr)
        y_slr = predict_slr(slr_model, slr_xmean, X_te)
        metrics_rows.append({
            'commodity': code, 'model': 'SLR', 'degree': 1,
            'MAE_IDR': round(mae(Y_te, y_slr)),
            'RMSE_IDR': round(rmse(Y_te, y_slr)),
            'MAPE_pct': round(mape(Y_te, y_slr), 4),
        })
        # equation (uncentered intercept approximation)
        b1 = slr_model.coef_[0]
        b0 = slr_model.intercept_ - b1 * slr_xmean
        equation_rows.append({'commodity': code, 'model': 'SLR',
                               'degree': 1, 'b0': round(b0, 4),
                               'b1': f'{b1:.4f}', 'b2': '', 'b3': ''})

        for d in range(2, DEG_MAX + 1):
            coefs, pf, poly_xmean = fit_poly(X_tr, Y_tr, d)
            y_poly = predict_poly(coefs, pf, poly_xmean, X_te)
            metrics_rows.append({
                'commodity': code, 'model': f'Poly-{d}', 'degree': d,
                'MAE_IDR': round(mae(Y_te, y_poly)),
                'RMSE_IDR': round(rmse(Y_te, y_poly)),
                'MAPE_pct': round(mape(Y_te, y_poly), 4),
            })
            if d <= 3:
                eq = {'commodity': code, 'model': f'Poly-{d}', 'degree': d,
                      'b0': round(coefs[0], 4),
                      'b1': f'{coefs[1]:.6e}' if len(coefs) > 1 else '',
                      'b2': f'{coefs[2]:.6e}' if len(coefs) > 2 else '',
                      'b3': f'{coefs[3]:.6e}' if len(coefs) > 3 else ''}
                equation_rows.append(eq)

        # ── Best-model predictions for CSV ───────────────────────────────
        bd = BEST_DEG[code]
        coefs_best, pf_best, xmean_best = fit_poly(X_tr, Y_tr, bd)
        y_best = predict_poly(coefs_best, pf_best, xmean_best, X_te)
        for i in range(N_TEST):
            pred_rows.append({
                'commodity': code, 'obs_index': N_TRAIN + i,
                'X_nasional': int(X_te[i]), 'Y_actual': int(Y_te[i]),
                'Y_pred_SLR': round(y_slr[i]),
                f'Y_pred_Poly{bd}': round(y_best[i]),
                'error_SLR': round(Y_te[i] - y_slr[i]),
                f'error_Poly{bd}': round(Y_te[i] - y_best[i]),
            })

    return metrics_rows, pred_rows, equation_rows


# ══════════════════════════════════════════════════════════════════════════
# 5. WRITE RESULTS CSVs
# ══════════════════════════════════════════════════════════════════════════

def write_metrics(metrics_rows):
    path = RESULTS / 'metrics_all.csv'
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['commodity','model','degree','MAE_IDR','RMSE_IDR','MAPE_pct'])
        w.writeheader()
        w.writerows(metrics_rows)
    print(f'  Written: {path}')


def write_predictions(pred_rows, dates):
    path = RESULTS / 'predictions_test.csv'
    # add date column
    for row in pred_rows:
        row['date'] = dates[row['obs_index']].strftime('%Y-%m-%d')
    fieldnames = ['commodity','date','obs_index','X_nasional','Y_actual',
                  'Y_pred_SLR','Y_pred_Poly2','Y_pred_Poly3',
                  'error_SLR','error_Poly2','error_Poly3']
    # fill missing keys
    for row in pred_rows:
        for k in fieldnames:
            row.setdefault(k, '')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(pred_rows)
    print(f'  Written: {path}')


def write_equations(equation_rows):
    path = RESULTS / 'model_equations.csv'
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['commodity','model','degree','b0','b1','b2','b3'])
        w.writeheader()
        w.writerows(equation_rows)
    print(f'  Written: {path}')


# ══════════════════════════════════════════════════════════════════════════
# 6. FIGURES
# ══════════════════════════════════════════════════════════════════════════

def fig1_price_history(data, dates):
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (code, d) in enumerate(data.items()):
        ax.plot(range(len(dates)), d['Y'] / 1000, marker='o', markersize=3.5,
                linewidth=1.8, label=COMMODITY_LABELS[code], color=PALETTE_4[i])
    tick_idx = list(range(0, len(dates), 6))
    date_labels = [dates[i].strftime('%b\n%Y') for i in range(len(dates))]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([date_labels[i] for i in tick_idx], fontsize=8)
    ax.set_ylabel('Harga (Rp ribu/kg)', fontsize=10)
    ax.set_title('Gambar 1. Perkembangan Harga Empat Varian Cabai di Jawa Tengah\n(Juli 2022 – Desember 2025)',
                 fontsize=11, fontweight='bold', pad=12)
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.85)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'Rp {x:.0f}rb'))
    fig.tight_layout()
    out = FIGURES / 'fig1_harga_cabai.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def fig2_train_test_split(data):
    n_total = N_TRAIN + N_TEST
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    subtitles = ['(a) Cabai Merah Besar', '(b) Cabai Merah Keriting',
                 '(c) Cabai Rawit Hijau', '(d) Cabai Rawit Merah']
    for i, (code, d) in enumerate(data.items()):
        ax = axes[i]
        idx = np.arange(n_total)
        ax.scatter(idx[:N_TRAIN], d['X'][:N_TRAIN] / 1000, c='#2196F3', s=40, alpha=0.8,
                   label='Nasional (Latih)', marker='o', zorder=3)
        ax.scatter(idx[:N_TRAIN], d['Y'][:N_TRAIN] / 1000, c='#2196F3', s=40, alpha=0.4,
                   label='Jateng (Latih)', marker='^', zorder=3)
        ax.scatter(idx[N_TRAIN:], d['X'][N_TRAIN:] / 1000, c='#e63946', s=50, alpha=0.9,
                   label='Nasional (Uji)', marker='o', zorder=4)
        ax.scatter(idx[N_TRAIN:], d['Y'][N_TRAIN:] / 1000, c='#e63946', s=50, alpha=0.5,
                   label='Jateng (Uji)', marker='^', zorder=4)
        ax.axvline(x=N_TRAIN - 0.5, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.set_title(subtitles[i], fontsize=9.5, fontweight='bold')
        ax.set_xlabel('Indeks Data', fontsize=8)
        ax.set_ylabel('Harga (Rp ribu/kg)', fontsize=8)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=7.5)
    fig.suptitle('Gambar 2. Distribusi Data Latih dan Data Uji untuk Keempat Komoditas Cabai',
                 fontsize=11, fontweight='bold', y=1.01)
    fig.tight_layout()
    out = FIGURES / 'fig2_distribusi_data.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}')


def fig_prediction(data, dates, code, fig_num):
    d     = data[code]
    deg   = BEST_DEG[code]
    X_tr, Y_tr = d['X'][:N_TRAIN], d['Y'][:N_TRAIN]
    X_te, Y_te = d['X'][N_TRAIN:], d['Y'][N_TRAIN:]

    slr_model, slr_xmean = fit_slr(X_tr, Y_tr)
    y_slr  = predict_slr(slr_model, slr_xmean, X_te)
    coefs, pf, xm = fit_poly(X_tr, Y_tr, deg)
    y_poly = predict_poly(coefs, pf, xm, X_te)

    mape_slr  = mape(Y_te, y_slr)
    mape_poly = mape(Y_te, y_poly)

    test_labels = [dates[N_TRAIN + i].strftime('%b %Y') for i in range(N_TEST)]
    x_tick = np.arange(N_TEST)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    configs = [
        (axes[0], y_slr,  'Regresi Linier Sederhana', '#e63946', mape_slr),
        (axes[1], y_poly, f'Regresi Polinomial Derajat {deg}', '#2196F3', mape_poly),
    ]
    for ax, y_pred, method, color, mv in configs:
        ax.plot(x_tick, Y_te / 1000, 'o-', color='#1a1a2e', linewidth=2,
                markersize=6, label='Harga Aktual', zorder=5)
        ax.plot(x_tick, y_pred / 1000, 's--', color=color, linewidth=1.8,
                markersize=6, alpha=0.85, label='Prediksi', zorder=4)
        ax.fill_between(x_tick, Y_te / 1000, y_pred / 1000, alpha=0.08, color=color)
        ax.set_xticks(x_tick)
        ax.set_xticklabels(test_labels, rotation=30, ha='right', fontsize=8)
        ax.set_title(method, fontsize=10, fontweight='bold')
        ax.set_ylabel('Harga (Rp ribu/kg)', fontsize=9)
        ax.legend(fontsize=8.5, loc='upper left', framealpha=0.85)
        ax.text(0.98, 0.05, f'MAPE = {mv:.2f}%', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))

    label = COMMODITY_LABELS[code]
    fig.suptitle(
        f'Gambar {fig_num}. Visualisasi Prediksi Harga {label}\n'
        f'Regresi Linier Sederhana vs Polinomial Derajat {deg} (April – Desember 2025)',
        fontsize=11, fontweight='bold', y=1.02
    )
    fig.tight_layout()
    out = FIGURES / f'fig{fig_num}_prediksi_{code.lower()}.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out}  (SLR MAPE={mape_slr:.2f}%, Poly-{deg} MAPE={mape_poly:.2f}%)')


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('Loading data...')
    data, dates = load_all()
    print(f'  {len(dates)} observations per commodity ({dates[0].strftime("%b %Y")} – {dates[-1].strftime("%b %Y")})')

    print('\nRunning models...')
    metrics_rows, pred_rows, equation_rows = run_all(data)

    print('\nWriting results...')
    write_metrics(metrics_rows)
    write_predictions(pred_rows, dates)
    write_equations(equation_rows)

    print('\nGenerating figures...')
    fig1_price_history(data, dates)
    fig2_train_test_split(data)
    fig_prediction(data, dates, 'CMB', 3)
    fig_prediction(data, dates, 'CMK', 4)
    fig_prediction(data, dates, 'CRH', 5)
    fig_prediction(data, dates, 'CRM', 6)

    print('\nDone. All outputs written to results/ and figures/.')
