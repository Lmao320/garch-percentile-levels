# ╔══════════════════════════════════════════════════════════════╗
# ║   CELDA 1 — Instalar librerías (ejecuta solo una vez)       ║
# ║   En Colab: copia cada bloque en una celda separada         ║
# ╚══════════════════════════════════════════════════════════════╝

# !pip install arch yfinance statsmodels scipy -q


# ╔══════════════════════════════════════════════════════════════╗
# ║   CELDA 2 — Imports                                          ║ 
# ╚══════════════════════════════════════════════════════════════╝

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from arch import arch_model
from scipy.stats import entropy
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from itertools import product 
 
# Estilo oscuro para las gráficas (opcional pero bonito en Colab)
plt.style.use("dark_background")


# ╔══════════════════════════════════════════════════════════════╗
# ║   CELDA 3 — Parámetros  ← EDITA AQUÍ                       ║
# ╚══════════════════════════════════════════════════════════════╝

TICKER      = "^NDX"       # Nasdaq 100. Cambia a "ES=F", "EURUSD=X", "BTC-USD", etc.
START_DATE  = "2015-01-01"
N_DAYS      = 5            # horizonte de forecast (días hábiles)
N_SIM       = 10_000       # simulaciones Monte Carlo
N_PLOT_SIM  = 150          # trayectorias a graficar (subset visual)
OOS_DAYS    = 60           # días para validación fuera de muestra
RANDOM_SEED = 42

P_ORDERS      = [1, 2]
Q_ORDERS      = [1, 2]
DISTRIBUTIONS = ["normal", "t", "skewt"]

Z_SCORE_CRITICO   =  1.5
Z_SCORE_ACEPTABLE = -1.0
PERCENTIL_VAR     = 5


# ╔══════════════════════════════════════════════════════════════╗
# ║   CELDA 4 — Descarga y preparación de datos                 ║
# ╚══════════════════════════════════════════════════════════════╝

print("="*60)
print(f"  ANÁLISIS GARCH MEJORADO — {TICKER}")
print("="*60)

df     = yf.download(TICKER, start=START_DATE, progress=False)
prices = df["Close"].dropna()

if len(prices) < 500:
    raise ValueError("Se necesitan al menos 500 observaciones.")

returns      = np.log(prices / prices.shift(1)).dropna()
returns.name = "log_returns"
returns_pct  = returns * 100.0

train_pct = returns_pct.iloc[:-OOS_DAYS]
oos_pct   = returns_pct.iloc[-OOS_DAYS:]

# ── Precio en tiempo real (evita el retraso de yfinance download) ──
try:
    ticker_obj  = yf.Ticker(TICKER)
    info        = ticker_obj.fast_info
    P0_realtime = float(info.last_price)

    # Validación: si difiere más de 5% del cierre histórico, algo raro → usar histórico
    P0_hist = float(prices.iloc[-1])
    if abs(P0_realtime - P0_hist) / P0_hist > 0.05:
        print(f"  ⚠️  Precio RT ({P0_realtime:,.2f}) difiere >5% del histórico "
              f"({P0_hist:,.2f}) — usando histórico por seguridad.")
        P0 = P0_hist
    else:
        P0 = P0_realtime

except Exception as e:
    print(f"  ⚠️  No se pudo obtener precio RT ({e}) — usando cierre histórico.")
    P0 = float(prices.iloc[-1])

print(f"\n  Datos       : {prices.index[0].date()} → {prices.index[-1].date()}")
print(f"  Obs totales : {len(returns):,}")
print(f"  Train       : {len(train_pct):,}  |  OOS: {len(oos_pct):,}")
print(f"  Cierre histórico  : ${float(prices.iloc[-1]):,.2f}")
print(f"  Precio usado (RT) : ${P0:,.2f}  ← base para todos los percentiles")


# ╔══════════════════════════════════════════════════════════════╗
# ║   CELDA 5 — Tests estadísticos                              ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n" + "─"*60)
print("  TESTS ESTADÍSTICOS")
print("─"*60)

adf_stat, adf_p, *_ = adfuller(returns)
print(f"  ADF            : stat={adf_stat:.4f}  p={adf_p:.4f}  "
      f"({'estacionaria ✓' if adf_p < 0.05 else 'NO estacionaria ✗'})")

lb_ret  = acorr_ljungbox(returns,    lags=[10], return_df=True)
lb_ret2 = acorr_ljungbox(returns**2, lags=[10], return_df=True)
print(f"  Ljung-Box ret  : p={lb_ret['lb_pvalue'].values[0]:.4f}")
print(f"  Ljung-Box ret² : p={lb_ret2['lb_pvalue'].values[0]:.4f}  "
      f"({'efecto ARCH ✓' if lb_ret2['lb_pvalue'].values[0] < 0.05 else 'sin efecto ARCH'})")

arch_f, arch_p, *_ = het_arch(returns, nlags=10)
print(f"  ARCH LM        : F={arch_f:.4f}  p={arch_p:.4f}  "
      f"({'ARCH presente ✓' if arch_p < 0.05 else 'sin ARCH'})")


# ╔══════════════════════════════════════════════════════════════╗
# ║   CELDA 6 — Selección automática de modelo                  ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n" + "─"*60)
print("  SELECCIÓN AUTOMÁTICA DE MODELO (AIC/BIC)")
print("─"*60)

results_grid = []

for p, q, dist in product(P_ORDERS, Q_ORDERS, DISTRIBUTIONS):
    try:
        m   = arch_model(train_pct, mean="Constant", vol="Garch",
                         p=p, q=q, dist=dist)
        fit = m.fit(disp="off", show_warning=False)
        results_grid.append({"p": p, "q": q, "dist": dist,
                              "AIC": fit.aic, "BIC": fit.bic,
                              "LogL": fit.loglikelihood, "fit": fit})
        print(f"  GARCH({p},{q}) [{dist:6s}] → AIC={fit.aic:9.2f}  BIC={fit.bic:9.2f}")
    except Exception as e:
        print(f"  GARCH({p},{q}) [{dist:6s}] → ERROR: {e}")

grid_df       = pd.DataFrame([{k: v for k, v in r.items() if k != "fit"}
                               for r in results_grid])
best_aic_idx  = grid_df["AIC"].idxmin()
best_bic_idx  = grid_df["BIC"].idxmin()
best_row      = results_grid[best_aic_idx]
model_fit     = best_row["fit"]

print(f"\n  ★ Mejor (AIC): GARCH({best_row['p']},{best_row['q']}) "
      f"dist={best_row['dist']}  AIC={best_row['AIC']:.2f}")
print(f"  ★ Mejor (BIC): GARCH({results_grid[best_bic_idx]['p']},"
      f"{results_grid[best_bic_idx]['q']}) dist={results_grid[best_bic_idx]['dist']}")


# ╔══════════════════════════════════════════════════════════════╗
# ║   CELDA 7 — Diagnóstico y parámetros                        ║
# ╚══════════════════════════════════════════════════════════════╝

std_resid = (model_fit.resid / model_fit.conditional_volatility).dropna()
lb_r2     = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)
print(f"\n  LB residuos²   : p={lb_r2['lb_pvalue'].values[0]:.4f}  "
      f"({'bien especificado ✓' if lb_r2['lb_pvalue'].values[0] > 0.05 else 'revisar orden'})")

params = model_fit.params
omega  = float(params.get("omega",    params.get("Omega",   None)))
alpha  = float(params.get("alpha[1]", params.get("alpha",   None)))
beta   = float(params.get("beta[1]",  params.get("beta",    None)))
mu     = float(params.get("mu",       0.0))
nu     = float(params.get("nu",       30.0))
lam    = float(params.get("lambda",   0.0))

last_h      = float(model_fit.conditional_volatility.iloc[-1]) ** 2
last_r      = float(train_pct.iloc[-1])
persistence = alpha + beta
half_life   = np.log(0.5) / np.log(persistence) if persistence < 1 else np.inf

print(f"\n  ω={omega:.6f}  α={alpha:.6f}  β={beta:.6f}  μ={mu:.6f}")
print(f"  ν={nu:.4f}  Persistencia={persistence:.6f}  Half-life={half_life:.1f}d")


# ╔══════════════════════════════════════════════════════════════╗
# ║   CELDA 8 — Simulación Monte Carlo GARCH                    ║
# ╚══════════════════════════════════════════════════════════════╝

def simulate_garch_paths(n_sim, horizon, omega, alpha, beta, mu,
                          last_h, last_r, nu, lam=0.0,
                          dist="t", random_state=None):
    rng = np.random.default_rng(random_state)

    if dist == "normal":
        draw_z = lambda size: rng.standard_normal(size)
    elif dist == "t":
        scale  = np.sqrt(nu / (nu - 2.0))
        draw_z = lambda size: rng.standard_t(df=nu, size=size) / scale
    elif dist == "skewt":
        scale = np.sqrt(nu / (nu - 2.0))
        def draw_z(size):
            z = rng.standard_t(df=nu, size=size) / scale
            if abs(lam) > 1e-6:
                pos = z > 0
                z[pos]  *= (1 + lam)
                z[~pos] *= (1 - lam)
            return z
    else:
        draw_z = lambda size: rng.standard_normal(size)

    sim_returns = np.zeros((n_sim, horizon))
    sim_h_arr   = np.zeros((n_sim, horizon))
    h_t = np.full(n_sim, last_h)
    r_t = np.full(n_sim, last_r)

    for t in range(horizon):
        z     = draw_z(n_sim)
        h_new = omega + alpha * (r_t - mu)**2 + beta * h_t
        h_new = np.maximum(h_new, 1e-8)
        r_new = mu + np.sqrt(h_new) * z
        sim_h_arr[:, t]   = h_new
        sim_returns[:, t] = r_new
        h_t, r_t = h_new, r_new

    return sim_returns, sim_h_arr


sim_ret_pct, sim_h_arr = simulate_garch_paths(
    n_sim=N_SIM, horizon=N_DAYS,
    omega=omega, alpha=alpha, beta=beta, mu=mu,
    last_h=last_h, last_r=last_r,
    nu=nu, lam=lam, dist=best_row["dist"],
    random_state=RANDOM_SEED
)

sim_log_ret      = sim_ret_pct / 100.0
sim_prices       = np.zeros_like(sim_log_ret)
sim_prices[:, 0] = P0 * np.exp(sim_log_ret[:, 0])
for t in range(1, N_DAYS):
    sim_prices[:, t] = sim_prices[:, t-1] * np.exp(sim_log_ret[:, t])

future_dates = pd.bdate_range(start=prices.index[-1], periods=N_DAYS + 1)[1:]
sim_df       = pd.DataFrame(sim_prices, columns=future_dates)
pct_levels   = [1, 5, 10, 25, 50, 75, 90, 95, 99]
sim_summary  = sim_df.quantile(q=np.array(pct_levels) / 100.0)
acum_log_ret = np.log(sim_prices[:, -1] / P0)

print(f"\n  Simulación completa: {N_SIM:,} trayectorias × {N_DAYS} días ✓")


# ╔══════════════════════════════════════════════════════════════╗
# ║   CELDA 9 — VaR / CVaR acumulado                            ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n" + "─"*60)
print(f"  VAR / CVaR — HORIZONTE {N_DAYS} DÍAS (retorno compuesto)")
print("─"*60)

VaR_99  = np.percentile(acum_log_ret, 1)
VaR_95  = np.percentile(acum_log_ret, 5)
VaR_90  = np.percentile(acum_log_ret, 10)
CVaR_99 = acum_log_ret[acum_log_ret <= VaR_99].mean()
CVaR_95 = acum_log_ret[acum_log_ret <= VaR_95].mean()
CVaR_90 = acum_log_ret[acum_log_ret <= VaR_90].mean()
VaR_95_up = np.percentile(acum_log_ret, 95)
VaR_99_up = np.percentile(acum_log_ret, 99)

prob_ganancia = (acum_log_ret > 0).mean()
esperanza_ret = acum_log_ret.mean()

print(f"\n  {'Métrica':<25} {'Log-ret':>10}  {'Precio':>12}")
print(f"  {'─'*50}")
for label, val in [("VaR 99%",  VaR_99), ("VaR 95%",  VaR_95), ("VaR 90%",  VaR_90),
                   ("CVaR 99%", CVaR_99), ("CVaR 95%", CVaR_95), ("CVaR 90%", CVaR_90),
                   ("Upside 95%", VaR_95_up), ("Upside 99%", VaR_99_up)]:
    print(f"  {label:<25} {val:>10.5f}  {P0*np.exp(val):>12.2f}")

print(f"\n  Prob. ganancia ({N_DAYS}d) : {prob_ganancia:.2%}")
print(f"  Retorno esperado        : {esperanza_ret*100:.4f}%")


# ╔══════════════════════════════════════════════════════════════╗
# ║   CELDA 10 — Señal de trading                               ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n" + "─"*60)
print("  SEÑAL DE TRADING")
print("─"*60)

cond_vol     = model_fit.conditional_volatility / 100.0
rolling_mean = cond_vol.rolling(252).mean()
rolling_std  = cond_vol.rolling(252).std()
z_vol        = ((cond_vol - rolling_mean) / rolling_std).dropna()

current_vol  = float(cond_vol.iloc[-1])
current_z    = float(z_vol.iloc[-1])
vol_pct_rank = float((cond_vol <= current_vol).mean() * 100)

if current_z > Z_SCORE_CRITICO:
    signal_label  = "🔴 CRÍTICA — Volatilidad elevada"
    recomendacion = "Reducir tamaño. Alta probabilidad de movimientos extremos."
elif current_z < Z_SCORE_ACEPTABLE:
    signal_label  = "🟢 ACEPTABLE — Volatilidad comprimida"
    recomendacion = "Condiciones favorables. Ideal para estrategias de breakout."
else:
    signal_label  = "🟡 NEUTRAL — Volatilidad media"
    recomendacion = "Tamaño estándar. Monitorear dirección del z-score."

if   vol_pct_rank >= 80: regimen = "Régimen ALTO (pct ≥ 80)"
elif vol_pct_rank <= 20: regimen = "Régimen BAJO (pct ≤ 20)"
else:                    regimen = "Régimen MEDIO (pct 20-80)"

print(f"\n  Vol. diaria actual        : {current_vol*100:.4f}%")
print(f"  Z-score (vs 252d MA)      : {current_z:+.3f}")
print(f"  Percentil histórico       : {vol_pct_rank:.1f}%  → {regimen}")
print(f"  Movimiento esperado 1σ/día: ±{current_vol*100:.3f}%  (≈ ±{P0*current_vol:,.0f} pts)")
print(f"\n  SEÑAL  : {signal_label}")
print(f"  ACCIÓN : {recomendacion}")


# ╔══════════════════════════════════════════════════════════════╗
# ║   CELDA 11 — Rolling forecast OOS                           ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n" + "─"*60)
print(f"  ROLLING FORECAST OOS ({OOS_DAYS} días) — puede tardar ~1-2 min")
print("─"*60)

all_returns_pct   = returns_pct.copy()
train_size        = len(train_pct)
rolling_forecasts = []
rolling_realized  = []
rolling_errors    = []

for i in range(OOS_DAYS):
    window = all_returns_pct.iloc[:train_size + i]
    try:
        m_roll   = arch_model(window, mean="Constant", vol="Garch",
                              p=best_row["p"], q=best_row["q"],
                              dist=best_row["dist"])
        fit_roll  = m_roll.fit(disp="off", show_warning=False)
        fcast     = fit_roll.forecast(horizon=1, reindex=False)
        vol_fcast = float(np.sqrt(fcast.variance.values[-1, 0])) / 100.0
        realized  = float(all_returns_pct.iloc[train_size + i]) / 100.0
        rolling_forecasts.append(vol_fcast)
        rolling_realized.append(abs(realized))
        rolling_errors.append(abs(vol_fcast - abs(realized)))
    except Exception:
        rolling_forecasts.append(np.nan)
        rolling_realized.append(np.nan)
        rolling_errors.append(np.nan)

rf_arr  = np.array(rolling_forecasts, dtype=float)
rr_arr  = np.array(rolling_realized,  dtype=float)
mae_oos = np.nanmean(rolling_errors)
corr    = np.corrcoef(rf_arr[~np.isnan(rf_arr)], rr_arr[~np.isnan(rr_arr)])[0, 1]

print(f"  MAE rolling OOS           : {mae_oos*100:.5f}%")
print(f"  Correlación forecast/real : {corr:.4f}  "
      f"({'buen poder predictivo ✓' if corr > 0.3 else 'correlación baja'})")


# ╔══════════════════════════════════════════════════════════════╗
# ║   CELDA 12 — Entropía y reporte final                       ║
# ╚══════════════════════════════════════════════════════════════╝

hist_vol    = cond_vol.iloc[-500:]
vol_bins    = pd.cut(hist_vol, bins=20, labels=False)
vol_counts  = np.bincount(vol_bins.dropna().astype(int), minlength=20)
vol_dist    = vol_counts / vol_counts.sum()
mkt_entropy = entropy(vol_dist, base=2)
norm_entropy = mkt_entropy / np.log2(20)

print("\n" + "═"*60)
print("  REPORTE FINAL CONSOLIDADO")
print("═"*60)

reporte = pd.DataFrame({
    "Métrica": [
        "Ticker", "Precio actual", "Mejor modelo", "Distribución",
        "Persistencia (α+β)", "Half-life (días)",
        "Vol. diaria actual", "Z-score vol (252d)", "Percentil hist. vol",
        f"VaR 95% ({N_DAYS}d)", f"CVaR 95% ({N_DAYS}d)",
        f"Prob. ganancia ({N_DAYS}d)", f"Retorno esperado ({N_DAYS}d)",
        f"Upside 95% ({N_DAYS}d)",
        "MAE rolling OOS", "Corr. forecast/real",
        "Entropía mercado (norm.)", "Señal de trading",
    ],
    "Valor": [
        TICKER, f"${P0:,.2f}",
        f"GARCH({best_row['p']},{best_row['q']})", best_row["dist"],
        f"{persistence:.6f}", f"{half_life:.1f}",
        f"{current_vol*100:.4f}%", f"{current_z:+.3f}", f"{vol_pct_rank:.1f}%",
        f"{VaR_95:.5f}  (${P0*np.exp(VaR_95):,.2f})",
        f"{CVaR_95:.5f}  (${P0*np.exp(CVaR_95):,.2f})",
        f"{prob_ganancia:.2%}", f"{esperanza_ret*100:.4f}%",
        f"{VaR_95_up:.5f}  (${P0*np.exp(VaR_95_up):,.2f})",
        f"{mae_oos*100:.5f}%", f"{corr:.4f}",
        f"{norm_entropy:.2%}", signal_label,
    ]
})

print(reporte.to_string(index=False))


# ╔══════════════════════════════════════════════════════════════╗
# ║   CELDA 13 — Gráficas                                       ║
# ╚══════════════════════════════════════════════════════════════╝

BG    = "#0e1420"
SURF  = "#080c10"
BORDE = "#1e2d42"
VERDE = "#00d4aa"
AZUL  = "#0a8fff"
ORO   = "#c9a84c"
ROJO  = "#e85d5d"
TEXTO = "#e8edf4"
MUTED = "#6a7d95"

def estilo_ax(ax):
    ax.set_facecolor(SURF)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.title.set_color(TEXTO)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_color(BORDE)

# ── Figura 1: Datos y diagnóstico ─────────────────────────────
fig1, axes = plt.subplots(3, 2, figsize=(16, 12))
fig1.set_facecolor(BG)
fig1.suptitle(f"{TICKER} — GARCH({best_row['p']},{best_row['q']}) dist={best_row['dist']}",
              fontsize=14, fontweight="bold", color=TEXTO)
for ax in axes.flat:
    estilo_ax(ax)

axes[0,0].plot(prices, color=VERDE, linewidth=1)
axes[0,0].set_title("Precio de cierre")

axes[0,1].plot(returns, color=AZUL, linewidth=0.7, alpha=0.8)
axes[0,1].axhline(0, color=BORDE, linewidth=1)
axes[0,1].set_title("Rendimientos logarítmicos diarios")

axes[1,0].plot(cond_vol * 100, color=ORO, linewidth=1)
axes[1,0].axhline(float(cond_vol.mean()) * 100, color=ROJO,
                  linestyle="--", linewidth=1, label=f"Media {cond_vol.mean()*100:.3f}%")
axes[1,0].set_title("Volatilidad condicional (% diario)")
axes[1,0].legend(fontsize=8, labelcolor=MUTED, facecolor=BG, edgecolor=BORDE)

axes[1,1].plot(z_vol, color=ROJO, linewidth=1)
axes[1,1].axhline(Z_SCORE_CRITICO,   color="red",   linestyle="--", linewidth=1, label=f"Crítico {Z_SCORE_CRITICO}")
axes[1,1].axhline(Z_SCORE_ACEPTABLE, color="green", linestyle="--", linewidth=1, label=f"Aceptable {Z_SCORE_ACEPTABLE}")
axes[1,1].axhline(0, color=BORDE, linewidth=0.8)
axes[1,1].scatter([z_vol.index[-1]], [current_z], color="white", zorder=5, s=50)
axes[1,1].set_title("Z-score volatilidad (ventana 252d)")
axes[1,1].legend(fontsize=8, labelcolor=MUTED, facecolor=BG, edgecolor=BORDE)

axes[2,0].plot(std_resid, color=MUTED, linewidth=0.6, alpha=0.8)
axes[2,0].axhline( 3, color=ROJO, linestyle="--", linewidth=0.8)
axes[2,0].axhline(-3, color=ROJO, linestyle="--", linewidth=0.8)
axes[2,0].axhline(0, color=BORDE)
axes[2,0].set_title("Residuos estandarizados")

model_labels = [f"GARCH({r['p']},{r['q']})\n{r['dist']}" for r in results_grid]
aic_vals     = [r["AIC"] for r in results_grid]
colors_bar   = [VERDE if i == best_aic_idx else BORDE for i in range(len(results_grid))]
axes[2,1].barh(model_labels, aic_vals, color=colors_bar, edgecolor=BORDE)
axes[2,1].set_title("Comparación AIC (verde = mejor)")
axes[2,1].invert_xaxis()

fig1.tight_layout()
plt.show()

# ── Figura 2: Forecast + Monte Carlo ──────────────────────────
fig2 = plt.figure(figsize=(16, 10))
fig2.set_facecolor(BG)
gs      = gridspec.GridSpec(2, 2, figure=fig2)
ax_main = fig2.add_subplot(gs[0, :])
ax_mc   = fig2.add_subplot(gs[1, 0])
ax_hist = fig2.add_subplot(gs[1, 1])
for ax in [ax_main, ax_mc, ax_hist]:
    estilo_ax(ax)

fig2.suptitle(f"{TICKER} — Forecast Monte Carlo ({N_SIM:,} sim, {N_DAYS}d)",
              fontsize=14, fontweight="bold", color=TEXTO)

hist_slice = prices.iloc[-120:]
p5  = sim_summary.loc[0.05].values
p25 = sim_summary.loc[0.25].values
p50 = sim_summary.loc[0.50].values
p75 = sim_summary.loc[0.75].values
p95 = sim_summary.loc[0.95].values

ax_main.plot(hist_slice, color=TEXTO, linewidth=1.5, label="Histórico (120d)")
ax_main.plot(future_dates, p50, color=VERDE, linestyle="--",
             linewidth=2, marker="o", markersize=5, label="Mediana forecast")
ax_main.fill_between(future_dates, p5,  p95, color=VERDE, alpha=0.08, label="IC 90%")
ax_main.fill_between(future_dates, p25, p75, color=VERDE, alpha=0.18, label="IC 50%")
ax_main.axvline(prices.index[-1], color=BORDE, linestyle="--", linewidth=1)
ax_main.set_title("Precio histórico + Forecast GARCH")
ax_main.legend(fontsize=9, labelcolor=MUTED, facecolor=BG, edgecolor=BORDE)

rng_plot = np.random.default_rng(0)
idx_plot = rng_plot.choice(N_SIM, size=min(N_PLOT_SIM, N_SIM), replace=False)
for i in idx_plot:
    ax_mc.plot(future_dates, sim_prices[i], color=VERDE, alpha=0.04, linewidth=0.8)
ax_mc.plot(future_dates, sim_prices.mean(axis=0), color=ROJO, linewidth=2, label="Promedio")
ax_mc.plot(future_dates, p50, color="white", linewidth=1.5, linestyle="--", label="Mediana")
ax_mc.set_title(f"Trayectorias Monte Carlo ({N_PLOT_SIM} subset)")
ax_mc.legend(fontsize=8, labelcolor=MUTED, facecolor=BG, edgecolor=BORDE)

ax_hist.hist(acum_log_ret, bins=80, color=AZUL, alpha=0.7, edgecolor="none", density=True)
ax_hist.axvline(VaR_95,    color=ROJO,  linewidth=1.5, label=f"VaR 95%: {VaR_95:.4f}")
ax_hist.axvline(CVaR_95,   color=ORO,   linewidth=1.5, linestyle="--", label=f"CVaR 95%: {CVaR_95:.4f}")
ax_hist.axvline(VaR_95_up, color=VERDE, linewidth=1.5, label=f"Upside 95%: {VaR_95_up:.4f}")
ax_hist.axvline(0, color=MUTED, linewidth=1)
ax_hist.set_title(f"Distribución retorno acumulado ({N_DAYS}d)")
ax_hist.legend(fontsize=8, labelcolor=MUTED, facecolor=BG, edgecolor=BORDE)

fig2.tight_layout()
plt.show()

# ── Figura 3: Rolling OOS ──────────────────────────────────────
oos_dates = oos_pct.index[:OOS_DAYS]
fig3, axes3 = plt.subplots(2, 1, figsize=(14, 8))
fig3.set_facecolor(BG)
fig3.suptitle("Validación fuera de muestra — Rolling Forecast",
              fontsize=14, fontweight="bold", color=TEXTO)
for ax in axes3:
    estilo_ax(ax)

axes3[0].plot(oos_dates, rf_arr * 100, color=VERDE, linewidth=1.5, label="Forecast vol (%)")
axes3[0].plot(oos_dates, rr_arr * 100, color=ORO, linewidth=1, alpha=0.8, label="|Retorno real| (%)")
axes3[0].fill_between(oos_dates, rf_arr * 100, rr_arr * 100, alpha=0.15, color=ROJO)
axes3[0].set_title(f"Vol forecast vs realizada — MAE={mae_oos*100:.4f}%  Corr={corr:.4f}", color=TEXTO)
axes3[0].legend(fontsize=9, labelcolor=MUTED, facecolor=BG, edgecolor=BORDE)

bar_colors = [ROJO if e > mae_oos else VERDE for e in rolling_errors]
axes3[1].bar(oos_dates, np.array(rolling_errors) * 100, color=bar_colors, alpha=0.8, width=0.8)
axes3[1].axhline(mae_oos * 100, color="white", linestyle="--", linewidth=1,
                 label=f"MAE={mae_oos*100:.4f}%")
axes3[1].set_title("Error absoluto de forecast (rojo = sobre MAE)", color=TEXTO)
axes3[1].legend(fontsize=9, labelcolor=MUTED, facecolor=BG, edgecolor=BORDE)

fig3.tight_layout()
plt.show()

print("\n" + "═"*60)
print("  ✅ ANÁLISIS COMPLETO FINALIZADO")
print("═"*60)


# ╔══════════════════════════════════════════════════════════════╗
# ║   CELDA 14 — CLASIFICADOR DE DÍA POR PERCENTILES            ║
# ║   Lógica operativa:                                         ║
# ║   • Precio entre P25 y P75  → Día de REVERSIÓN             ║
# ║   • Precio bajo P25         → Día DIRECCIONAL BAJISTA       ║
# ║   • Precio sobre P75        → Día DIRECCIONAL ALCISTA       ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n" + "═"*60)
print("  CLASIFICADOR DE DÍA POR PERCENTILES GARCH")
print("═"*60)

# ── Niveles de precio para HOY (día 1 del forecast) ──────────
# Usamos el primer día del horizonte ya que es el operativamente relevante
dia_objetivo = 0   # 0 = mañana, 1 = pasado mañana, etc. (máx N_DAYS-1)

precios_dia  = sim_prices[:, dia_objetivo]
fecha_dia    = future_dates[dia_objetivo]

p05_d = np.percentile(precios_dia, 5)
p25_d = np.percentile(precios_dia, 25)
p50_d = np.percentile(precios_dia, 50)
p75_d = np.percentile(precios_dia, 75)
p95_d = np.percentile(precios_dia, 95)

# Retornos correspondientes a cada nivel
r05 = (p05_d / P0 - 1) * 100
r25 = (p25_d / P0 - 1) * 100
r50 = (p50_d / P0 - 1) * 100
r75 = (p75_d / P0 - 1) * 100
r95 = (p95_d / P0 - 1) * 100

# Ancho del rango de reversión (P75 - P25) en puntos y %
ancho_rango_pts = p75_d - p25_d
ancho_rango_pct = (ancho_rango_pts / P0) * 100

# Probabilidades de cada zona
prob_bajo_p25  = (precios_dia < p25_d).mean()   # direccional bajista
prob_en_rango  = ((precios_dia >= p25_d) & (precios_dia <= p75_d)).mean()  # reversión
prob_sobre_p75 = (precios_dia > p75_d).mean()   # direccional alcista

print(f"\n  Activo          : {TICKER}")
print(f"  Precio actual   : ${P0:.2f}")
print(f"  Fecha forecast  : {fecha_dia.date()}")

print(f"\n  {'─'*54}")
print(f"  {'Nivel':<12} {'Precio':>12}  {'Retorno':>10}  {'Zona'}")
print(f"  {'─'*54}")
print(f"  {'P95 (techo)':<12} ${p95_d:>11.2f}  {r95:>+9.3f}%  ← extremo alcista")
print(f"  {'P75 (límite)':<12} ${p75_d:>11.2f}  {r75:>+9.3f}%  ← borde DIRECCIONAL ↑")
print(f"  {'P50 (mediana)':<12} ${p50_d:>11.2f}  {r50:>+9.3f}%  ← centro de valor")
print(f"  {'P25 (límite)':<12} ${p25_d:>11.2f}  {r25:>+9.3f}%  ← borde DIRECCIONAL ↓")
print(f"  {'P05 (suelo)':<12} ${p05_d:>11.2f}  {r05:>+9.3f}%  ← extremo bajista")
print(f"  {'─'*54}")

print(f"\n  Rango de reversión (P25→P75) : "
      f"${p25_d:.2f} – ${p75_d:.2f}  "
      f"({ancho_rango_pts:.0f} pts  /  {ancho_rango_pct:.3f}%)")

print(f"\n  PROBABILIDADES DE ZONA:")
print(f"  🔴 Direccional BAJISTA (< P25)  : {prob_bajo_p25:.2%}")
print(f"  🟡 Reversión       (P25 – P75)  : {prob_en_rango:.2%}")
print(f"  🟢 Direccional ALCISTA (> P75)  : {prob_sobre_p75:.2%}")

# ── Clasificación del día ─────────────────────────────────────
if prob_en_rango >= 0.50:
    tipo_dia   = "🟡 DÍA DE REVERSIÓN"
    estrategia = (
        "El precio tiene alta probabilidad de mantenerse entre P25 y P75.\n"
        "  → Estrategia: fade los extremos del rango, tomar utilidades en la mediana (P50).\n"
        "  → Evita breakouts — el mercado probablemente rechaza niveles extremos."
    )
elif prob_sobre_p75 > prob_bajo_p25:
    tipo_dia   = "🟢 DÍA DIRECCIONAL ALCISTA"
    estrategia = (
        "Alta probabilidad de precio sobre P75. Sesgo comprador.\n"
        "  → Estrategia: buscar entradas long en retrocesos hacia P50.\n"
        "  → Stop referencial bajo P25. Target inicial: P95."
    )
else:
    tipo_dia   = "🔴 DÍA DIRECCIONAL BAJISTA"
    estrategia = (
        "Alta probabilidad de precio bajo P25. Sesgo vendedor.\n"
        "  → Estrategia: buscar entradas short en rebotes hacia P50.\n"
        "  → Stop referencial sobre P75. Target inicial: P05."
    )

print(f"\n  CLASIFICACIÓN DEL DÍA : {tipo_dia}")
print(f"  ESTRATEGIA SUGERIDA   :\n  {estrategia}")

# ── Tabla completa por cada día del horizonte ─────────────────
print(f"\n  {'─'*70}")
print(f"  NIVELES COMPLETOS — TODOS LOS DÍAS DEL HORIZONTE ({N_DAYS}d)")
print(f"  {'─'*70}")
print(f"  {'Fecha':<14} {'P05':>10} {'P25':>10} {'P50':>10} {'P75':>10} {'P95':>10}  {'Tipo'}")
print(f"  {'─'*70}")

for i, fecha in enumerate(future_dates):
    pd_i   = sim_prices[:, i]
    p05_i  = np.percentile(pd_i, 5)
    p25_i  = np.percentile(pd_i, 25)
    p50_i  = np.percentile(pd_i, 50)
    p75_i  = np.percentile(pd_i, 75)
    p95_i  = np.percentile(pd_i, 95)
    pr_en  = ((pd_i >= p25_i) & (pd_i <= p75_i)).mean()
    pr_up  = (pd_i > p75_i).mean()
    pr_dn  = (pd_i < p25_i).mean()
    if   pr_en >= 0.50:        emoji = "🟡 Reversión"
    elif pr_up > pr_dn:        emoji = "🟢 Direcc. ↑"
    else:                      emoji = "🔴 Direcc. ↓"
    print(f"  {str(fecha.date()):<14} "
          f"{p05_i:>10.2f} {p25_i:>10.2f} {p50_i:>10.2f} "
          f"{p75_i:>10.2f} {p95_i:>10.2f}  {emoji}")

print(f"  {'─'*70}")

# ── Gráfica de percentiles por día ────────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(16, 7))
fig4.set_facecolor(BG)
fig4.suptitle(f"{TICKER} — Niveles de Percentiles Operativos (forecast {N_DAYS}d)",
              fontsize=14, fontweight="bold", color=TEXTO)

for ax in axes4:
    estilo_ax(ax)

# Panel izquierdo: bandas de percentiles día a día
dias_num = list(range(1, N_DAYS + 1))
p05_arr  = [np.percentile(sim_prices[:, i],  5) for i in range(N_DAYS)]
p25_arr  = [np.percentile(sim_prices[:, i], 25) for i in range(N_DAYS)]
p50_arr  = [np.percentile(sim_prices[:, i], 50) for i in range(N_DAYS)]
p75_arr  = [np.percentile(sim_prices[:, i], 75) for i in range(N_DAYS)]
p95_arr  = [np.percentile(sim_prices[:, i], 95) for i in range(N_DAYS)]

# Zona de reversión (P25–P75) — verde suave
axes4[0].fill_between(dias_num, p25_arr, p75_arr,
                       color=VERDE, alpha=0.20, label="Zona Reversión (P25–P75)")
# Zona direccional alcista (P75–P95) — azul
axes4[0].fill_between(dias_num, p75_arr, p95_arr,
                       color=AZUL, alpha=0.15, label="Zona Direccional ↑ (P75–P95)")
# Zona direccional bajista (P05–P25) — rojo
axes4[0].fill_between(dias_num, p05_arr, p25_arr,
                       color=ROJO, alpha=0.15, label="Zona Direccional ↓ (P05–P25)")

# Líneas de percentiles
axes4[0].plot(dias_num, p95_arr, color=AZUL,  linewidth=1.5, linestyle="--", label="P95")
axes4[0].plot(dias_num, p75_arr, color=AZUL,  linewidth=2,   label="P75")
axes4[0].plot(dias_num, p50_arr, color=TEXTO, linewidth=2.5, label="P50 (mediana)")
axes4[0].plot(dias_num, p25_arr, color=ROJO,  linewidth=2,   label="P25")
axes4[0].plot(dias_num, p05_arr, color=ROJO,  linewidth=1.5, linestyle="--", label="P05")

# Precio actual como referencia
axes4[0].axhline(P0, color=ORO, linewidth=1.5, linestyle=":",
                 label=f"Precio actual ${P0:.0f}")

# Etiquetas de precio en el último día
offset = (p95_arr[-1] - p05_arr[-1]) * 0.02
for val, label, col in [
    (p95_arr[-1], f"P95\n${p95_arr[-1]:.0f}", AZUL),
    (p75_arr[-1], f"P75\n${p75_arr[-1]:.0f}", AZUL),
    (p50_arr[-1], f"P50\n${p50_arr[-1]:.0f}", TEXTO),
    (p25_arr[-1], f"P25\n${p25_arr[-1]:.0f}", ROJO),
    (p05_arr[-1], f"P05\n${p05_arr[-1]:.0f}", ROJO),
]:
    axes4[0].annotate(label, xy=(N_DAYS, val),
                      xytext=(N_DAYS + 0.1, val),
                      fontsize=8, color=col,
                      va="center", fontfamily="monospace")

axes4[0].set_xlim(0.8, N_DAYS + 0.9)
axes4[0].set_xticks(dias_num)
axes4[0].set_xticklabels([str(d.date()) for d in future_dates],
                          rotation=30, ha="right", fontsize=8)
axes4[0].set_title("Bandas de percentiles por día", color=TEXTO)
axes4[0].set_ylabel("Precio", color=MUTED)
axes4[0].legend(fontsize=8, labelcolor=MUTED, facecolor=BG,
                edgecolor=BORDE, loc="upper left")

# Panel derecho: distribución del día 1 con zonas marcadas
axes4[1].hist(precios_dia, bins=100, color=AZUL, alpha=0.5,
              edgecolor="none", density=True, label="Distribución día 1")

ymax = axes4[1].get_ylim()[1]
# Zonas coloreadas
axes4[1].axvspan(p05_d, p25_d, color=ROJO,  alpha=0.15, label="Zona Direccional ↓")
axes4[1].axvspan(p25_d, p75_d, color=VERDE, alpha=0.15, label="Zona Reversión")
axes4[1].axvspan(p75_d, p95_d, color=AZUL,  alpha=0.15, label="Zona Direccional ↑")

# Líneas verticales de cada percentil
for val, lbl, col, ls in [
    (p05_d, "P05", ROJO,  "--"),
    (p25_d, "P25", ROJO,  "-"),
    (p50_d, "P50", TEXTO, "-"),
    (p75_d, "P75", AZUL,  "-"),
    (p95_d, "P95", AZUL,  "--"),
    (P0,    "HOY", ORO,   ":"),
]:
    axes4[1].axvline(val, color=col, linewidth=1.8, linestyle=ls)
    axes4[1].text(val, ymax * 0.95, f" {lbl}\n ${val:.0f}",
                  color=col, fontsize=8, va="top", fontfamily="monospace")

# Probabilidades como texto en cada zona
mid_dn  = (p05_d + p25_d) / 2
mid_rv  = (p25_d + p75_d) / 2
mid_up  = (p75_d + p95_d) / 2
for mid, prob, col, lbl in [
    (mid_dn, prob_bajo_p25,  ROJO,  f"↓ {prob_bajo_p25:.1%}"),
    (mid_rv, prob_en_rango,  VERDE, f"↔ {prob_en_rango:.1%}"),
    (mid_up, prob_sobre_p75, AZUL,  f"↑ {prob_sobre_p75:.1%}"),
]:
    axes4[1].text(mid, ymax * 0.60, lbl, color=col,
                  fontsize=11, fontweight="bold", ha="center", fontfamily="monospace")

axes4[1].set_title(f"Distribución día 1 ({fecha_dia.date()}) — {tipo_dia}",
                   color=TEXTO)
axes4[1].set_xlabel("Precio", color=MUTED)
axes4[1].legend(fontsize=8, labelcolor=MUTED, facecolor=BG, edgecolor=BORDE)

fig4.tight_layout()
plt.show()

print("\n" + "═"*60)
print("  ✅ ANÁLISIS COMPLETO FINALIZADO")
print("═"*60)