"""
ForecastGarch.py · Niveles diarios por percentiles con GARCH
============================================================

Pensado para correrlo una vez al día, antes o durante la sesión.

  1. Descarga el histórico diario y usa SOLO sesiones completas. Si lo corres
     con el mercado abierto, la vela de hoy (incompleta) se descarta: los
     niveles son para la sesión de hoy y salen iguales a cualquier hora.
  2. Elige el mejor GARCH por AIC.
  3. Calcula P5, P25, P50, P75 y P95 del CIERRE de la sesión, medidos desde el
     último cierre completo. Son los cuantiles exactos del modelo (sin ruido de
     simulación), así que con los mismos datos siempre dan lo mismo.
  4. Comprueba los niveles fuera de muestra (últimos DIAS_EVALUACION días):
       - Llega:  % de sesiones en que el precio tocó el nivel.
       - Rebota: de las sesiones que abrieron del lado de adentro y lo tocaron,
                 % que CERRÓ de vuelta adentro. Si el precio se moviera al azar
                 sería ≈50% (principio de reflexión). Más de 50% con p-valor
                 bajo = el nivel tiende a frenar el precio.
       - Calibración: % de cierres por debajo de cada nivel (debería parecerse
                 al percentil: 5%, 25%, 50%, 75%, 95%).
  5. Muestra las gráficas UNA POR UNA: al cerrar una ventana se abre la siguiente.
"""
import sys
import warnings
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from arch import arch_model
from scipy.stats import binomtest
from statsmodels.stats.diagnostic import het_arch

warnings.filterwarnings("ignore")      # arch y statsmodels avisan mucho; así la salida queda limpia
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ╔══════════════════════════════════════════════════════════════╗
# ║   PARÁMETROS  ← EDITA AQUÍ                                  ║
# ╚══════════════════════════════════════════════════════════════╝

TICKER          = "^NDX"         # índice de contado: lo que sigue tu CFD "US 100 Cash"
NOMBRE          = "Nasdaq 100"
START_DATE      = "2015-01-01"
PERCENTILES     = [5, 25, 50, 75, 95]
DIAS_EVALUACION = 750            # ~3 años para comprobar los niveles fuera de muestra
N_DAYS          = 5              # sesiones del forecast de varios días
N_SIM           = 10_000         # simulaciones para el forecast de varios días
RANDOM_SEED     = 42
N_VELAS         = 40             # velas diarias en la gráfica principal

P_ORDERS        = [1, 2]
Q_ORDERS        = [1, 2]
DISTRIBUTIONS   = ["normal", "t", "skewt"]

UMBRAL_DIRECCIONAL = 0.6         # sesión direccional: |cierre - apertura| / rango > 0.6
CIERRE_NY       = (16, 15)       # desde esta hora (NY) la vela del día se da por completa
ALFA            = 0.05           # significancia; se divide entre los 4 niveles probados
MOSTRAR_GRAFICOS = True


# ── Estilo ────────────────────────────────────────────────────
BG, BORDE, GRID = "#0b0f14", "#30363d", "#1b222c"
TEXTO, MUTED = "#e6edf3", "#8b949e"
VERDE, ROJO = "#26a69a", "#ef5350"
ORO, AZUL, GRIS, MORADO = "#f0b429", "#4c9aff", "#c9d1d9", "#c297ff"
COLOR_NIVEL = {5: ORO, 25: AZUL, 50: GRIS, 75: AZUL, 95: ORO}
LINEA_NIVEL = {5: "-", 25: "-", 50: "--", 75: "-", 95: "-"}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": BORDE, "axes.labelcolor": MUTED, "text.color": TEXTO,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": TEXTO,
    "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.facecolor": BG, "legend.edgecolor": BORDE, "legend.labelcolor": MUTED,
    "legend.fontsize": 9,
})

DIAS_ES = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
MESES_ES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def num(v, dec=1, signo=False):
    """Número en formato colombiano, como TradingView en tu PC: 30.008,0"""
    s = f"{v:+,.{dec}f}" if signo else f"{v:,.{dec}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def fecha_larga(ts):
    return f"{DIAS_ES[ts.weekday()]} {ts.day} {MESES_ES[ts.month - 1]} {ts.year}"


def fecha_corta(ts):
    return f"{ts.day} {MESES_ES[ts.month - 1]}"


# ╔══════════════════════════════════════════════════════════════╗
# ║   1 — Datos                                                 ║
# ╚══════════════════════════════════════════════════════════════╝

def descargar():
    df = yf.download(TICKER, start=START_DATE, interval="1d", auto_adjust=True,
                     multi_level_index=False, progress=False)
    if df is None or df.empty:
        raise SystemExit("No llegaron datos de Yahoo Finance. Revisa la conexión o intenta en unos minutos.")
    return df


def preparar(df, ahora):
    """Limpia los datos y decide para qué sesión son los niveles.

    Devuelve (df con sesiones completas, sesión objetivo, precio de la vela
    incompleta de hoy o None).
    """
    df = df[["Open", "High", "Low", "Close"]].astype(float).dropna()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    df.index = idx.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["Close"] > 0].copy()
    # Coherencia OHLC: a veces Yahoo trae la apertura fuera del rango del día
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)

    hoy = ahora.tz_localize(None).normalize()
    fin_sesion = ahora.normalize() + pd.Timedelta(hours=CIERRE_NY[0], minutes=CIERRE_NY[1])
    precio_parcial = None
    if df.index[-1] == hoy and ahora < fin_sesion:
        # La vela de hoy aún no termina: se descarta y los niveles son para HOY
        precio_parcial = float(df["Close"].iloc[-1])
        df = df.iloc[:-1]
        objetivo = hoy
    else:
        # Último día completo → los niveles son para la siguiente sesión
        # (no descuenta festivos: si cae en uno, son para la sesión siguiente)
        objetivo = df.index[-1] + pd.offsets.BDay(1)
    return df, objetivo, precio_parcial


def precio_reciente(base, precio_parcial):
    """Último precio del índice en Yahoo (solo informativo)."""
    try:
        p = float(yf.Ticker(TICKER).fast_info.last_price)
        if np.isfinite(p) and abs(p / base - 1) < 0.10:
            return p
    except Exception:
        pass
    return precio_parcial


# ╔══════════════════════════════════════════════════════════════╗
# ║   2 — Modelo                                                ║
# ╚══════════════════════════════════════════════════════════════╝

def ajustar(datos, p, q, dist):
    return arch_model(datos, mean="Constant", vol="Garch",
                      p=p, q=q, dist=dist).fit(disp="off", show_warning=False)


def seleccionar(train):
    """Ajusta todos los GARCH de la grilla y los ordena por AIC (solo los que convergen)."""
    filas = []
    for p, q, dist in product(P_ORDERS, Q_ORDERS, DISTRIBUTIONS):
        try:
            fit = ajustar(train, p, q, dist)
        except Exception:
            continue
        if fit.convergence_flag == 0 and np.isfinite(fit.aic):
            filas.append({"p": p, "q": q, "dist": dist, "aic": fit.aic, "fit": fit})
    if not filas:
        raise SystemExit("Ningún GARCH convergió con estos datos.")
    return sorted(filas, key=lambda f: f["aic"])


def cuantiles_z(res, probs):
    """Cuantiles de la distribución estandarizada (varianza 1) del modelo."""
    dist = res.model.distribution
    k = dist.num_params
    params_dist = np.asarray(res.params)[-k:] if k else None
    return np.asarray(dist.ppf(np.asarray(probs, dtype=float), params_dist), dtype=float)


def niveles_desde(base, mu, sigma, z):
    """Precio de cada percentil: base · exp(μ + σ·z), con μ y σ en % diario."""
    return base * np.exp((mu + sigma * z) / 100)


# ╔══════════════════════════════════════════════════════════════╗
# ║   3 — Comprobación fuera de muestra                         ║
# ╚══════════════════════════════════════════════════════════════╝

def evaluar(df, ret, spec, fit_train, n_eval):
    """Aplica los niveles a cada sesión de los últimos n_eval días y mide qué pasó.

    Los parámetros salen SOLO de los datos anteriores al periodo evaluado, y la σ
    de cada sesión usa solo información hasta el cierre anterior.
    """
    fijo = arch_model(ret, mean="Constant", vol="Garch", p=spec["p"], q=spec["q"],
                      dist=spec["dist"]).fix(fit_train.params)
    mu = float(fit_train.params["mu"])
    z = cuantiles_z(fit_train, np.array(PERCENTILES) / 100)

    d = df.loc[ret.index].copy()
    d["prev"] = df["Close"].shift(1).loc[ret.index]
    d["sigma"] = fijo.conditional_volatility
    rango = d["High"] - d["Low"]
    d["rango_sigmas"] = (rango / d["prev"] * 100) / d["sigma"]
    d["direccional"] = ((d["Close"] - d["Open"]).abs() / rango.where(rango > 0)) > UMBRAL_DIRECCIONAL
    # Percentil de la σ de cada sesión frente a las 252 anteriores (sin mirar el futuro)
    d["pct_sigma"] = d["sigma"].rolling(253).apply(lambda s: (s[:-1] <= s[-1]).mean() * 100, raw=True)
    ev = d.iloc[-n_eval:].copy()
    ev["regimen"] = pd.cut(ev["pct_sigma"], [-0.1, 100 / 3, 200 / 3, 100.1],
                           labels=["BAJA", "MEDIA", "ALTA"])

    filas = []
    for q_, zq in zip(PERCENTILES, z):
        nivel = niveles_desde(ev["prev"], mu, ev["sigma"], zq)
        fila = {"q": q_, "calib": (ev["Close"] < nivel).mean() * 100}
        if q_ == 50:
            # ¿El precio vuelve a pasar por el centro (≈ cierre anterior) durante la sesión?
            fila.update(llega=((ev["Low"] <= nivel) & (ev["High"] >= nivel)).mean() * 100,
                        n=np.nan, rebota=np.nan, p_frena=np.nan, p_rompe=np.nan)
        else:
            if q_ > 50:   # nivel arriba: ¿llega el máximo? ¿cierra de vuelta debajo?
                llega, adentro, cierra_adentro = ev["High"] >= nivel, ev["Open"] < nivel, ev["Close"] < nivel
            else:         # nivel abajo: ¿llega el mínimo? ¿cierra de vuelta encima?
                llega, adentro, cierra_adentro = ev["Low"] <= nivel, ev["Open"] > nivel, ev["Close"] > nivel
            toques = adentro & llega            # abrió adentro y luego lo tocó (sin contar gaps)
            n = int(toques.sum())
            k = int((toques & cierra_adentro).sum())
            fila.update(
                llega=llega.mean() * 100, n=n,
                rebota=k / n * 100 if n else np.nan,
                p_frena=binomtest(k, n, 0.5, alternative="greater").pvalue if n else np.nan,
                p_rompe=binomtest(k, n, 0.5, alternative="less").pvalue if n else np.nan,
            )
        filas.append(fila)
    return pd.DataFrame(filas).set_index("q"), ev


def lectura(fila, alfa):
    if fila.name == 50:
        return "centro (≈ cierre anterior)"
    if fila["n"] < 20:
        return "pocos casos"
    if fila["p_frena"] < alfa:
        return "✓ suele frenar el precio"
    if fila["p_rompe"] < alfa:
        return "✗ suele romperse"
    return "≈ como al azar"


def zona_de(precio, niveles):
    nombres = [f"P{q}" for q in PERCENTILES]
    if precio < niveles[0]:
        return "debajo de P5"
    if precio > niveles[-1]:
        return "encima de P95"
    for i in range(len(niveles) - 1):
        if niveles[i] <= precio <= niveles[i + 1]:
            return f"entre {nombres[i]} y {nombres[i + 1]}"
    return ""


# ╔══════════════════════════════════════════════════════════════╗
# ║   4 — Gráficas (se muestran una por una)                    ║
# ╚══════════════════════════════════════════════════════════════╝

def repartir(valores, sep):
    """Separa posiciones de etiquetas para que no se encimen (unidades del eje)."""
    valores = np.asarray(valores, dtype=float)
    orden = np.argsort(valores)
    pos = valores[orden].copy()
    for _ in range(500):
        movido = False
        for i in range(1, len(pos)):
            hueco = pos[i] - pos[i - 1]
            if hueco < sep - 1e-9:
                empuje = (sep - hueco) / 2
                pos[i - 1] -= empuje
                pos[i] += empuje
                movido = True
        if not movido:
            break
    salida = np.empty_like(pos)
    salida[orden] = pos
    return salida


def fig_niveles(r):
    """Gráfica principal: últimas velas diarias + niveles para la sesión objetivo."""
    d = r["df"].iloc[-N_VELAS:]
    n = len(d)
    x = np.arange(n)
    xt = n                                   # columna de la sesión objetivo
    niveles, base, tabla = r["niveles"], r["base"], r["tabla"]

    fig = plt.figure(figsize=(14.5, 7.8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3.2, 1.35], wspace=0.0,
                           left=0.065, right=0.99, top=0.85, bottom=0.13)
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharey=ax)

    # Velas
    sube = (d["Close"] >= d["Open"]).to_numpy()
    col = np.where(sube, VERDE, ROJO)
    ax.vlines(x, d["Low"], d["High"], color=col, lw=1.1, zorder=3)
    cuerpo = np.maximum((d["Close"] - d["Open"]).abs().to_numpy(),
                        (d["High"] - d["Low"]).median() * 0.015)
    ax.bar(x, cuerpo, bottom=np.minimum(d["Open"], d["Close"]), width=0.62,
           color=col, linewidth=0, zorder=4)

    # Columna de la sesión objetivo
    ax.axvspan(xt - 0.5, xt + 0.5, color="white", alpha=0.05, zorder=1)
    ax.text(xt, 1.01, "SESIÓN", transform=ax.get_xaxis_transform(), ha="center",
            va="bottom", fontsize=8, color=MUTED, fontweight="bold")

    # Zonas y líneas de nivel (en el gráfico y en el panel de etiquetas)
    for a in (ax, axr):
        a.axhspan(niveles[1], niveles[3], color=AZUL, alpha=0.07, lw=0, zorder=0)
        a.axhspan(niveles[0], niveles[1], color=ORO, alpha=0.05, lw=0, zorder=0)
        a.axhspan(niveles[3], niveles[4], color=ORO, alpha=0.05, lw=0, zorder=0)
    for q_, v in zip(PERCENTILES, niveles):
        ax.axhline(v, color=COLOR_NIVEL[q_], ls=LINEA_NIVEL[q_], lw=1.3, alpha=0.95, zorder=2)

    actual = r["actual"]
    if actual is not None:
        ax.axhline(actual, color=MORADO, ls=":", lw=1.4, zorder=5)

    # Límites del eje
    valores = list(d["Low"]) + list(d["High"]) + list(niveles) + ([actual] if actual else [])
    lo, hi = min(valores), max(valores)
    pad = (hi - lo) * 0.06
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.8, xt + 0.8)

    # Etiquetas a la derecha, repartidas para que no se encimen
    etiquetas = []
    for q_, v in zip(PERCENTILES, niveles):
        f = tabla.loc[q_]
        linea1 = f"P{q_:<3} {num(v):>10}  {num(v - base, 0, True):>6}"
        if q_ == 50:
            linea2 = f"pasa por aquí {f['llega']:.0f}% de sesiones"
        else:
            marca = {"✓ suele frenar el precio": "  ✓", "✗ suele romperse": "  ✗"}.get(r["lecturas"][q_], "")
            linea2 = f"llega {f['llega']:.0f}% · rebota {f['rebota']:.0f}%{marca}"
        etiquetas.append((v, linea1, linea2, COLOR_NIVEL[q_]))
    if actual is not None:
        etiquetas.append((actual, f"Ahora {num(actual):>11}", "", MORADO))

    alto_eje = ax.get_ylim()[1] - ax.get_ylim()[0]
    pos = repartir([e[0] for e in etiquetas], alto_eje * 0.078)
    trans = axr.get_yaxis_transform()        # x en fracción del panel, y en precio
    for (v, l1, l2, c), y in zip(etiquetas, pos):
        axr.plot([0, 0.05], [v, y], color=c, lw=1, alpha=0.8, transform=trans, clip_on=False)
        sep = alto_eje * 0.017 if l2 else 0
        axr.text(0.07, y + sep, l1, color=c, transform=trans, va="center", ha="left",
                 fontsize=9.5, fontfamily="monospace", fontweight="bold")
        if l2:
            axr.text(0.07, y - sep, l2, color=MUTED, transform=trans, va="center", ha="left",
                     fontsize=8.3, fontfamily="monospace")
    axr.set_xlim(0, 1)
    axr.axis("off")

    # Ejes
    ticks = list(range(n - 1, -1, -5))[::-1]
    ax.set_xticks(ticks + [xt])
    ax.set_xticklabels([fecha_corta(d.index[i]) for i in ticks] + [fecha_corta(r["objetivo"])], fontsize=8.5)
    ax.get_xticklabels()[-1].set_color(TEXTO)
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v, 0)))
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.spines["left"].set_color(BORDE)
    ax.spines["bottom"].set_color(BORDE)

    # Textos
    fig.text(0.065, 0.945, f"{NOMBRE} · Niveles para el {fecha_larga(r['objetivo'])}",
             fontsize=17, fontweight="bold", color=TEXTO)
    fig.text(0.065, 0.905,
             f"Base: cierre del {fecha_corta(r['df'].index[-1])} = {num(base)}   ·   "
             f"σ prevista {num(r['sigma'], 2)}% (±{num(base * r['sigma'] / 100, 0)} pts)   ·   "
             f"Volatilidad {r['regimen']}   ·   {r['modelo']}",
             fontsize=10, color=MUTED)
    fig.text(0.065, 0.03,
             "Niveles = percentiles del CIERRE de la sesión.   Llega: % de sesiones que tocaron el nivel.   "
             "Rebota: de las que abrieron adentro y lo tocaron, % que cerró de vuelta adentro "
             f"(al azar ≈ 50%).   Últimas {r['n_eval']} sesiones, fuera de muestra.",
             fontsize=8.3, color=MUTED)
    return fig


def fig_validacion(r):
    tabla, alfa = r["tabla"], r["alfa"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.6))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.8, bottom=0.12, wspace=0.25)
    fig.suptitle(f"¿Funcionan los niveles?  ·  últimas {r['n_eval']} sesiones, fuera de muestra",
                 fontsize=15, fontweight="bold", x=0.05, ha="left", y=0.95)
    etq = [f"P{q}" for q in tabla.index]
    cols = [COLOR_NIVEL[q] for q in tabla.index]

    a = axs[0]
    barras = a.bar(etq, tabla["llega"], color=cols, alpha=0.85, width=0.6)
    for b, v in zip(barras, tabla["llega"]):
        a.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10, color=TEXTO)
    a.set_ylim(0, 105)
    a.set_title("¿Qué tan seguido llega el precio?", loc="left")
    a.set_ylabel("% de sesiones")

    a = axs[1]
    t = tabla.drop(index=50)
    barras = a.bar([f"P{q}" for q in t.index], t["rebota"], width=0.6,
                   color=[COLOR_NIVEL[q] for q in t.index], alpha=0.85)
    a.axhline(50, color=MUTED, ls="--", lw=1.2, label="Al azar (50%)")
    for b, (q_, f) in zip(barras, t.iterrows()):
        marca = " ✓" if f["p_frena"] < alfa else (" ✗" if f["p_rompe"] < alfa else "")
        a.text(b.get_x() + b.get_width() / 2, f["rebota"] + 2,
               f"{f['rebota']:.0f}%{marca}\nn={int(f['n'])}", ha="center", fontsize=9.5, color=TEXTO)
    a.set_ylim(0, 110)
    a.set_title("Cuando lo toca, ¿cierra de vuelta adentro?", loc="left")
    a.set_ylabel("% de toques")
    a.legend(loc="lower right")

    a = axs[2]
    xs = np.arange(len(tabla))
    a.bar(xs - 0.2, tabla.index, width=0.38, color=MUTED, alpha=0.35, label="Esperado")
    a.bar(xs + 0.2, tabla["calib"], width=0.38, color=cols, alpha=0.9, label="Real")
    for xi, v in zip(xs, tabla["calib"]):
        a.text(xi + 0.2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9, color=TEXTO)
    a.set_xticks(xs)
    a.set_xticklabels(etq)
    a.set_ylim(0, 108)
    a.set_title("Calibración: % de cierres por debajo", loc="left")
    a.legend(loc="upper left")

    for a in axs:
        a.grid(axis="y", color=GRID, lw=0.7)
    fig.text(0.05, 0.025, "✓ = rebota más que al azar (significativo).  ✗ = se rompe más que al azar.  "
             "Sin marca = no se distingue del azar.  n = número de toques.",
             fontsize=8.5, color=MUTED)
    return fig


def fig_volatilidad(r):
    cv = r["cond_vol"].iloc[-DIAS_EVALUACION:]
    ultimo_anio = r["cond_vol"].iloc[-252:]
    q33, q66 = np.percentile(ultimo_anio, [100 / 3, 200 / 3])

    fig = plt.figure(figsize=(14.5, 5.8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1], wspace=0.12,
                           left=0.06, right=0.98, top=0.8, bottom=0.12)
    ax, axh = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    fig.suptitle(f"Volatilidad prevista  ·  {num(r['sigma'], 2)}% para la sesión  ·  {r['regimen']}",
                 fontsize=15, fontweight="bold", x=0.06, ha="left", y=0.95)

    ax.plot(cv.index, cv, color=ORO, lw=1.2, label="σ diaria del modelo (%)")
    ax.axhline(q33, color=VERDE, ls=":", lw=1, label="Límite baja/media (último año)")
    ax.axhline(q66, color=ROJO, ls=":", lw=1, label="Límite media/alta (último año)")
    ax.scatter([r["objetivo"]], [r["sigma"]], color=TEXTO, s=60, zorder=5,
               label=f"Prevista {fecha_corta(r['objetivo'])}")
    ax.set_title("σ diaria (% de cierre a cierre)", loc="left")
    ax.legend(loc="upper left")
    ax.grid(color=GRID, lw=0.7)

    axh.hist(ultimo_anio, bins=30, color=ORO, alpha=0.6, orientation="horizontal")
    axh.axhline(r["sigma"], color=TEXTO, lw=2)
    axh.text(axh.get_xlim()[1] * 0.97, r["sigma"], f"  percentil {r['pct_sigma']:.0f}",
             color=TEXTO, va="bottom", ha="right", fontsize=10, fontweight="bold")
    axh.set_title("Último año", loc="left")
    axh.set_ylim(ax.get_ylim())
    axh.grid(color=GRID, lw=0.7)
    return fig


def fig_varios_dias(r):
    pct, fechas, base = r["mc_pct"], r["mc_fechas"], r["base"]
    x = np.arange(len(fechas))
    fig, ax = plt.subplots(figsize=(13, 6.2))
    fig.subplots_adjust(left=0.08, right=0.86, top=0.84, bottom=0.12)
    fig.suptitle(f"Próximas {len(fechas)} sesiones  ·  percentiles de cierre ({N_SIM:,} simulaciones)".replace(",", "."),
                 fontsize=15, fontweight="bold", x=0.08, ha="left", y=0.95)
    ax.fill_between(x, pct[0], pct[4], color=ORO, alpha=0.08)
    ax.fill_between(x, pct[1], pct[3], color=AZUL, alpha=0.14)
    for i, q_ in enumerate(PERCENTILES):
        ax.plot(x, pct[i], color=COLOR_NIVEL[q_], ls=LINEA_NIVEL[q_], lw=1.6, marker="o", ms=3.5)
        ax.text(x[-1] + 0.08, pct[i, -1], f"P{q_}  {num(pct[i, -1])}", color=COLOR_NIVEL[q_],
                va="center", fontsize=9.5, fontfamily="monospace", fontweight="bold")
    ax.axhline(base, color=MUTED, ls=":", lw=1.2)
    ax.text(x[0] - 0.08, base, f"Base {num(base)}", color=MUTED, ha="right", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([fecha_corta(f) for f in fechas])
    ax.set_xlim(-0.6, x[-1] + 0.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v, 0)))
    ax.grid(color=GRID, lw=0.7)
    return fig


def mostrar_una_por_una(r):
    figuras = [
        ("Niveles para la sesión", fig_niveles),
        ("¿Funcionan los niveles?", fig_validacion),
        ("Volatilidad prevista", fig_volatilidad),
        (f"Próximas {N_DAYS} sesiones", fig_varios_dias),
    ]
    print("\n  Gráficas (cierra cada ventana para ver la siguiente):")
    for i, (titulo, crear) in enumerate(figuras, 1):
        fig = crear(r)
        try:
            fig.canvas.manager.set_window_title(f"{i}/{len(figuras)} · {titulo}")
        except Exception:
            pass
        print(f"   [{i}/{len(figuras)}] {titulo}")
        plt.show()
        plt.close(fig)


# ╔══════════════════════════════════════════════════════════════╗
# ║   5 — Programa                                              ║
# ╚══════════════════════════════════════════════════════════════╝

def main():
    ancho = 74
    ahora = pd.Timestamp.now(tz="America/New_York")
    print("═" * ancho)
    print(f"  GARCH · NIVELES DIARIOS · {NOMBRE} ({TICKER})")
    print("═" * ancho)
    print("  Descargando datos...")
    df, objetivo, parcial = preparar(descargar(), ahora)
    if len(df) < 800:
        raise SystemExit("Se necesitan al menos 800 sesiones de historia.")

    ret = np.log(df["Close"]).diff().dropna() * 100           # retornos en % diario
    n_eval = min(DIAS_EVALUACION, len(ret) // 3)
    train = ret.iloc[:-n_eval]
    print(f"  {len(df):,} sesiones completas: {fecha_corta(df.index[0])} {df.index[0].year}"
          f" → {fecha_corta(df.index[-1])} {df.index[-1].year}".replace(",", "."))
    if parcial is not None:
        print("  La sesión de hoy sigue abierta: su vela se descarta y los niveles son para HOY.")

    print("  Eligiendo modelo y comprobando niveles...")
    ranking = seleccionar(train)
    spec = ranking[0]
    tabla, ev = evaluar(df, ret, spec, spec["fit"], n_eval)
    alfa = ALFA / 4                                            # 4 niveles probados

    # Modelo final con TODOS los datos → estado de hoy
    fit = ajustar(ret, spec["p"], spec["q"], spec["dist"])
    sigma = float(np.sqrt(fit.forecast(horizon=1, reindex=False).variance.values[-1, 0]))
    mu = float(fit.params["mu"])
    base = float(df["Close"].iloc[-1])
    niveles = niveles_desde(base, mu, sigma, cuantiles_z(fit, np.array(PERCENTILES) / 100))

    cond_vol = fit.conditional_volatility
    pct_sigma = (cond_vol.iloc[-252:] <= sigma).mean() * 100
    regimen = "BAJA" if pct_sigma <= 100 / 3 else ("ALTA" if pct_sigma > 200 / 3 else "MEDIA")

    rango_q = ev["rango_sigmas"].quantile([0.25, 0.5, 0.75]).to_numpy() * sigma / 100 * base
    dir_reg = ev.groupby("regimen", observed=True)["direccional"].mean() * 100
    dir_media = ev["direccional"].mean() * 100
    lecturas = {q_: lectura(tabla.loc[q_], alfa) for q_ in PERCENTILES}

    persist = float(fit.params.filter(like="alpha").sum() + fit.params.filter(like="beta").sum())
    arch_p = het_arch(ret, nlags=10)[1]
    modelo = f"GARCH({spec['p']},{spec['q']}) {spec['dist']}"
    actual = precio_reciente(base, parcial)

    # ── Salida principal ─────────────────────────────────────
    print("\n" + "═" * ancho)
    print(f"  NIVELES PARA EL {fecha_larga(objetivo).upper()}")
    print("═" * ancho)
    print(f"  Base: cierre del {fecha_corta(df.index[-1])} = {num(base)}")
    print(f"  σ prevista: {num(sigma, 2)}% (≈ ±{num(base * sigma / 100, 0)} pts)  ·  "
          f"Volatilidad {regimen} (percentil {pct_sigma:.0f} del último año)")
    print("  " + "─" * (ancho - 2))
    print(f"  {'Nivel':<6}{'Precio':>11}{'Dist.':>8}{'Llega':>8}{'Rebota':>8}{'Toques':>8}   Lectura")
    for q_, v in reversed(list(zip(PERCENTILES, niveles))):
        f = tabla.loc[q_]
        reb = "—" if q_ == 50 else f"{f['rebota']:.0f}%"
        toq = "—" if q_ == 50 else f"{int(f['n'])}"
        print(f"  P{q_:<5}{num(v):>11}{num(v - base, 0, True):>8}{f['llega']:>7.0f}%"
              f"{reb:>8}{toq:>8}   {lecturas[q_]}")
    print("  " + "─" * (ancho - 2))
    print(f"  Rango típico de la sesión: ~{num(rango_q[1], 0)} pts "
          f"(la mitad de las sesiones entre {num(rango_q[0], 0)} y {num(rango_q[2], 0)})")
    if regimen in dir_reg.index:
        print(f"  Sesiones direccionales con volatilidad {regimen}: {dir_reg[regimen]:.0f}% "
              f"(promedio {dir_media:.0f}%)")
    if actual is not None:
        if abs(actual - base) < 0.5:
            print(f"  Último precio del índice: {num(actual)} (sin cambios desde el cierre: aún no abre)")
        else:
            print(f"  Último precio del índice: {num(actual)} → {zona_de(actual, niveles)}")
    print("\n  Para TradingView:  " + "   ".join(f"P{q_} {num(v)}" for q_, v in zip(PERCENTILES, niveles)))

    # ── Comprobación ─────────────────────────────────────────
    print("\n" + "─" * ancho)
    print(f"  CÓMO SE COMPORTARON ESTOS NIVELES (últimas {n_eval} sesiones, fuera de muestra)")
    print("─" * ancho)
    print("  Llega:  % de sesiones en que el precio tocó el nivel.")
    print("  Rebota: de las sesiones que abrieron del lado de adentro y lo tocaron,")
    print("          % que cerró de vuelta adentro. Al azar sería ≈50%.")
    print("  Calibración (% de cierres por debajo; ideal 5 · 25 · 50 · 75 · 95):")
    print("    " + "  ·  ".join(f"P{q_} {num(tabla.loc[q_, 'calib'], 1)}%" for q_ in PERCENTILES))

    # ── Varias sesiones ──────────────────────────────────────
    mc_fechas = pd.bdate_range(objetivo, periods=N_DAYS)
    fc = fit.forecast(horizon=N_DAYS, method="simulation", simulations=N_SIM, reindex=False,
                      random_state=np.random.RandomState(RANDOM_SEED))
    caminos = base * np.exp(np.cumsum(fc.simulations.values[-1] / 100, axis=1))
    mc_pct = np.percentile(caminos, PERCENTILES, axis=0)
    print("\n" + "─" * ancho)
    print(f"  PRÓXIMAS {N_DAYS} SESIONES (percentiles de cierre)")
    print("─" * ancho)
    print(f"  {'Sesión':<10}" + "".join(f"{'P' + str(q_):>12}" for q_ in PERCENTILES))
    for j, fecha in enumerate(mc_fechas):
        print(f"  {fecha_corta(fecha):<10}" + "".join(f"{num(v):>12}" for v in mc_pct[:, j]))

    print("\n" + "─" * ancho)
    print(f"  Modelo: {modelo} (mejor AIC de {len(ranking)})  ·  persistencia {num(persist, 3)}  ·  "
          f"efecto ARCH p={arch_p:.3f}")
    print("═" * ancho)

    if MOSTRAR_GRAFICOS:
        r = dict(df=df, objetivo=objetivo, niveles=niveles, base=base, sigma=sigma, regimen=regimen,
                 pct_sigma=pct_sigma, tabla=tabla, lecturas=lecturas, alfa=alfa, n_eval=n_eval,
                 actual=actual, modelo=modelo, cond_vol=cond_vol, mc_pct=mc_pct, mc_fechas=mc_fechas)
        mostrar_una_por_una(r)


if __name__ == "__main__":
    main()