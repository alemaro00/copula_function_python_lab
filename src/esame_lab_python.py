import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sc
import yfinance as yf
import time
import sys
import re
from datetime import datetime
from pathlib import Path
from scipy.stats import jarque_bera, rankdata, kendalltau, norm, t
from scipy.special import gammaln
from numpy.random import multivariate_normal
from copulas.bivariate import Clayton, Frank, Gumbel

#cambiare qualsiasi commodities, azione, strumento finanziario preso da yahoo finance
waahid= "BTC-USD"#"GC=F"#
ithnaan=  "ETH-USD"#"SI=F"#
inizio_periodo="2017-01-01"
fine_periodo="2026-04-10"


def validate_inputs(asset_a, asset_b, start_date, end_date):
    if not isinstance(asset_a, str) or not asset_a.strip():
        raise ValueError("Ticker waahid non valido: inserire una stringa non vuota.")
    if not isinstance(asset_b, str) or not asset_b.strip():
        raise ValueError("Ticker ithnaan non valido: inserire una stringa non vuota.")

    asset_a = asset_a.strip()
    asset_b = asset_b.strip()
    if asset_a == asset_b:
        raise ValueError("I due ticker sono uguali: inserire due strumenti diversi.")

    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            "Formato data non valido. Usa YYYY-MM-DD per inizio_periodo e fine_periodo."
        ) from exc

    if start_dt >= end_dt:
        raise ValueError("Intervallo date non valido: inizio_periodo deve essere minore di fine_periodo.")


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def build_output_log_path(asset_a, asset_b, start_date, end_date):
    output_dir = Path(__file__).resolve().parents[1] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_name = f"{asset_a}&{asset_b}_{start_date}&{end_date}.txt"
    safe_name = re.sub(r'[<>:"/\\|?*]+', "-", raw_name)
    return output_dir / safe_name


LOG_FILE_PATH = build_output_log_path(waahid, ithnaan, inizio_periodo, fine_periodo)
_log_file_handle = LOG_FILE_PATH.open("w", encoding="utf-8", buffering=1)
_original_stdout = sys.stdout
_original_stderr = sys.stderr
sys.stdout = TeeStream(_original_stdout, _log_file_handle)
sys.stderr = TeeStream(_original_stderr, _log_file_handle)

validate_inputs(waahid, ithnaan, inizio_periodo, fine_periodo)

# Toggle visualizzazioni per evitare grafici ridondanti
SHOW_EMPIRICAL_SCATTER = True
SHOW_SIMULATION_COMPARISON = True

#grafico rendimenti
def returns_graphs():
    # Creazione dei due grafici dei rendimenti affiancati
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9))
    # Primo grafico: Rendimenti Oro
    ax1.plot(returns.index, returns[waahid], label=f'Rendimenti {waahid}', color='blue')
    ax1.set_title(f"Rendimenti Logaritmici di {waahid}")
    ax1.set_xlabel("Data")
    ax1.set_ylabel("Rendimento Logaritmico")
    ax1.legend()
    ax1.grid(True)
    # Secondo grafico: Rendimenti Argento
    ax2.plot(returns.index, returns[ithnaan], label=f'Rendimenti {ithnaan}', color='red')
    ax2.set_title(f"Rendimenti Logaritmici di {ithnaan}")
    ax2.set_xlabel("Data")
    ax2.set_ylabel("Rendimento Logaritmico")
    ax2.legend()
    ax2.grid(True)
    # Visualizzazione del grafico completo
    plt.tight_layout()
    plt.show()
#jarque-bera test con H0 dati considerati norm distrib
def norm_test():

    # Test di normalità (Jarque-Bera)
    jb_waahid = jarque_bera(returns[waahid])
    jb_ithnaan = jarque_bera(returns[ithnaan])
    # Applicazione del test e stampa dei risultati una sola volta
    print("\nTest for Normality: Jarque-Bera\n")

    print(f"{waahid} -> p-value: {jb_waahid.pvalue:.4f}")
    if jb_waahid.pvalue < 0.05:
        print("Distribuzione NON normale\n")
    else:
        print("Distribuzione normale\n")

    print(f"{ithnaan} -> p-value: {jb_ithnaan.pvalue:.4f}")
    if jb_ithnaan.pvalue < 0.05:
        print("Distribuzione NON normale\n")
    else:
        print("Distribuzione normale\n")
#grafico qq plot per osservare la normalità
def qqplot_graphs():
    # Q-Q plot per waahid
    sm.qqplot(returns[waahid], line='45', fit=True)
    plt.title(f"Q-Q Plot dei Rendimenti Logaritmici di {waahid}")
    plt.show()

    # Q-Q plot per ithnaan
    sm.qqplot(returns[ithnaan], line='45', fit=True)
    plt.title(f"Q-Q Plot dei Rendimenti Logaritmici di {ithnaan}")
    plt.show()
#grafico distribuzioni marginali
def distr_graphs():
    # Istogrammi con KDE dei rendimenti
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # Calcola il range massimo assoluto per centraggio simmetrico
    max_abs = max(
        abs(returns[waahid].min()), abs(returns[waahid].max()),
        abs(returns[ithnaan].min()), abs(returns[ithnaan].max())
    )
    x_limits = (-max_abs, max_abs)

    # Plot 1 - Waahid
    sns.histplot(returns[waahid], kde=True, color='blue', ax=ax1) #kde=True aggiunge la curva continua sopra l’istogramma, rappresenta la distrib stimata. kernel density estimation
    ax1.axvline(p5_waahid, color='black', label='5° percentile')
    ax1.axvline(p95_waahid, color='black', label='95° percentile')
    ax1.text(p5_waahid, ax1.get_ylim()[1]*(-0.05), f"{p5_waahid:.4f}", color='blue', ha='center')
    ax1.text(p95_waahid, ax1.get_ylim()[1]*(-0.05), f"{p95_waahid:.4f}", color='blue', ha='center')
    ax1.set_xlim(x_limits)
    ax1.set_title(f"Distribuzione dei Rendimenti Logaritmici di {waahid}")
    ax1.set_xlabel("Rendimento Logaritmico")
    ax1.set_ylabel("Frequenza")
    ax1.legend()

    # Plot 2 - Ithnaan
    sns.histplot(returns[ithnaan], kde=True, color='red', ax=ax2)
    ax2.axvline(p5_ithnaan, color='black', label='5° percentile')
    ax2.axvline(p95_ithnaan, color='black', label='95° percentile')
    ax2.text(p5_ithnaan, ax2.get_ylim()[1]*(-0.05), f"{p5_ithnaan:.4f}", color='red', ha='center')
    ax2.text(p95_ithnaan, ax2.get_ylim()[1]*(-0.05), f"{p95_ithnaan:.4f}", color='red', ha='center')
    ax2.set_xlim(x_limits)
    ax2.set_title(f"Distribuzione dei Rendimenti Logaritmici di {ithnaan}")
    ax2.set_xlabel("Rendimento Logaritmico")
    ax2.set_ylabel("Frequenza")
    ax2.legend()

    plt.tight_layout()
    plt.show()
#misure di correlazione, dipendenza e concordanza
def corr_measures():
    #Calcolo della correlazione Pearson [-1,1]
    pearson_rho = returns.corr().loc[waahid, ithnaan]
    #misure di concordanza più robuste di Pearson perchè si osserva la dipendenza tra variabili che può essere non lineare [-1,1]
    kendall_tau, _ = kendalltau(returns[waahid], returns[ithnaan])
    spearman_rho, _ = sc.stats.spearmanr(returns[waahid], returns[ithnaan])
    print(f"\nRho di Pearson tra {waahid} e {ithnaan}:{pearson_rho:.4f}") #dipendenza lineare positiva importante
    print(f"Tau di Kendall tra {waahid} e {ithnaan}: {kendall_tau:.4f}") #trasformando le variabili con funzione crescenti come devo fare, avrò delle misure che restano identiche
    print(f"Rho di Spearman tra {waahid} e {ithnaan}: {spearman_rho:.4f}")

    # Tail dependence [0,1], misurano quanto fortemente due variabili tendono a muoversi insieme nelle code della distribuzione, è una probabilità condizionata
    quantile_bound = 0.05
    empirical_lambda_L = np.mean((u < quantile_bound) & (v < quantile_bound)) / quantile_bound
    empirical_lambda_U = np.mean((u > 1 - quantile_bound) & (v > 1 - quantile_bound)) / quantile_bound
    print(f"Lower Tail Dependence (soglia q={quantile_bound}): {empirical_lambda_L:.4f}")
    print(f"Upper Tail Dependence (soglia q={1 - quantile_bound}): {empirical_lambda_U:.4f}")
# Funzione per plottare confronto copule
def plot_copula_comparison(real_data, sim_data_list, labels, title):
    total_plots = len(sim_data_list) + 1
    if total_plots == 6:
        n_rows, n_cols = 2, 3
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10))
    else:
        n_cols = min(3, total_plots)
        n_rows = int(np.ceil(total_plots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))

    axes = np.array(axes).reshape(-1)

    axes[0].scatter(real_data[:, 0], real_data[:, 1], alpha=0.5)
    axes[0].set_title("Copula empirica (u,v)")

    for i, (sim_data, label) in enumerate(zip(sim_data_list, labels), start=1):
        axes[i].scatter(sim_data[:, 0], sim_data[:, 1], alpha=0.5, color='orange')
        axes[i].set_title(f"Copula {label}")

    for ax in axes[:total_plots]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    for ax in axes[total_plots:]:
        ax.set_visible(False)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def build_empirical_copula_grid(u_vals, v_vals, grid_size=60):
    grid = np.linspace(0, 1, grid_size)
    c_n = np.zeros((grid_size, grid_size))

    for i, uu in enumerate(grid):
        for j, vv in enumerate(grid):
            c_n[j, i] = np.mean((u_vals <= uu) & (v_vals <= vv))

    return grid, c_n


def plot_empirical_copula(u_vals, v_vals, title="Copula empirica C_n(u,v)"):
    plt.figure(figsize=(8, 6))
    plt.scatter(u_vals, v_vals, alpha=0.5, s=18, color="steelblue")
    plt.xlabel("u")
    plt.ylabel("v")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def clayton_cdf(u_vals, v_vals, theta):
    if theta <= 0:
        return np.full_like(u_vals, np.nan, dtype=float)
    eps = 1e-12
    u_safe = np.clip(u_vals, eps, 1)
    v_safe = np.clip(v_vals, eps, 1)
    with np.errstate(over="ignore", invalid="ignore"):
        term = u_safe ** (-theta) + v_safe ** (-theta) - 1
        c_val = term ** (-1 / theta)
    return np.clip(c_val, 0, 1)


def frank_cdf(u_vals, v_vals, theta):
    if np.isclose(theta, 0.0):
        return u_vals * v_vals
    num = (np.exp(-theta * u_vals) - 1) * (np.exp(-theta * v_vals) - 1)
    den = np.exp(-theta) - 1
    c_val = -(1 / theta) * np.log1p(num / den)
    return np.clip(c_val, 0, 1)


def gumbel_cdf(u_vals, v_vals, theta):
    if theta < 1:
        return np.full_like(u_vals, np.nan, dtype=float)
    x = -np.log(np.clip(u_vals, 1e-12, 1))
    y = -np.log(np.clip(v_vals, 1e-12, 1))
    c_val = np.exp(-((x ** theta + y ** theta) ** (1 / theta)))
    return np.clip(c_val, 0, 1)


def gaussian_cdf(u_vals, v_vals, rho):
    if abs(rho) >= 1:
        return np.full_like(u_vals, np.nan, dtype=float)
    u_safe = np.clip(u_vals, 1e-12, 1 - 1e-12)
    v_safe = np.clip(v_vals, 1e-12, 1 - 1e-12)
    x = norm.ppf(u_safe)
    y = norm.ppf(v_safe)
    points = np.column_stack((x, y))
    cov = np.array([[1.0, rho], [rho, 1.0]])
    c_vals = np.array([
        sc.stats.multivariate_normal.cdf(point, mean=np.zeros(2), cov=cov)
        for point in points
    ])
    return np.clip(c_vals, 0, 1)


def student_t_cdf(u_vals, v_vals, rho, nu):
    if abs(rho) >= 1 or nu <= 0:
        return np.full_like(u_vals, np.nan, dtype=float)
    u_safe = np.clip(u_vals, 1e-12, 1 - 1e-12)
    v_safe = np.clip(v_vals, 1e-12, 1 - 1e-12)
    x = t.ppf(u_safe, df=nu)
    y = t.ppf(v_safe, df=nu)
    points = np.column_stack((x, y))
    shape = np.array([[1.0, rho], [rho, 1.0]])
    c_vals = np.array([
        sc.stats.multivariate_t.cdf(point, loc=np.zeros(2), shape=shape, df=nu)
        for point in points
    ])
    return np.clip(c_vals, 0, 1)


def evaluate_copula_cdf_on_grid(grid, family, **params):
    u_mesh, v_mesh = np.meshgrid(grid, grid)
    u_flat = u_mesh.ravel()
    v_flat = v_mesh.ravel()

    if family == "clayton":
        c_flat = clayton_cdf(u_flat, v_flat, params["theta"])
    elif family == "frank":
        c_flat = frank_cdf(u_flat, v_flat, params["theta"])
    elif family == "gumbel":
        c_flat = gumbel_cdf(u_flat, v_flat, params["theta"])
    elif family == "gaussian":
        c_flat = gaussian_cdf(u_flat, v_flat, params["rho"])
    elif family == "student-t":
        c_flat = student_t_cdf(u_flat, v_flat, params["rho"], params["nu"])
    else:
        raise ValueError(f"Famiglia non supportata per CDF su griglia: {family}")

    return c_flat.reshape(u_mesh.shape)


def copula_grid_distance_metrics(c_empirical, c_parametric):
    valid = np.isfinite(c_empirical) & np.isfinite(c_parametric)
    if not np.any(valid):
        return np.nan, np.nan

    diff = c_empirical[valid] - c_parametric[valid]
    mse = np.mean(diff ** 2)
    max_abs = np.max(np.abs(diff))
    return mse, max_abs


def fit_mixture_weights_from_densities(component_densities):
    n_components = component_densities.shape[1]
    w0 = np.full(n_components, 1.0 / n_components)

    def objective(weights):
        mix_density = component_densities @ weights
        return -copula_log_likelihood(mix_density)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_components

    result = sc.optimize.minimize(
        objective,
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    if result.success and np.all(np.isfinite(result.x)):
        weights = np.clip(result.x, 0.0, 1.0)
        weights = weights / np.sum(weights)
        ll = -float(result.fun)
    else:
        weights = w0
        ll = copula_log_likelihood(component_densities @ weights)
        print("Warning: ottimizzazione mixture non convergente, uso pesi uniformi.")

    return weights, ll


def mixture_cdf_on_grid(grid, component_specs, weights):
    c_mix = np.zeros((len(grid), len(grid)), dtype=float)
    for w, (family, params) in zip(weights, component_specs):
        c_mix += w * evaluate_copula_cdf_on_grid(grid, family, **params)
    return np.clip(c_mix, 0, 1)


def clayton_density(u_vals, v_vals, theta):
    if theta <= 0:
        return np.full_like(u_vals, np.nan, dtype=float)

    term = (u_vals ** (-theta) + v_vals ** (-theta) - 1)
    density = (1 + theta) * (u_vals * v_vals) ** (-1 - theta) * term ** (-2 - 1 / theta)
    return density


def frank_density(u_vals, v_vals, theta):
    if np.isclose(theta, 0.0):
        return np.ones_like(u_vals, dtype=float)

    one_minus_exp_theta = 1 - np.exp(-theta)
    one_minus_exp_theta_u = 1 - np.exp(-theta * u_vals)
    one_minus_exp_theta_v = 1 - np.exp(-theta * v_vals)

    numerator = theta * one_minus_exp_theta * np.exp(-theta * (u_vals + v_vals))
    denominator = (one_minus_exp_theta - one_minus_exp_theta_u * one_minus_exp_theta_v) ** 2
    return numerator / denominator


def gumbel_density(u_vals, v_vals, theta):
    if theta < 1:
        return np.full_like(u_vals, np.nan, dtype=float)

    x = -np.log(u_vals)
    y = -np.log(v_vals)
    a = x ** theta + y ** theta
    c_val = np.exp(-a ** (1 / theta))
    multiplier = (x * y) ** (theta - 1) / (u_vals * v_vals)
    shape_term = a ** (2 / theta - 2)
    correction = 1 + (theta - 1) * a ** (-1 / theta)
    return c_val * multiplier * shape_term * correction


def copula_log_likelihood(density_values, eps=1e-300):
    density_array = np.asarray(density_values, dtype=float)
    # Penalizza valori non finiti o non positivi invece di propagare NaN/inf.
    density_array = np.where(np.isfinite(density_array), density_array, 0.0)
    density_clipped = np.clip(density_array, eps, None)
    return float(np.sum(np.log(density_clipped)))


def gaussian_copula_density(u_vals, v_vals, rho):
    if abs(rho) >= 1:
        return np.full_like(u_vals, np.nan, dtype=float)

    x = norm.ppf(u_vals)
    y = norm.ppf(v_vals)
    denom = np.sqrt(1 - rho ** 2)
    exponent = -(rho ** 2 * (x ** 2 + y ** 2) - 2 * rho * x * y) / (2 * (1 - rho ** 2))
    return np.exp(exponent) / denom


def student_t_copula_density(u_vals, v_vals, rho, nu):
    if abs(rho) >= 1 or nu <= 2:
        return np.full_like(u_vals, np.nan, dtype=float)

    x = t.ppf(u_vals, df=nu)
    y = t.ppf(v_vals, df=nu)

    det_r = 1 - rho ** 2
    inv_quad = (x ** 2 - 2 * rho * x * y + y ** 2) / det_r

    log_const_biv = (
        gammaln((nu + 2) / 2)
        - gammaln(nu / 2)
        - np.log(nu * np.pi)
        - 0.5 * np.log(det_r)
    )
    log_kernel_biv = -((nu + 2) / 2) * np.log1p(inv_quad / nu)
    log_biv = log_const_biv + log_kernel_biv

    log_uni_x = np.log(np.clip(t.pdf(x, df=nu), 1e-300, None))
    log_uni_y = np.log(np.clip(t.pdf(y, df=nu), 1e-300, None))

    return np.exp(log_biv - log_uni_x - log_uni_y)


def fit_gaussian_copula_mle(u_vals, v_vals):
    tau_emp, _ = kendalltau(u_vals, v_vals)
    rho0 = np.sin(np.pi * tau_emp / 2)

    def objective(params):
        rho = params[0]
        dens = gaussian_copula_density(u_vals, v_vals, rho)
        return -copula_log_likelihood(dens)

    result = sc.optimize.minimize(
        objective,
        x0=np.array([rho0]),
        bounds=[(-0.99, 0.99)],
        method="L-BFGS-B"
    )
    if result.success and np.isfinite(result.fun):
        rho_hat = float(result.x[0])
        ll = -float(result.fun)
    else:
        rho_hat = float(np.clip(rho0, -0.99, 0.99))
        ll = copula_log_likelihood(gaussian_copula_density(u_vals, v_vals, rho_hat))
        print("Warning: ottimizzazione Gaussian non convergente, uso parametro iniziale.")
    return rho_hat, ll


def fit_student_t_copula_mle(u_vals, v_vals):
    tau_emp, _ = kendalltau(u_vals, v_vals)
    rho0 = np.sin(np.pi * tau_emp / 2)
    nu0 = 8.0

    def objective(params):
        rho, nu = params
        dens = student_t_copula_density(u_vals, v_vals, rho, nu)
        return -copula_log_likelihood(dens)

    result = sc.optimize.minimize(
        objective,
        x0=np.array([rho0, nu0]),
        bounds=[(-0.99, 0.99), (2.01, 200.0)],
        method="L-BFGS-B"
    )
    if result.success and np.isfinite(result.fun):
        rho_hat = float(result.x[0])
        nu_hat = float(result.x[1])
        ll = -float(result.fun)
    else:
        rho_hat = float(np.clip(rho0, -0.99, 0.99))
        nu_hat = float(np.clip(nu0, 2.01, 200.0))
        ll = copula_log_likelihood(student_t_copula_density(u_vals, v_vals, rho_hat, nu_hat))
        print("Warning: ottimizzazione Student-t non convergente, uso parametri iniziali.")
    return rho_hat, nu_hat, ll


def simulate_gaussian_copula(n_obs, rho):
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = multivariate_normal(mean=[0, 0], cov=cov, size=n_obs)
    return norm.cdf(z)


def simulate_student_t_copula(n_obs, rho, nu):
    cov = np.array([[1.0, rho], [rho, 1.0]])
    z = multivariate_normal(mean=[0, 0], cov=cov, size=n_obs)
    w = np.random.chisquare(df=nu, size=n_obs)
    x = z / np.sqrt((w / nu))[:, None]
    return t.cdf(x, df=nu)


def download_close_with_cache(tickers, start, end, max_retries=4, base_wait=3):
    cache_file = Path(__file__).with_name("prezzi_close_cache.csv")
    expected_tickers = list(dict.fromkeys(tickers))

    for attempt in range(1, max_retries + 1):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False
            )

            if isinstance(raw.columns, pd.MultiIndex):
                data = raw["Close"]
            else:
                data = raw.copy()

            data = data.dropna(how="all")
            if not data.empty:
                missing_cols = [col for col in expected_tickers if col not in data.columns]
                if missing_cols:
                    raise RuntimeError(
                        "Ticker non trovati o senza dati su Yahoo nel periodo richiesto: "
                        + ", ".join(missing_cols)
                    )
                data = data[expected_tickers]
                data.to_csv(cache_file)
                print(f"Dati scaricati da Yahoo e salvati in cache: {cache_file.name}")
                return data
        except Exception as exc:
            print(f"Tentativo {attempt}/{max_retries} fallito: {exc}")

        if attempt < max_retries:
            time.sleep(base_wait * attempt)

    if cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        missing_in_cache = [col for col in expected_tickers if col not in cached.columns]
        if missing_in_cache:
            raise RuntimeError(
                "Yahoo rate-limited e cache incompleta: mancano i ticker "
                + ", ".join(missing_in_cache)
                + "."
            )
        cached = cached[expected_tickers]
        cached = cached.dropna(how="all")
        if not cached.empty:
            print(f"Yahoo rate-limited: uso cache locale {cache_file.name}")
            return cached

    raise RuntimeError("Impossibile ottenere dati da Yahoo e nessuna cache disponibile.")
#scarico dataframe
df1 = download_close_with_cache([waahid, ithnaan], inizio_periodo, fine_periodo)
print(f"Il dataframe con {waahid} e {ithnaan} è:\n\n",df1.head())

# Conteggi NaN: righe con almeno un NaN e celle NaN totali
row_has_nan = df1.isna().any(axis=1)
rows_with_nan = int(row_has_nan.sum())
cells_with_nan = int(df1.isna().sum().sum())
print(f"\nNumero di righe con almeno un NaN: {rows_with_nan}")
print(f"Numero totale di celle NaN: {cells_with_nan}\n")

# Pulizia: rimuove le righe con almeno un NaN
dfgood = df1.loc[~row_has_nan].copy()
if dfgood.empty:
    raise RuntimeError("Nessun dato disponibile dopo la rimozione dei NaN. Controlla ticker e periodo.")
print(dfgood)

#calcolo rendimenti logaritmici
returns=np.log(dfgood/dfgood.shift(1)).dropna()
if returns.empty:
    raise RuntimeError("Rendimenti vuoti dopo il calcolo logaritmico. Verifica ticker e finestra temporale.")
print("\n=== Rendimenti logaritmici ===\n")
print(returns)

returns_graphs() #grafico rendimenti
norm_test() #test jarque bera per normalità
qqplot_graphs() #grafico qq plot per vedere se si discosta dalla normale

## Calcolo dei percentili 5° e 95°
p5_waahid = np.percentile(returns[waahid], 5)
p95_waahid = np.percentile(returns[waahid], 95)
p5_ithnaan = np.percentile(returns[ithnaan], 5)
p95_ithnaan = np.percentile(returns[ithnaan], 95)

print(f"\nIl quinto percentile di {waahid} è: {p5_waahid}")
print(f"Il novantacinquesimo percentile di {waahid} è: {p95_waahid}")
print(f"Il quinto percentile di {ithnaan} è: {p5_ithnaan}")
print(f"Il novantacinquesimo percentile di {ithnaan} è: {p95_ithnaan}")

# Valori sotto il 5° percentile ordinati
under_p5_waahid_vals = returns[returns[waahid] < p5_waahid][waahid].sort_values().reset_index(drop=True)
#sort_values() permette di ordinare in modo crescente i rentimenti
#reset_index permette di eliminare la colonna indice date. in modo che non venga rispettato l'ordine delle date
under_p5_ithnaan_vals = returns[returns[ithnaan] < p5_ithnaan][ithnaan].sort_values().reset_index(drop=True)
# Allineamento e creazione DataFrame
df_p5 = pd.concat([under_p5_waahid_vals, under_p5_ithnaan_vals], axis=1)
df_p5.columns = [waahid, ithnaan]
# Valori sopra il 95° percentile ordinati
over_p95_waahid_vals = returns[returns[waahid] > p95_waahid][waahid].sort_values().reset_index(drop=True)
over_p95_ithnaan_vals = returns[returns[ithnaan] > p95_ithnaan][ithnaan].sort_values().reset_index(drop=True)
# Allineamento e creazione DataFrame
df_p95 = pd.concat([over_p95_waahid_vals, over_p95_ithnaan_vals], axis=1)
df_p95.columns = [waahid, ithnaan]

# Stampa dei risultati
print("\nValori sotto il 5° percentile")
print(df_p5)

print("\nValori sopra il 95° percentile")
print(df_p95)

distr_graphs() #grafico distribuzioni marginali

# Normalizzazione dei rendimenti in [0,1] usando i ranks (empirical CDF) #marginali
u = (rankdata(returns[waahid])
     / (len(returns) + 1))
print("\nValori di u", u)

v = (rankdata(returns[ithnaan])
     / (len(returns) + 1))
print("\nValori di v", v)

corr_measures() #richiama misure di dipendenza

# Preparazione dei dati, congiunta in un array 2D #array bidim
data_uv = np.column_stack((u, v))
print("\nValori di u e v combinati\n", data_uv)

# Vera copula empirica C_n(u,v) su griglia
grid_emp, c_n_emp = build_empirical_copula_grid(u, v, grid_size=60)
if SHOW_EMPIRICAL_SCATTER:
    plot_empirical_copula(u, v, title="Pseudo-osservazioni (u,v) nel quadrato unitario - Oro/Argento")

levels = [0.01, 0.05, 0.10, 0.15, 0.85, 0.90, 0.95, 0.99]
print("\n=== Probabilita congiunte empiriche su quantili=== \n(es. Copula empirica C_n(0.01, 0.01)=P(U<=0.01, V<=0.01) è la probabilita che \nl'oro sia <= 1% e l'argento sia <= 1% e cosi via. Mentre per P(U>0.99, V>0.99) è la probabilità \nche l'oro sia > 99% e l'argento sia > 99% e si calcola come 1 - P(U<=0.99) - P(V<=0.99) + C_n(0.99,0.99))")
for q in levels:
    q_pct = int(round(q * 100))
    p_u_le_q = np.mean(u <= q)
    p_v_le_q = np.mean(v <= q)
    c_q = np.mean((u <= q) & (v <= q))
    # Formula complementare: P(U>q, V>q) = 1 - P(U<=q) - P(V<=q) + C_n(q,q)
    p_joint_upper_formula = 1 - p_u_le_q - p_v_le_q + c_q
    p_joint_upper_direct = np.mean((u > q) & (v > q))

    print(f"Copula empirica C_n({q:.2f}, {q:.2f})=P(U<={q:.2f}, V<={q:.2f}): {c_q:.4f}")
    print(f"P(U>{q:.2f}, V>{q:.2f}): {p_joint_upper_direct:.4f}")

print("\n=== Confronto code opposte (sinistra vs destra) ===")
for q_left in [0.01, 0.05, 0.10, 0.15]:
    q_right = 1 - q_left
    c_left = np.mean((u <= q_left) & (v <= q_left))
    c_right = np.mean((u <= q_right) & (v <= q_right))
    p_u_le_right = np.mean(u <= q_right)
    p_v_le_right = np.mean(v <= q_right)
    p_joint_right = 1 - p_u_le_right - p_v_le_right + c_right

    if c_left > p_joint_right:
        verdict = "SINISTRA > DESTRA"
    elif c_left < p_joint_right:
        verdict = "SINISTRA < DESTRA"
    else:
        verdict = "SINISTRA = DESTRA"

    print(
        f"C_n({q_left:.2f},{q_left:.2f}) = {c_left:.4f}  vs  "
        f"P(U>{q_right:.2f},V>{q_right:.2f}) = {p_joint_right:.4f}  ->  {verdict}"
    )

# Stima delle copule
copula_clayton = Clayton()
copula_clayton.fit(data_uv)
theta = copula_clayton.theta #parametro di dipendenza della copula
lambda_L = 2 ** (-1 / theta)  # dipendenza in coda inferiore 2^(-1/theta)  77% di dipendenza condizionata nei minimi
lambda_U = 0                  # Clayton non ha upper tail dependence
print(f"\nClayton Copula:\nTheta: {theta:.4f}\nLower Tail Dependence: {lambda_L:.4f}\nUpper Tail Dependence: {lambda_U:.4f}")
#Tail dependence usate per catturare la dipendenza delle code congiunte nella distribuzione

copula_frank = Frank()
copula_frank.fit(data_uv)
theta = copula_frank.theta
lambda_L = 0
lambda_U = 0
print(f"\nFrank Copula:\nTheta: {theta:.4f}\nLower Tail Dependence: {lambda_L:.4f}\nUpper Tail Dependence: {lambda_U:.4f}")


copula_gumbel = Gumbel()
copula_gumbel.fit(data_uv)
theta = copula_gumbel.theta
lambda_U = 2 - 2 ** (1 / theta)  # dipendenza in coda superiore
lambda_L = 0                     # Gumbel non ha lower tail dependence
print(f"\nGumbel Copula:\nTheta: {theta:.4f}\nLower Tail Dependence: {lambda_L:.4f}\nUpper Tail Dependence: {lambda_U:.4f}")

# Gaussian copula (ellittica, nessuna tail dependence asintotica)
rho_gauss, ll_gauss_mle = fit_gaussian_copula_mle(u, v)
lambda_L = 0
lambda_U = 0
print(f"\nGaussian Copula:\nRho: {rho_gauss:.4f}\nLower Tail Dependence: {lambda_L:.4f}\nUpper Tail Dependence: {lambda_U:.4f}")

# Student-t copula (tail dependence simmetrica)
rho_t, nu_t, ll_t_mle = fit_student_t_copula_mle(u, v)
lambda_t = 2 * t.cdf(-np.sqrt(((nu_t + 1) * (1 - rho_t)) / (1 + rho_t)), df=nu_t + 1)
print(f"\nStudent-t Copula:\nRho: {rho_t:.4f}\nNu: {nu_t:.4f}\nLower Tail Dependence: {lambda_t:.4f}\nUpper Tail Dependence: {lambda_t:.4f}")

# Mixture copula statica: combina copule stimate con pesi non negativi che sommano a 1.
density_clayton = clayton_density(u, v, copula_clayton.theta)
density_frank = frank_density(u, v, copula_frank.theta)
density_gumbel = gumbel_density(u, v, copula_gumbel.theta)
density_gaussian = gaussian_copula_density(u, v, rho_gauss)
density_student_t = student_t_copula_density(u, v, rho_t, nu_t)

mixture_component_names = ["Clayton", "Frank", "Gumbel", "Gaussian", "Student-t"]
mixture_density_matrix = np.column_stack([
    density_clayton,
    density_frank,
    density_gumbel,
    density_gaussian,
    density_student_t,
])
mixture_weights, ll_mixture = fit_mixture_weights_from_densities(mixture_density_matrix)

print("\nMixture Copula Statica (Clayton + Frank + Gumbel + Gaussian + Student-t, pesi MLE):")
for name, weight in zip(mixture_component_names, mixture_weights):
    print(f"w_{name}: {weight:.4f}")
print(f"Somma pesi (controllo): {np.sum(mixture_weights):.6f}")
dominant_idx = int(np.argmax(mixture_weights))

# Verifica dipendenza nella zona centrale: [30%, 70%] x [30%, 70%]
center_low, center_high = 0.30, 0.70
p_emp_center = np.mean(
    (u > center_low) & (u <= center_high) & (v > center_low) & (v <= center_high)
)

def rectangle_prob_from_cdf(cdf_func, low, high, **params):
    low_arr = np.array([low], dtype=float)
    high_arr = np.array([high], dtype=float)
    c_hh = float(cdf_func(high_arr, high_arr, **params)[0])
    c_lh = float(cdf_func(low_arr, high_arr, **params)[0])
    c_hl = float(cdf_func(high_arr, low_arr, **params)[0])
    c_ll = float(cdf_func(low_arr, low_arr, **params)[0])
    p_rect = c_hh - c_lh - c_hl + c_ll
    return float(np.clip(p_rect, 0.0, 1.0))

p_center_clayton = rectangle_prob_from_cdf(
    clayton_cdf, center_low, center_high, theta=copula_clayton.theta
)
p_center_frank = rectangle_prob_from_cdf(
    frank_cdf, center_low, center_high, theta=copula_frank.theta
)
p_center_gumbel = rectangle_prob_from_cdf(
    gumbel_cdf, center_low, center_high, theta=copula_gumbel.theta
)
p_center_gaussian = rectangle_prob_from_cdf(
    gaussian_cdf, center_low, center_high, rho=rho_gauss
)
p_center_student_t = rectangle_prob_from_cdf(
    student_t_cdf, center_low, center_high, rho=rho_t, nu=nu_t
)
p_center_mixture = (
    mixture_weights[0] * p_center_clayton
    + mixture_weights[1] * p_center_frank
    + mixture_weights[2] * p_center_gumbel
    + mixture_weights[3] * p_center_gaussian
    + mixture_weights[4] * p_center_student_t
)

print("\n=== Dipendenza centrale (30%-70%) ===")
print(f"Empirica P(0.30<U<=0.70, 0.30<V<=0.70): {p_emp_center:.4f}")
print(
    f"Clayton   model: {p_center_clayton:.4f}, "
    f"|errore|: {abs(p_emp_center - p_center_clayton):.4f}"
)
print(
    f"Frank     model: {p_center_frank:.4f}, "
    f"|errore|: {abs(p_emp_center - p_center_frank):.4f}"
)
print(
    f"Gumbel    model: {p_center_gumbel:.4f}, "
    f"|errore|: {abs(p_emp_center - p_center_gumbel):.4f}"
)
print(
    f"Gaussian  model: {p_center_gaussian:.4f}, "
    f"|errore|: {abs(p_emp_center - p_center_gaussian):.4f}"
)
print(
    f"Student-t model: {p_center_student_t:.4f}, "
    f"|errore|: {abs(p_emp_center - p_center_student_t):.4f}"
)
print(
    f"Mixture   model: {p_center_mixture:.4f}, "
    f"|errore|: {abs(p_emp_center - p_center_mixture):.4f}"
)

# Confronto diretto C_n(u,v) vs C_theta(u,v) su griglia (Cramer-von Mises discreto + distanza sup)
c_grid_clayton = evaluate_copula_cdf_on_grid(grid_emp, "clayton", theta=copula_clayton.theta)
c_grid_frank = evaluate_copula_cdf_on_grid(grid_emp, "frank", theta=copula_frank.theta)
c_grid_gumbel = evaluate_copula_cdf_on_grid(grid_emp, "gumbel", theta=copula_gumbel.theta)
c_grid_gaussian = evaluate_copula_cdf_on_grid(grid_emp, "gaussian", rho=rho_gauss)
c_grid_student_t = evaluate_copula_cdf_on_grid(grid_emp, "student-t", rho=rho_t, nu=nu_t)
c_grid_mixture = mixture_cdf_on_grid(
    grid_emp,
    [
        ("clayton", {"theta": copula_clayton.theta}),
        ("frank", {"theta": copula_frank.theta}),
        ("gumbel", {"theta": copula_gumbel.theta}),
        ("gaussian", {"rho": rho_gauss}),
        ("student-t", {"rho": rho_t, "nu": nu_t}),
    ],
    mixture_weights,
)

mse_clayton, dmax_clayton = copula_grid_distance_metrics(c_n_emp, c_grid_clayton)
mse_frank, dmax_frank = copula_grid_distance_metrics(c_n_emp, c_grid_frank)
mse_gumbel, dmax_gumbel = copula_grid_distance_metrics(c_n_emp, c_grid_gumbel)
mse_gaussian, dmax_gaussian = copula_grid_distance_metrics(c_n_emp, c_grid_gaussian)
mse_student_t, dmax_student_t = copula_grid_distance_metrics(c_n_emp, c_grid_student_t)
mse_mixture, dmax_mixture = copula_grid_distance_metrics(c_n_emp, c_grid_mixture)

print("\n=== Distanza C_n vs C_theta su griglia ===")
print(f"Clayton - MSE: {mse_clayton:.6f}, Max|Delta|: {dmax_clayton:.6f}")
print(f"Frank   - MSE: {mse_frank:.6f}, Max|Delta|: {dmax_frank:.6f}")
print(f"Gumbel  - MSE: {mse_gumbel:.6f}, Max|Delta|: {dmax_gumbel:.6f}")
print(f"Gaussian- MSE: {mse_gaussian:.6f}, Max|Delta|: {dmax_gaussian:.6f}")
print(f"Student-t- MSE: {mse_student_t:.6f}, Max|Delta|: {dmax_student_t:.6f}")
print(f"Mixture - MSE: {mse_mixture:.6f}, Max|Delta|: {dmax_mixture:.6f}")

# numero osservazioni in data_uv
n_sim = len(data_uv)

sim_clayton = copula_clayton.sample(n_sim)
sim_frank = copula_frank.sample(n_sim)
sim_gumbel = copula_gumbel.sample(n_sim)
sim_gaussian = simulate_gaussian_copula(n_sim, rho_gauss)
sim_student_t = simulate_student_t_copula(n_sim, rho_t, nu_t)

# Chiamata alla funzione di confronto
if SHOW_SIMULATION_COMPARISON:
    plot_copula_comparison(
        data_uv,
        [sim_clayton, sim_frank, sim_gumbel, sim_gaussian, sim_student_t],
        [
            "Clayton (per coda inf)",
            "Frank (per dip centr)",
            "Gumbel (per coda sup)",
            "Gaussian (ellittica)",
            "Student-t (code simm)"
        ],
        "Confronto tra Copule Stimate e Dati Empirici"
    )

# Funzione per calcolare AIC e BIC (classico confronto MLE)
def calculate_aic_bic(log_likelihood, num_params, n_obs):
    if not np.isfinite(log_likelihood):
        return np.nan, np.nan
    aic = 2 * num_params - 2 * log_likelihood
    bic = np.log(n_obs) * num_params - 2 * log_likelihood
    return aic, bic

# Calcolo log-likelihood MLE usando le densita' copula nei punti osservati (u,v)
num_params = 1
n_obs = len(u)

ll_clayton = copula_log_likelihood(density_clayton)
ll_frank = copula_log_likelihood(density_frank)
ll_gumbel = copula_log_likelihood(density_gumbel)
ll_gaussian = copula_log_likelihood(density_gaussian)
ll_student_t = copula_log_likelihood(density_student_t)

# Calcolo AIC e BIC per ciascuna copula
aic_clayton, bic_clayton = calculate_aic_bic(ll_clayton, num_params, n_obs)
aic_frank, bic_frank = calculate_aic_bic(ll_frank, num_params, n_obs)
aic_gumbel, bic_gumbel = calculate_aic_bic(ll_gumbel, num_params, n_obs)
aic_gaussian, bic_gaussian = calculate_aic_bic(ll_gaussian, 1, n_obs)
aic_student_t, bic_student_t = calculate_aic_bic(ll_student_t, 2, n_obs)
# Condizionale ai parametri delle componenti gia' stimati: k = (#pesi - 1).
aic_mixture, bic_mixture = calculate_aic_bic(ll_mixture, len(mixture_weights) - 1, n_obs)

# Stampa dei risultati
print("\n=== AIC, BIC e Log-Likelihood (MLE) per ogni copula ===")
print(f"Clayton - AIC: {aic_clayton:.2f}, BIC: {bic_clayton:.2f}, Log-Likelihood MLE: {ll_clayton:.2f}")
print(f"Frank   - AIC: {aic_frank:.2f}, BIC: {bic_frank:.2f}, Log-Likelihood MLE: {ll_frank:.2f}")
print(f"Gumbel  - AIC: {aic_gumbel:.2f}, BIC: {bic_gumbel:.2f}, Log-Likelihood MLE: {ll_gumbel:.2f}")
print(f"Gaussian- AIC: {aic_gaussian:.2f}, BIC: {bic_gaussian:.2f}, Log-Likelihood MLE: {ll_gaussian:.2f}")
print(f"Student-t- AIC: {aic_student_t:.2f}, BIC: {bic_student_t:.2f}, Log-Likelihood MLE: {ll_student_t:.2f}")
print(f"Mixture - AIC: {aic_mixture:.2f}, BIC: {bic_mixture:.2f}, Log-Likelihood MLE: {ll_mixture:.2f}")

print(f"\nOutput completo salvato in: {LOG_FILE_PATH}")
sys.stdout = _original_stdout
sys.stderr = _original_stderr
_log_file_handle.close()

#creare una copula per osservare la dipendenza nelle code superiori tra oro, argento e vix
