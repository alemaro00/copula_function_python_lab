import pandas as pd
import numpy as np
import copulas as cp
import sympy as sp
import statsmodels.api as sm
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sc
import yfinance as yf
import time
from pathlib import Path
from scipy.stats import jarque_bera, rankdata, kendalltau, gaussian_kde, norm, t
from numpy.random import multivariate_normal
from copulas.bivariate import Clayton, Frank, Gumbel

#cambiare qualsiasi commodities, azione, strumento finanziario preso da yahoo finance
waahid= "GC=F"#"BTC-USD"
ithnaan=  "SI=F" #"ETH-USD"
inizio_periodo="2015-01-01"
fine_periodo="2025-01-01"

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
    print(f"Lower Tail Dependence: {empirical_lambda_L:.4f}")
    print(f"Upper Tail Dependence: {empirical_lambda_U:.4f}")
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
    density_clipped = np.clip(density_values, eps, None)
    return np.sum(np.log(density_clipped))


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
        sp.log(sp.gamma((nu + 2) / 2))
        - sp.log(sp.gamma(nu / 2))
        - sp.log(nu * np.pi)
        - 0.5 * np.log(det_r)
    )
    log_kernel_biv = -((nu + 2) / 2) * np.log1p(inv_quad / nu)
    log_biv = float(log_const_biv) + log_kernel_biv

    log_uni_x = np.log(t.pdf(x, df=nu))
    log_uni_y = np.log(t.pdf(y, df=nu))

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
    rho_hat = float(result.x[0])
    ll = -float(result.fun)
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
    rho_hat = float(result.x[0])
    nu_hat = float(result.x[1])
    ll = -float(result.fun)
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
                data.to_csv(cache_file)
                print(f"Dati scaricati da Yahoo e salvati in cache: {cache_file.name}")
                return data
        except Exception as exc:
            print(f"Tentativo {attempt}/{max_retries} fallito: {exc}")

        if attempt < max_retries:
            time.sleep(base_wait * attempt)

    if cache_file.exists():
        cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        cached = cached[[col for col in tickers if col in cached.columns]]
        cached = cached.dropna(how="all")
        if not cached.empty:
            print(f"Yahoo rate-limited: uso cache locale {cache_file.name}")
            return cached

    raise RuntimeError("Impossibile ottenere dati da Yahoo e nessuna cache disponibile.")
#scarico dataframe
df1 = download_close_with_cache([waahid, ithnaan], inizio_periodo, fine_periodo)
print(f"Il dataframe con {waahid} e {ithnaan} è:\n\n",df1.head())

# Aggiunta della colonna "has_nan"
df1['has_nan'] = df1.iloc[:,:].isna().any(axis=1).astype(int) #colonna nuova chiamata has_nan per mettere valore 1 se c'è almeno un nan nella riga del dataframe
# oppure 0 se non ci sono NaN
count_has_nan=df1.isna().sum().sum() #conta quanti NaN ci sono nel dataframe
print(f"\nNumero di righe con almeno un NaN: {count_has_nan}\n")

data_clean=df1[df1['has_nan'] != 1].copy() #crea nuovo dataframe senza le righe dove la colonna has_nan ha valore 0 cioè tutte
data_clean.drop('has_nan', axis=1, inplace=True) #droppa la colonna has_nan che ha solo 0
dfgood= data_clean #NUOVO dataframe
print(dfgood)

#calcolo rendimenti logaritmici
returns=np.log(dfgood/dfgood.shift(1)).dropna()
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
plot_empirical_copula(u, v, title="Copula empirica C_n(u,v) - Oro/Argento")
print(f"\nCopula empirica C_n(0.05, 0.05): {np.mean((u <= 0.05) & (v <= 0.05)):.4f}")
print(f"Copula empirica C_n(0.95, 0.95): {np.mean((u <= 0.95) & (v <= 0.95)):.4f}")

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

# numero osservazioni in data_uv
n_sim = len(data_uv)

sim_clayton = copula_clayton.sample(n_sim)
sim_frank = copula_frank.sample(n_sim)
sim_gumbel = copula_gumbel.sample(n_sim)
sim_gaussian = simulate_gaussian_copula(n_sim, rho_gauss)
sim_student_t = simulate_student_t_copula(n_sim, rho_t, nu_t)

# Chiamata alla funzione di confronto
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
    aic = 2 * num_params - 2 * log_likelihood
    bic = np.log(n_obs) * num_params - 2 * log_likelihood
    return aic, bic

# Calcolo log-likelihood MLE usando le densita' copula nei punti osservati (u,v)
num_params = 1
n_obs = len(u)

density_clayton = clayton_density(u, v, copula_clayton.theta)
density_frank = frank_density(u, v, copula_frank.theta)
density_gumbel = gumbel_density(u, v, copula_gumbel.theta)
density_gaussian = gaussian_copula_density(u, v, rho_gauss)
density_student_t = student_t_copula_density(u, v, rho_t, nu_t)

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

# Stampa dei risultati
print("\n=== AIC, BIC e Log-Likelihood (MLE) per ogni copula ===")
print(f"Clayton - AIC: {aic_clayton:.2f}, BIC: {bic_clayton:.2f}, Log-Likelihood MLE: {ll_clayton:.2f}")
print(f"Frank   - AIC: {aic_frank:.2f}, BIC: {bic_frank:.2f}, Log-Likelihood MLE: {ll_frank:.2f}")
print(f"Gumbel  - AIC: {aic_gumbel:.2f}, BIC: {bic_gumbel:.2f}, Log-Likelihood MLE: {ll_gumbel:.2f}")
print(f"Gaussian- AIC: {aic_gaussian:.2f}, BIC: {bic_gaussian:.2f}, Log-Likelihood MLE: {ll_gaussian:.2f}")
print(f"Student-t- AIC: {aic_student_t:.2f}, BIC: {bic_student_t:.2f}, Log-Likelihood MLE: {ll_student_t:.2f}")

#creare una copula per osservare la dipendenza nelle code superiori tra oro, argento e vix
