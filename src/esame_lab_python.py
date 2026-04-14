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
from scipy.stats import t
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
    fig, axes = plt.subplots(1, len(sim_data_list) + 1, figsize=(18, 5))

    axes[0].scatter(real_data[:, 0], real_data[:, 1], alpha=0.5)
    axes[0].set_title("Dati reali (u,v)")

    for i, (sim_data, label) in enumerate(zip(sim_data_list, labels)):
        axes[i + 1].scatter(sim_data[:, 0], sim_data[:, 1], alpha=0.5, color='orange')
        axes[i + 1].set_title(f"Copula {label}")

    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


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
lambda_L = 1 / theta  # dipendenza in coda inferiore per copula Frank
lambda_U = 0          # nessuna dipendenza in coda superiore
print(f"\nClayton Copula:\nTheta: {theta:.4f}\nLower Tail Dependence: {lambda_L:.4f}\nUpper Tail Dependence: {lambda_U:.4f}")


copula_gumbel = Gumbel()
copula_gumbel.fit(data_uv)
theta = copula_gumbel.theta
lambda_U = 2 - 2 ** (1 / theta)  # dipendenza in coda superiore
lambda_L = 0                     # Gumbel non ha lower tail dependence
print(f"\nGumbel Copula:\nTheta: {theta:.4f}\nLower Tail Dependence: {lambda_L:.4f}\nUpper Tail Dependence: {lambda_U:.4f}")

# numero osservazioni in data_uv
n_sim = len(data_uv)

sim_clayton = copula_clayton.sample(n_sim)
sim_frank = copula_frank.sample(n_sim)
sim_gumbel = copula_gumbel.sample(n_sim)

# Chiamata alla funzione di confronto
plot_copula_comparison(
    data_uv,
    [sim_clayton, sim_frank, sim_gumbel],
    ["Clayton (per coda inf)", "Frank (per dip centr)", "Gumbel (per coda sup)"],
    "Confronto tra Copule Stimate e Dati Empirici"
)

# Numero di parametri del modello Clayton, Frank e Gumbel (parametro unico stimato è theta)
num_params = 1  # Modifica in base al numero di parametri per ciascuna copula

# Funzione di log-likelihood approssimata per i dati simulati da una copula
def empirical_log_likelihood(uv_real, uv_sim):
    kde = gaussian_kde(uv_sim.T)  # Stima la densità dei dati simulati
    log_lik = np.log(kde(uv_real.T))  # Valuta la densità nei punti reali
    return np.sum(log_lik)

# Funzione per calcolare AIC e BIC
def calculate_aic_bic(log_likelihood, num_params, n_sim):
    aic = 2 * num_params - 2 * log_likelihood
    bic = np.log(n_sim) * num_params - 2 * log_likelihood
    return aic, bic

# Calcolo della log-likelihood empirica per ogni copula
ll_clayton = empirical_log_likelihood(data_uv, sim_clayton)
ll_frank = empirical_log_likelihood(data_uv, sim_frank)
ll_gumbel = empirical_log_likelihood(data_uv, sim_gumbel)

# Calcolo AIC e BIC per ciascuna copula
aic_clayton, bic_clayton = calculate_aic_bic(ll_clayton, num_params, n_sim)
aic_frank, bic_frank = calculate_aic_bic(ll_frank, num_params, n_sim)
aic_gumbel, bic_gumbel = calculate_aic_bic(ll_gumbel, num_params, n_sim)

# Stampa dei risultati
print("\n=== AIC, BIC e Log-Likelihood Empirica (KDE) per ogni copula ===")
print(f"Clayton - AIC: {aic_clayton:.2f}, BIC: {bic_clayton:.2f}, Log-Likelihood Empirica (KDE):{ll_clayton:.2f}")
print(f"Frank   - AIC: {aic_frank:.2f}, BIC: {bic_frank:.2f}, Log-Likelihood Empirica (KDE):{ll_frank:.2f}")
print(f"Gumbel  - AIC: {aic_gumbel:.2f}, BIC: {bic_gumbel:.2f}, Log-Likelihood Empirica (KDE):  {ll_gumbel:.2f}")

#creare una copula per osservare la dipendenza nelle code superiori tra oro, argento e vix


