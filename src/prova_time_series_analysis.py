import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf

# Cambiare qualsiasi commodity, azione o strumento finanziario preso da Yahoo Finance.
waahid = "GC=F"  # "BTC-USD"
ithnaan = "SI=F"  # "ETH-USD"
inizio_periodo = "2017-01-01"
fine_periodo = "2026-04-10"


def _pick_price_series(raw_data: pd.DataFrame, ticker: str) -> pd.Series:
	"""Seleziona una serie prezzi robusta da un DataFrame yfinance."""
	if raw_data.empty:
		raise ValueError(f"Nessun dato disponibile per {ticker}.")

	if isinstance(raw_data.columns, pd.MultiIndex):
		# In alcuni casi yfinance restituisce colonne MultiIndex.
		if "Adj Close" in raw_data.columns.get_level_values(0):
			series = raw_data["Adj Close"].squeeze()
		elif "Close" in raw_data.columns.get_level_values(0):
			series = raw_data["Close"].squeeze()
		else:
			raise ValueError(f"Colonna prezzo non trovata per {ticker}.")
	else:
		if "Adj Close" in raw_data.columns:
			series = raw_data["Adj Close"]
		elif "Close" in raw_data.columns:
			series = raw_data["Close"]
		else:
			raise ValueError(f"Colonna prezzo non trovata per {ticker}.")

	series = pd.to_numeric(series, errors="coerce").dropna()
	if series.empty:
		raise ValueError(f"La serie prezzi di {ticker} e' vuota dopo la pulizia.")
	return series


def download_price_series(
	ticker: str,
	start: str,
	end: str,
	max_retries: int = 3,
	base_wait_seconds: float = 1.0,
) -> pd.Series:
	"""Scarica i prezzi con retry/backoff per gestire limiti temporanei Yahoo."""
	last_error = None
	for attempt in range(1, max_retries + 1):
		try:
			data = yf.download(
				ticker,
				start=start,
				end=end,
				progress=False,
				auto_adjust=False,
				threads=False,
			)
			return _pick_price_series(data, ticker)
		except Exception as exc:  # noqa: BLE001
			last_error = exc
			if attempt < max_retries:
				time.sleep(base_wait_seconds * (2 ** (attempt - 1)))

	raise RuntimeError(
		f"Download fallito per {ticker} dopo {max_retries} tentativi: {last_error}"
	)


def sample_moments(returns: pd.Series) -> dict:
	"""Calcola i 4 momenti campionari (media, varianza, skewness, kurtosis)."""
	x = returns.dropna().to_numpy(dtype=float)
	n = x.size
	if n < 2:
		raise ValueError("Numero osservazioni insufficiente per i momenti campionari.")

	mean = x.mean()
	centered = x - mean
	variance = np.sum(centered**2) / (n - 1)
	std = np.sqrt(variance)

	if std == 0:
		skewness = np.nan
		kurtosis = np.nan
	else:
		skewness = np.sum(centered**3) / ((n - 1) * (std**3))
		kurtosis = np.sum(centered**4) / ((n - 1) * (std**4))

	return {
		"n": n,
		"mean": mean,
		"variance": variance,
		"std": std,
		"skewness": skewness,
		"kurtosis": kurtosis,
		"excess_kurtosis": kurtosis - 3 if not np.isnan(kurtosis) else np.nan,
	}


def normality_tests(returns: pd.Series, moments: dict) -> dict:
	"""Replica test su skewness, kurtosis e Jarque-Bera mostrati nelle pagine."""
	n = moments["n"]
	s = moments["skewness"]
	k = moments["kurtosis"]

	if np.isnan(s) or np.isnan(k):
		return {
			"t_skew": np.nan,
			"p_skew": np.nan,
			"t_kurt": np.nan,
			"p_kurt": np.nan,
			"jb": np.nan,
			"jb_p": np.nan,
		}

	t_skew = s / np.sqrt(6.0 / n)
	p_skew = 2 * (1 - stats.norm.cdf(abs(t_skew)))

	t_kurt = (k - 3.0) / np.sqrt(24.0 / n)
	p_kurt = 2 * (1 - stats.norm.cdf(abs(t_kurt)))

	jb = (n * (s**2) / 6.0) + (n * ((k - 3.0) ** 2) / 24.0)
	jb_p = 1 - stats.chi2.cdf(jb, df=2)

	ttest_res = stats.ttest_1samp(returns.dropna().to_numpy(dtype=float), popmean=0.0)

	return {
		"t_skew": t_skew,
		"p_skew": p_skew,
		"t_kurt": t_kurt,
		"p_kurt": p_kurt,
		"jb": jb,
		"jb_p": jb_p,
		"t_mean": ttest_res.statistic,
		"p_mean": ttest_res.pvalue,
	}


def print_report(name: str, ticker: str, returns: pd.Series) -> None:
	moments = sample_moments(returns)
	tests = normality_tests(returns, moments)

	quantile_levels = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
	quantiles = returns.quantile(quantile_levels)

	print(f"\n{'=' * 72}")
	print(f"Serie: {name} ({ticker}) | Periodo: {inizio_periodo} -> {fine_periodo}")
	print(f"Osservazioni utili: {moments['n']}")
	print("Rendimenti semplici percentuali: 100 * (P_t / P_(t-1) - 1)")

	print("\n4 momenti della distribuzione campionaria")
	print(f"Media        : {moments['mean']:.8f}")
	print(f"Varianza     : {moments['variance']:.8f}")
	print(f"Skewness     : {moments['skewness']:.8f}")
	print(f"Kurtosis     : {moments['kurtosis']:.8f}")
	print(f"Excess Kurt. : {moments['excess_kurtosis']:.8f}")

	print("\nQuantili dei rendimenti (%)")
	for q, val in quantiles.items():
		print(f"q={q:>4.0%} : {val:.8f}")

	print("\nTest riportati nelle pagine")
	print(f"Test media=0 (t)         : t={tests['t_mean']:.6f}, p-value={tests['p_mean']:.6g}")
	print(f"Test skewness=0          : t={tests['t_skew']:.6f}, p-value={tests['p_skew']:.6g}")
	print(f"Test excess kurtosis=0   : t={tests['t_kurt']:.6f}, p-value={tests['p_kurt']:.6g}")
	print(f"Jarque-Bera normalita'   : JB={tests['jb']:.6f}, p-value={tests['jb_p']:.6g}")


def main() -> None:
	tickers = {"waahid": waahid, "ithnaan": ithnaan}

	for name, ticker in tickers.items():
		prices = download_price_series(ticker, inizio_periodo, fine_periodo)
		simple_returns = prices.pct_change().dropna() * 100

		if simple_returns.empty:
			print(
				f"Serie {name} ({ticker}) senza rendimenti validi nel periodo {inizio_periodo} - {fine_periodo}."
			)
			continue

		print_report(name, ticker, simple_returns)


if __name__ == "__main__":
	main()
