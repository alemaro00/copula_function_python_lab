import time
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as stats
import yfinance as yf
from plotly.subplots import make_subplots

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


def _safe_token(value: str) -> str:
	"""Rende una stringa sicura per essere usata nel nome file."""
	return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def plot_report(name: str, ticker: str, prices: pd.Series, returns: pd.Series) -> Path:
	"""Crea un dashboard HTML con distribuzione e grafici diagnostici."""
	out_dir = Path("output") / "plots"
	out_dir.mkdir(parents=True, exist_ok=True)

	fig = make_subplots(
		rows=2,
		cols=2,
		subplot_titles=(
			"Prezzo nel tempo",
			"Distribuzione rendimenti (%)",
			"Boxplot rendimenti (%)",
			"Q-Q plot (normalita')",
		),
	)

	fig.add_trace(
		go.Scatter(x=prices.index, y=prices.values, mode="lines", name="Prezzo"),
		row=1,
		col=1,
	)

	ret_vals = returns.to_numpy(dtype=float)
	fig.add_trace(
		go.Histogram(
			x=ret_vals,
			nbinsx=80,
			histnorm="probability density",
			name="Istogramma",
			opacity=0.65,
		),
		row=1,
		col=2,
	)

	if returns.nunique() > 1:
		kde = stats.gaussian_kde(ret_vals)
		x_grid = np.linspace(ret_vals.min(), ret_vals.max(), 300)
		fig.add_trace(
			go.Scatter(x=x_grid, y=kde(x_grid), mode="lines", name="KDE"),
			row=1,
			col=2,
		)

	fig.add_trace(
		go.Box(y=ret_vals, name="Rendimenti", boxmean=True),
		row=2,
		col=1,
	)

	theo_q, samp_q = stats.probplot(ret_vals, dist="norm", fit=False)
	fig.add_trace(
		go.Scatter(x=theo_q, y=samp_q, mode="markers", name="Q-Q"),
		row=2,
		col=2,
	)

	qq_min = min(np.min(theo_q), np.min(samp_q))
	qq_max = max(np.max(theo_q), np.max(samp_q))
	fig.add_trace(
		go.Scatter(
			x=[qq_min, qq_max],
			y=[qq_min, qq_max],
			mode="lines",
			name="Linea 45deg",
			line=dict(dash="dash"),
		),
		row=2,
		col=2,
	)

	fig.update_xaxes(title_text="Data", row=1, col=1)
	fig.update_yaxes(title_text="Prezzo", row=1, col=1)
	fig.update_xaxes(title_text="Rendimento %", row=1, col=2)
	fig.update_yaxes(title_text="Densita'", row=1, col=2)
	fig.update_yaxes(title_text="Rendimento %", row=2, col=1)
	fig.update_xaxes(title_text="Quantili teorici", row=2, col=2)
	fig.update_yaxes(title_text="Quantili campionari", row=2, col=2)

	fig.update_layout(
		title=f"{name} ({ticker}) - Analisi grafica",
		template="plotly_white",
		height=850,
		width=1300,
		legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
	)

	file_name = f"{_safe_token(name)}_{_safe_token(ticker)}_dashboard.html"
	out_path = out_dir / file_name
	fig.write_html(out_path, include_plotlyjs="cdn")
	return out_path


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
		plot_path = plot_report(name, ticker, prices, simple_returns)
		print(f"Grafico salvato in: {plot_path}")


if __name__ == "__main__":
	main()
