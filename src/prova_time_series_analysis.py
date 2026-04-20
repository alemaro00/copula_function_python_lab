import time
import re
from pathlib import Path
from zoneinfo import ZoneInfo

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
mostra_demo_yfinance_apis = True


def _pick_price_series(raw_data: pd.DataFrame, ticker: str) -> pd.Series:
	"""Seleziona una serie prezzi robusta da un DataFrame yfinance."""
	if raw_data.empty:
		raise ValueError(f"Nessun dato disponibile per {ticker}.")

	series: pd.Series | None = None
	if isinstance(raw_data.columns, pd.MultiIndex):
		first_level = raw_data.columns.get_level_values(0)
		second_level = raw_data.columns.get_level_values(1)

		# Formato comune: livello 0 = campo prezzo, livello 1 = ticker.
		if "Adj Close" in first_level or "Close" in first_level:
			price_col = "Adj Close" if "Adj Close" in first_level else "Close"
			price_slice = raw_data[price_col]
			if isinstance(price_slice, pd.Series):
				series = price_slice
			elif ticker in price_slice.columns:
				series = price_slice[ticker]
			elif price_slice.shape[1] == 1:
				series = price_slice.iloc[:, 0]

		# Fallback: livello 0 = ticker, livello 1 = campo prezzo.
		elif ticker in first_level and ("Adj Close" in second_level or "Close" in second_level):
			ticker_slice = raw_data[ticker]
			if "Adj Close" in ticker_slice.columns:
				series = ticker_slice["Adj Close"]
			elif "Close" in ticker_slice.columns:
				series = ticker_slice["Close"]
	else:
		if "Adj Close" in raw_data.columns:
			series = raw_data["Adj Close"]
		elif "Close" in raw_data.columns:
			series = raw_data["Close"]

	if series is None:
		raise ValueError(f"Colonna prezzo non trovata per {ticker}.")

	series = pd.to_numeric(series, errors="coerce").dropna()
	if series.empty:
		raise ValueError(f"La serie prezzi di {ticker} e' vuota dopo la pulizia.")
	series.name = ticker
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
				group_by="column",
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


def download_price_series_batch(
	tickers: list[str],
	start: str,
	end: str,
	max_retries: int = 3,
	base_wait_seconds: float = 1.0,
) -> dict[str, pd.Series]:
	"""Scarica piu' ticker in un solo round-trip per ridurre latenza e richieste."""
	unique_tickers = list(dict.fromkeys(tickers))
	if not unique_tickers:
		return {}

	last_error = None
	for attempt in range(1, max_retries + 1):
		try:
			data = yf.download(
				unique_tickers,
				start=start,
				end=end,
				progress=False,
				auto_adjust=False,
				group_by="column",
				threads=True,
			)
			return {ticker: _pick_price_series(data, ticker) for ticker in unique_tickers}
		except Exception as exc:  # noqa: BLE001
			last_error = exc
			if attempt < max_retries:
				time.sleep(base_wait_seconds * (2 ** (attempt - 1)))

	raise RuntimeError(
		f"Download batch fallito per {', '.join(unique_tickers)} dopo {max_retries} tentativi: {last_error}"
	)


def _returns_to_numpy(returns: pd.Series | np.ndarray) -> np.ndarray:
	"""Converte i rendimenti in ndarray float senza NaN."""
	if isinstance(returns, np.ndarray):
		x = np.asarray(returns, dtype=float)
		return x[np.isfinite(x)]
	return returns.dropna().to_numpy(dtype=float)


def sample_moments(returns: pd.Series | np.ndarray) -> dict:
	"""Calcola i 4 momenti campionari (media, varianza, skewness, kurtosis)."""
	x = _returns_to_numpy(returns)
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


def normality_tests(returns: pd.Series | np.ndarray, moments: dict) -> dict:
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

	ttest_res = stats.ttest_1samp(_returns_to_numpy(returns), popmean=0.0)

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
	ret_vals = _returns_to_numpy(returns)
	moments = sample_moments(ret_vals)
	tests = normality_tests(ret_vals, moments)

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


def _print_dataframe_preview(title: str, df: pd.DataFrame, max_rows: int = 10) -> None:
	"""Stampa un'anteprima leggibile di un DataFrame."""
	print(f"\n{title}")
	if df.empty:
		print("Nessun risultato.")
		return
	print(f"Righe: {len(df)} | Colonne: {len(df.columns)}")
	print(df.head(max_rows).to_string())


def _lookup_unique_symbol_count(df: pd.DataFrame) -> int:
	"""Conta i simboli unici in un risultato lookup."""
	if df.empty:
		return 0
	if "symbol" in df.columns:
		return df["symbol"].astype(str).nunique(dropna=True)
	return pd.Index(df.index).astype(str).nunique(dropna=True)


def _dedupe_lookup_by_symbol(df: pd.DataFrame) -> pd.DataFrame:
	"""Rimuove duplicati mantenendo la prima occorrenza per simbolo."""
	if df.empty:
		return df
	if "symbol" in df.columns:
		return df.drop_duplicates(subset=["symbol"], keep="first")
	return df[~df.index.duplicated(keep="first")]


def _lookup_fixed_with_retry(
	query: str,
	lookup_type: str,
	count: int,
	max_retries: int = 3,
	base_wait_seconds: float = 1.0,
) -> pd.DataFrame:
	"""Lookup con count fisso e retry/backoff per risposte vuote intermittenti."""
	lookup = yf.Lookup(query)
	getter_map = {
		"all": lookup.get_all,
		"stock": lookup.get_stock,
		"future": lookup.get_future,
	}
	if lookup_type not in getter_map:
		raise ValueError(f"lookup_type non supportato: {lookup_type}")

	getter = getter_map[lookup_type]
	last_df = pd.DataFrame()
	for attempt in range(1, max_retries + 1):
		last_df = getter(count=count)
		if not last_df.empty:
			return last_df
		if attempt < max_retries:
			time.sleep(base_wait_seconds * (2 ** (attempt - 1)))
	return last_df


def lookup_all_by_query(
	query: str,
	lookup_type: str = "all",
	initial_count: int = 100,
	step: int = 100,
	max_count: int = 5000,
	plateau_rounds: int = 2,
	empty_retries: int = 2,
	empty_retry_wait_seconds: float = 1.0,
) -> pd.DataFrame:
	"""Recupera piu' risultati possibili aumentando count finche' la crescita si ferma."""
	lookup = yf.Lookup(query)
	getter_map = {
		"all": lookup.get_all,
		"stock": lookup.get_stock,
		"future": lookup.get_future,
	}
	if lookup_type not in getter_map:
		raise ValueError(f"lookup_type non supportato: {lookup_type}")

	getter = getter_map[lookup_type]
	best_df = pd.DataFrame()
	previous_len = -1
	plateau_hits = 0

	count = max(1, initial_count)
	while count <= max_count:
		df = pd.DataFrame()
		for attempt in range(1, empty_retries + 2):
			df = getter(count=count)
			if not df.empty:
				break
			if attempt < empty_retries + 1:
				time.sleep(empty_retry_wait_seconds * (2 ** (attempt - 1)))

		if df.empty:
			# Se e' un vuoto temporaneo dopo aver gia' raccolto dati, restituisce il miglior risultato disponibile.
			if not best_df.empty:
				break
			count += max(1, step)
			continue

		best_df = df
		current_len = len(df)

		# Se Yahoo restituisce meno del richiesto, abbiamo verosimilmente finito i risultati.
		if current_len < count:
			break

		if current_len == previous_len:
			plateau_hits += 1
			if plateau_hits >= plateau_rounds:
				break
		else:
			plateau_hits = 0

		previous_len = current_len
		count += max(1, step)

	return best_df


def demo_yfinance_lookup(query: str, count: int | None = None) -> None:
	"""Esempio API Lookup: cerca simboli/strumenti a partire da una query testuale."""
	if count is None:
		print("Lookup in modalita' auto-count (senza count fisso, con limite di sicurezza).")
		all_results = lookup_all_by_query(query, lookup_type="all")
		stock_results = lookup_all_by_query(query, lookup_type="stock")
		future_results = lookup_all_by_query(query, lookup_type="future")

		# Fallback robusto: se auto-count non trova nulla, prova con count fisso.
		if all_results.empty:
			all_results = _lookup_fixed_with_retry(query, "all", count=200)
		if stock_results.empty:
			stock_results = _lookup_fixed_with_retry(query, "stock", count=200)
		if future_results.empty:
			future_results = _lookup_fixed_with_retry(query, "future", count=200)
	else:
		all_results = _lookup_fixed_with_retry(query, "all", count=count)
		stock_results = _lookup_fixed_with_retry(query, "stock", count=count)
		future_results = _lookup_fixed_with_retry(query, "future", count=count)

	lookup_results = {
		"ALL": all_results,
		"STOCK": stock_results,
		"FUTURE": future_results,
	}
	for label, df in lookup_results.items():
		unique_symbols = _lookup_unique_symbol_count(df)
		dedup_df = _dedupe_lookup_by_symbol(df)
		duplicates_removed = len(df) - len(dedup_df)
		print(
			f"Lookup {label}: righe={len(df)}, simboli unici={unique_symbols}, "
			f"duplicati rimossi={duplicates_removed}"
		)
		lookup_results[label] = dedup_df

	_print_dataframe_preview(f"Lookup ALL per query='{query}'", lookup_results["ALL"])
	_print_dataframe_preview(f"Lookup STOCK per query='{query}'", lookup_results["STOCK"])
	_print_dataframe_preview(f"Lookup FUTURE per query='{query}'", lookup_results["FUTURE"])


def demo_yfinance_market(market_code: str = "us") -> None:
	"""Esempio API Market: stato mercato e indici principali per una regione."""
	market = yf.Market(market_code)
	status = market.status
	summary = market.summary

	print(f"\nMarket status ({market_code.upper()})")
	status_keys = ["name", "status", "message"]
	for key in status_keys:
		if key in status:
			print(f"{key}: {status[key]}")

	for key in ["open", "close"]:
		if key in status:
			dt_utc = pd.Timestamp(status[key]).tz_convert("UTC")
			dt_rome = dt_utc.tz_convert(ZoneInfo("Europe/Rome"))
			print(
				f"{key}: {dt_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
				f"Italia: {dt_rome.strftime('%Y-%m-%d %H:%M:%S %Z')}"
			)

	if isinstance(summary, dict) and summary:
		summary_df = pd.DataFrame.from_dict(summary, orient="index")
		preferred_cols = ["shortName", "regularMarketPrice", "regularMarketChangePercent"]
		visible_cols = [col for col in preferred_cols if col in summary_df.columns]
		if visible_cols:
			summary_df = summary_df[visible_cols]
		_print_dataframe_preview(f"Market summary ({market_code.upper()})", summary_df)
	else:
		print("Market summary non disponibile.")


def demo_yfinance_calendars(limit: int = 5) -> None:
	"""Esempio API Calendars: earnings e IPO per prossimi eventi."""
	calendars = yf.Calendars()
	earnings = calendars.get_earnings_calendar(limit=limit)
	ipos = calendars.get_ipo_info_calendar(limit=limit)

	_print_dataframe_preview("Calendars - Earnings", earnings)
	_print_dataframe_preview("Calendars - IPO", ipos)


def run_yfinance_api_demo() -> None:
	"""Mostra in console un esempio pratico di Lookup, Market e Calendars."""
	print(f"\n{'=' * 72}")
	print("Demo yfinance APIs: Lookup, Market, Calendars")
	print(f"{'=' * 72}")

	try:
		demo_yfinance_lookup(query="byd", count=None)
	except Exception as exc:  # noqa: BLE001
		print(f"Errore demo Lookup: {exc}")

	try:
		demo_yfinance_market(market_code="us")
	except Exception as exc:  # noqa: BLE001
		print(f"Errore demo Market: {exc}")

	try:
		demo_yfinance_calendars(limit=5)
	except Exception as exc:  # noqa: BLE001
		print(f"Errore demo Calendars: {exc}")


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

	ret_vals = _returns_to_numpy(returns)
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
	prices_by_ticker: dict[str, pd.Series] = {}

	if mostra_demo_yfinance_apis:
		run_yfinance_api_demo()

	try:
		prices_by_ticker = download_price_series_batch(
			list(tickers.values()),
			inizio_periodo,
			fine_periodo,
		)
	except Exception as exc:  # noqa: BLE001
		print(
			"Download batch non riuscito, procedo con fallback per singolo ticker. "
			f"Dettaglio: {exc}"
		)

	for name, ticker in tickers.items():
		prices = prices_by_ticker.get(ticker)
		if prices is None:
			try:
				prices = download_price_series(ticker, inizio_periodo, fine_periodo)
			except Exception as exc:  # noqa: BLE001
				print(f"Serie {name} ({ticker}) non scaricabile: {exc}")
				continue

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
