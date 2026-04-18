import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sc
import yfinance as yf
from datetime import datetime

#cambiare qualsiasi commodities, azione, strumento finanziario preso da yahoo finance
waahid= "BTC-USD"#"GC=F"#
ithnaan=  "ETH-USD"#"SI=F"#
inizio_periodo="2017-01-01"
fine_periodo="2026-04-10"
