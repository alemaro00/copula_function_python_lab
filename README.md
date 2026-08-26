# Analisi della dipendenza tra due asset mediante copule

Questo progetto analizza come i **rendimenti logaritmici giornalieri di due asset** tendono a muoversi insieme, con particolare attenzione alle giornate caratterizzate da rendimenti molto bassi o molto alti.

Il programma risponde principalmente a queste domande:

- quanto sono collegati i rendimenti giornalieri dei due asset?
- se un asset rientra tra i propri rendimenti peggiori, quanto spesso accade lo stesso all'altro?
- quanto spesso entrambi presentano contemporaneamente rendimenti estremi?
- quale copula descrive meglio la dipendenza osservata?

I risultati descrivono il campione storico selezionato: non costituiscono una previsione dei rendimenti futuri.

## Configurazione dell'analisi

Aprire [`src/esame_lab_python.py`](src/esame_lab_python.py) e modificare le quattro variabili iniziali:

```python
waahid = "BZ=F"
ithnaan = "CL=F"
inizio_periodo = "2010-01-01"
fine_periodo = "2026-08-24"
```

- `waahid`: ticker del primo asset, rappresentato nell'analisi da `U`;
- `ithnaan`: ticker del secondo asset, rappresentato da `V`;
- `inizio_periodo`: data iniziale del campione;
- `fine_periodo`: data finale richiesta.

I ticker devono essere riconosciuti da Yahoo Finance. Alcuni esempi sono `BTC-USD` e `ETH-USD`, `GC=F` e `SI=F`, `V` e `MA`, oppure `BZ=F` e `CL=F`.

## Installazione e avvio con uv

Il progetto usa Python 3.14 e gestisce ambiente e dipendenze con [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
uv run python src/esame_lab_python.py
```

In alternativa, se l'ambiente virtuale esiste già:

```powershell
.\.venv\Scripts\python.exe .\src\esame_lab_python.py
```

I grafici vengono mostrati in successione. In alcuni casi è necessario **chiudere il grafico corrente** affinché il programma prosegua. Il confronto finale può richiedere più tempo perché valuta anche la copula Student-t su una griglia.

## Cosa fa il programma

1. Scarica da Yahoo Finance i prezzi di chiusura, oppure riutilizza quelli presenti nella cache.
2. Allinea le osservazioni dei due asset e calcola i rendimenti logaritmici giornalieri.
3. Mostra rendimenti, QQ plot, distribuzioni e misure di correlazione.
4. Trasforma i rendimenti in ranghi percentili, chiamati pseudo-osservazioni `U` e `V`.
5. Costruisce la copula empirica e calcola probabilità congiunte e condizionate per diverse soglie.
6. Stima Clayton, Frank, Gumbel, Gaussian, Student-t e una mixture delle cinque copule.
7. Confronta i modelli mediante distanza dalla copula empirica, Log-Likelihood, AIC e BIC.
8. Salva automaticamente l'intero output della console nella cartella `output`.

---

## Guida alla lettura dell'output

### File generati e cache

L’output completo della console viene salvato automaticamente in:

```text
output/ASSET1&ASSET2_DATA-INIZIALE&DATA-FINALE.txt
```

Per esempio:

```text
output/BZ=F&CL=F_2010-01-01&2026-08-24.txt
```

I prezzi scaricati vengono conservati in `src/prezzi_close_cache.csv`. La cache evita di scaricare nuovamente dati già disponibili; se ticker o periodo richiesti non sono coperti, il programma prova a recuperarli da Yahoo Finance e aggiorna il file.

I grafici vengono visualizzati ma non salvati automaticamente.

### Rendimenti e percentili

Il 5° percentile separa approssimativamente il 5% dei rendimenti giornalieri peggiori dal restante 95%. Il 95° percentile individua invece il limite oltre il quale si trova approssimativamente il 5% dei rendimenti migliori.

Le soglie sono calcolate **separatamente per ogni asset**. Dire che entrambi sono nel proprio 5% peggiore non significa che abbiano ottenuto lo stesso rendimento, ma che ciascuno si trova nella fascia peggiore della propria distribuzione storica.

### Probabilità congiunte e condizionate

Per ogni soglia `q`, il programma osserva due fasce:

- coda inferiore: rendimenti compresi tra i peggiori `q%` di ciascun asset;
- coda superiore: rendimenti compresi tra i migliori `q%` di ciascun asset.

#### Probabilità congiunta

Risponde alla domanda:

> In una giornata qualsiasi, qual è la probabilità che entrambi gli asset
> siano contemporaneamente nella fascia indicata?

Esempio:

`P(congiunta)=0.0323`

significa che nel 3,23% di tutte le giornate entrambi gli asset sono stati
contemporaneamente nel proprio 5% peggiore.

#### Probabilità condizionata

Risponde alla domanda:

> Considerando soltanto le giornate in cui il primo asset è nella fascia
> indicata, qual è la probabilità che anche il secondo si trovi nella propria
> fascia corrispondente?

Esempio:

`P(MA bassa | V bassa)=64.59% [n=209]`

significa che, tra le 209 giornate in cui Visa era nel proprio 5% peggiore,
Mastercard si trovava nel proprio 5% peggiore nel 64,59% dei casi.

La probabilità inversa risponde alla stessa domanda scambiando il ruolo dei
due asset.

Con pseudo-osservazioni basate sui ranghi e la stessa soglia per entrambi, i due eventi condizionanti contengono normalmente lo stesso numero di osservazioni; per questo le due probabilità condizionate possono risultare uguali o quasi uguali.

### Pearson, Spearman e Kendall

Questi valori sintetizzano quanto i due rendimenti tendono a muoversi insieme.

- vicino a `1`: forte movimento nella stessa direzione;
- vicino a `0`: legame debole;
- vicino a `-1`: movimento prevalentemente opposto.

Spearman e Kendall sono particolarmente utili nel progetto perché misurano
la concordanza dell’ordine dei rendimenti e sono meno dipendenti dalla forma
della loro distribuzione.

### Dipendenza nelle code

- `Lower Tail Dependence`: capacità del modello di rappresentare giornate
  estremamente negative condivise;
- `Upper Tail Dependence`: capacità di rappresentare giornate estremamente
  positive condivise.

Un valore più alto indica una maggiore tendenza dei due asset a trovarsi
insieme in quella coda. Non rappresenta però direttamente la probabilità
condizionata osservata a una specifica soglia: per quella lettura è preferibile
la sezione delle probabilità empiriche.

### Dipendenza centrale

La sezione `Dipendenza centrale (30%-70%)` confronta quanto spesso entrambi gli asset si trovano contemporaneamente nella parte centrale delle rispettive distribuzioni. Serve a verificare se un modello descrive bene non soltanto gli eventi estremi, ma anche le osservazioni ordinarie.

## Come individuare il modello migliore

Il programma confronta sei modelli:

- Clayton;
- Frank;
- Gumbel;
- Gaussian;
- Student-t;
- Mixture delle cinque copule precedenti.

### MSE e Max|Delta|

Confrontano la copula stimata con quella empirica su una griglia.

- MSE più basso: minore errore medio;
- Max|Delta| più basso: minore errore nel punto peggiore.

### Log-Likelihood

Una Log-Likelihood più alta indica che il modello si adatta meglio ai dati,
ma non penalizza direttamente i modelli più complessi.

### AIC e BIC

AIC e BIC confrontano adattamento e complessità.

- AIC più basso: modello preferibile secondo AIC;
- BIC più basso: modello preferibile secondo BIC;
- BIC penalizza più severamente i modelli complessi.

La riga `Risultato automatico` mostra il vincitore per ciascun criterio.

Se AIC e BIC indicano lo stesso modello, la scelta è più chiara.
Se indicano modelli differenti, non esiste un vincitore assoluto: AIC tende
a favorire l’adattamento, mentre BIC tende a preferire la semplicità.

Per la mixture, AIC e BIC considerano **10 parametri complessivi**: sei parametri delle cinque componenti e quattro pesi liberi. In questo modo il modello più flessibile riceve una penalizzazione coerente con la sua complessità.

## Limiti dell’analisi

- I risultati descrivono il periodo storico selezionato e possono cambiare
  scegliendo altre date.
- Dipendenza non significa causalità: il codice non stabilisce che un asset
  provochi il movimento dell’altro.
- Le probabilità sono frequenze storiche empiriche, non previsioni garantite.
- Un legame forte nelle code non implica che i rendimenti abbiano la stessa
  ampiezza.
- Le soglie più estreme, come l'1%, contengono poche osservazioni e producono
  quindi stime meno stabili.
- I risultati dipendono dalla qualità e disponibilità dei dati Yahoo Finance.

## Fondamenti teorici

### Teorema di Sklar
Ogni funzione di distribuzione congiunta $H(x,y)$, con distribuzioni marginali $F(x)$ e $G(y)$, può essere scritta come:

$$H(x,y) = C\bigl(F(x),\, G(y)\bigr)$$

dove $C:[0,1]^2\to[0,1]$ è una **copula**, cioè una distribuzione congiunta con marginali uniformi in $[0,1]$. Se $F$ e $G$ sono continue, la copula è unica.

### Pseudo-osservazioni
Le marginali vengono stimate in modo non parametrico e i rendimenti sono trasformati mediante i ranghi:

$$u_i = \frac{\text{rank}(x_i)}{n+1}, \quad v_i = \frac{\text{rank}(y_i)}{n+1}$$

### Copula empirica
$$C_n(u,v) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}(u_i \le u,\; v_i \le v)$$

Questa quantità rappresenta la quota di osservazioni che soddisfa contemporaneamente entrambe le condizioni.

---

### Famiglie di copule parametriche

#### Clayton Copula ($\theta > 0$, lower-tail dependence)

**CDF:**
$$C_\theta(u,v) = \bigl(u^{-\theta} + v^{-\theta} - 1\bigr)^{-1/\theta}$$

**Density:**
$$c_\theta(u,v) = (1+\theta)\,(uv)^{-1-\theta}\,\bigl(u^{-\theta}+v^{-\theta}-1\bigr)^{-2-1/\theta}$$

**Lower tail dependence:** $\lambda_L = 2^{-1/\theta}$, $\lambda_U = 0$

---

#### Frank Copula ($\theta \in \mathbb{R}\setminus\{0\}$, no tail dependence)

**CDF:**
$$C_\theta(u,v) = -\frac{1}{\theta}\ln\!\left(1 + \frac{(e^{-\theta u}-1)(e^{-\theta v}-1)}{e^{-\theta}-1}\right)$$

**Density:**
$$c_\theta(u,v) = \frac{\theta(1-e^{-\theta})\,e^{-\theta(u+v)}}{\bigl[(1-e^{-\theta})-(1-e^{-\theta u})(1-e^{-\theta v})\bigr]^2}$$

**Tail dependence:** $\lambda_L = \lambda_U = 0$

---

#### Gumbel Copula ($\theta \ge 1$, upper-tail dependence)

**CDF:**
$$C_\theta(u,v) = \exp\!\Bigl(-\bigl((-\ln u)^\theta + (-\ln v)^\theta\bigr)^{1/\theta}\Bigr)$$

**Density** (letting $x=-\ln u$, $y=-\ln v$, $A=x^\theta+y^\theta$):
$$c_\theta(u,v) = C_\theta(u,v)\cdot\frac{(xy)^{\theta-1}}{uv}\cdot A^{2/\theta-2}\cdot\bigl(1+(\theta-1)A^{-1/\theta}\bigr)$$

**Upper tail dependence:** $\lambda_U = 2 - 2^{1/\theta}$, $\lambda_L = 0$

---

#### Gaussian Copula ($\rho \in (-1,1)$, no tail dependence)

**CDF:**
$$C_\rho(u,v) = \Phi_\rho\!\bigl(\Phi^{-1}(u),\,\Phi^{-1}(v)\bigr)$$

where $\Phi_\rho$ is the bivariate standard normal CDF with correlation $\rho$.

**Density:**
$$c_\rho(u,v) = \frac{1}{\sqrt{1-\rho^2}}\exp\!\left(-\frac{\rho^2(x^2+y^2)-2\rho xy}{2(1-\rho^2)}\right), \quad x=\Phi^{-1}(u),\; y=\Phi^{-1}(v)$$

**Tail dependence:** $\lambda_L = \lambda_U = 0$

---

#### Student-t Copula ($\rho \in (-1,1)$, $\nu > 2$, symmetric tail dependence)

**CDF:**
$$C_{\rho,\nu}(u,v) = t_{\rho,\nu}\!\bigl(t_\nu^{-1}(u),\,t_\nu^{-1}(v)\bigr)$$

where $t_{\rho,\nu}$ is the bivariate Student-t CDF with correlation $\rho$ and degrees of freedom $\nu$.

**Density** (letting $x=t_\nu^{-1}(u)$, $y=t_\nu^{-1}(v)$):
$$c_{\rho,\nu}(u,v) = \frac{t_{\nu+2,\rho}\!\left(\sqrt{\frac{\nu+2}{\nu}}\cdot(x,y)\right)}{t_\nu(x)\,t_\nu(y)}$$

equivalently computed in log-space as:

$$\ln c = \ln\Gamma\!\tfrac{\nu+2}{2} - \ln\Gamma\!\tfrac{\nu}{2} - \ln(\nu\pi) - \tfrac{1}{2}\ln(1-\rho^2) - \tfrac{\nu+2}{2}\ln\!\left(1+\frac{x^2-2\rho xy+y^2}{\nu(1-\rho^2)}\right) - \ln t_\nu(x) - \ln t_\nu(y)$$

**Symmetric tail dependence:**
$$\lambda_L = \lambda_U = 2\,t_{\nu+1}\!\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right)$$

---

### Mixture statica

La mixture utilizzata dal programma combina Clayton, Frank, Gumbel, Gaussian e Student-t:

$$C_{mix}(u,v) = \sum_{j=1}^{5} w_j\, C_j(u,v), \quad w_j \ge 0,\; \sum_{j=1}^{5} w_j = 1$$

I parametri delle singole componenti vengono prima stimati sugli stessi dati. I pesi sono poi determinati massimizzando la Log-Likelihood della densità mixture:

$$\ell(\mathbf{w}) = \sum_{i=1}^{n} \ln\!\left(\sum_{j=1}^{5} w_j\, c_j(u_i,v_i)\right)$$

Poiché l'ottimizzatore numerico minimizza, il programma risolve il problema equivalente:

$$\hat{\mathbf{w}} = \arg\min_{\mathbf{w}}\; -\ell(\mathbf{w}) \quad \text{con } w_j \ge 0,\; \sum_{j=1}^{5} w_j = 1$$

L'ottimizzazione usa SLSQP. I parametri delle componenti ($\theta$, $\rho$, $\nu$) restano fissati alle rispettive stime MLE e vengono ottimizzati soltanto i quattro pesi liberi.

Per il confronto tramite AIC e BIC, il codice usa $p=10$: sei parametri delle componenti e quattro pesi liberi. Un peso non è libero perché la somma deve essere uguale a uno.

$$\text{AIC} = 2p - 2\hat\ell, \qquad \text{BIC} = p\ln n - 2\hat\ell, \qquad p=10$$
