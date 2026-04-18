# copula_function_python_lab

The goal of this repository is to understand the copula functions to keep the non linear dependency, during extreme events, of two or more economic variables.

In this code it is possible to see how Gold and Silver are well correlated during extreme events.
Among the parametric copulas analyzed (Clayton, Frank, Gumbel, Gaussian, Student-t), the Student-t copula achieves the best fit according to AIC, BIC, log-likelihood and grid distance metrics (lowest MSE and Max|Delta| against the empirical copula C_n).
It captures symmetric tail dependence: when Gold experiences an extreme move (positive or negative), there is a high probability that Silver moves in the same direction.

However, empirically the tail dependence is asymmetric: the lower tail (joint crashes) is stronger than the upper tail (joint rallies), as confirmed by the empirical comparison across opposite quantiles — C_n(q,q) > P(U>1-q, V>1-q) at every tested level (1%, 5%, 10%, 15%).
The Student-t copula, being symmetric by construction, cannot fully capture this asymmetry. A Clayton copula or a skewed extension would be needed to model it exactly, but among standard parametric families the Student-t remains the closest approximation to the observed joint distribution.

A static mixture copula of Clayton, Frank and Gumbel was also estimated by maximising the joint log-likelihood over the three component weights (w_Clayton, w_Frank, w_Gumbel ≥ 0, sum = 1). This combines lower-tail specialisation (Clayton), central dependence (Frank) and upper-tail specialisation (Gumbel) into a single model. The mixture improves over each single Archimedean copula but does not surpass the Student-t in AIC/BIC, confirming that the elliptic family provides a better global description of the Gold–Silver joint distribution than any convex combination of the three Archimedean families.

---

## Mathematical Foundations

### Sklar's Theorem
Every joint CDF $H(x,y)$ with marginals $F(x)$ and $G(y)$ can be written as:

$$H(x,y) = C\bigl(F(x),\, G(y)\bigr)$$

where $C:[0,1]^2\to[0,1]$ is a **copula** — a joint CDF with uniform marginals on $[0,1]$. If $F$ and $G$ are continuous, $C$ is unique.

### Pseudo-observations
Marginals are estimated non-parametrically via the empirical CDF using ranks:

$$u_i = \frac{\text{rank}(x_i)}{n+1}, \quad v_i = \frac{\text{rank}(y_i)}{n+1}$$

### Empirical Copula
$$C_n(u,v) = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}(u_i \le u,\; v_i \le v)$$

---

### Parametric Copula Families

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

Code CDF: `x=-log(u); y=-log(v); exp(-((x**theta+y**theta)**(1/theta)))` ✓  
Code density: `c_val*multiplier*shape_term*correction` with each term matching the formula ✓

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

### Static Mixture Copula

A mixture copula is a convex combination of copula densities:

$$C_{mix}(u,v) = \sum_{j=1}^{k} w_j\, C_j(u,v), \quad w_j \ge 0,\; \sum_{j=1}^{k} w_j = 1$$

The weights $\mathbf{w}=(w_1,\ldots,w_k)$ are estimated by maximising the mixture log-likelihood function:

$$\ell(\mathbf{w}) = \sum_{i=1}^{n} \ln\!\left(\sum_{j=1}^{k} w_j\, c_j(u_i,v_i)\right)$$

Because numerical optimisers minimise, the code solves the equivalent minimisation:

$$\hat{\mathbf{w}} = \arg\min_{\mathbf{w}}\; -\ell(\mathbf{w}) \quad \text{subject to } w_j \ge 0,\; \sum_{j=1}^{k} w_j = 1$$

solved via SLSQP (Sequential Least Squares Programming). The parametric component parameters ($\theta$, $\rho$, $\nu$) are held fixed at their individual MLE estimates; only the $k-1$ free weights are optimised.

For model comparison, AIC and BIC are computed with $k-1$ free parameters (one weight is determined by the sum constraint):

$$\text{AIC} = 2(k-1) - 2\hat\ell, \qquad \text{BIC} = (k-1)\ln n - 2\hat\ell$$
