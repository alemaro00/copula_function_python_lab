# copula_function_python_lab

The goal of this repository is to understand the copula functions to keep the non linear dependency, during extreme events, of two or more economic variables.

In this code it is possible to see how Gold and Silver are well correlated during extreme events.
Among the parametric copulas analyzed (Clayton, Frank, Gumbel, Gaussian, Student-t), the Student-t copula achieves the best fit according to AIC, BIC, log-likelihood and grid distance metrics (lowest MSE and Max|Delta| against the empirical copula C_n).
It captures symmetric tail dependence: when Gold experiences an extreme move (positive or negative), there is a high probability that Silver moves in the same direction.

However, empirically the tail dependence is asymmetric: the lower tail (joint crashes) is stronger than the upper tail (joint rallies), as confirmed by the empirical comparison across opposite quantiles — C_n(q,q) > P(U>1-q, V>1-q) at every tested level (1%, 5%, 10%, 15%).
The Student-t copula, being symmetric by construction, cannot fully capture this asymmetry. A Clayton copula or a skewed extension would be needed to model it exactly, but among standard parametric families the Student-t remains the closest approximation to the observed joint distribution.
