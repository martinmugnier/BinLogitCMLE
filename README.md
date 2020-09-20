# BinLogitCMLE
A Python implementation of Stata's clogit : Conditional Maximum Likelihood Estimation (CMLE) for binary logistic models in panel data with unobserved heterogeneity. 

The reader can refer to Woolridge (Section 15.8, 15.8.3) but we still put here some practical key features of the CMLE in the context of binary panel model. The key is that by assuming a logistic distribution of the error term we get that $n_i := \sum_{i=1}^T y_{it}$ is a sufficient statistics for estimating $\beta_0$ in the model : \begin{align} & \mathbb{P}(y_{it} = 1 | X_i, \gamma_i) = \mathbb{P}(y_{it} = 1 | X_{it}, \gamma_i) = \Lambda(X_{it}'\beta_0 + \gamma_i) \quad \text{where } \Lambda(x) = \frac{1}{1 + e^{-x}}, \forall x \in \mathbb{R} \ & y_{i1}, \ldots, y_{iT} \text{ are independent conditional on } (X_i, c_i) \end{align}

These two assumptions are key to ensure that $n_i$ is such that $y_i | X_i, \gamma_i, n_i$ has the same distribution than $y_i | X_i, n_i$. One can then estimate consistently $\beta_0$ through C.M.L.E, that is maximizing the following conditional log-likelihood of the model :

\begin{equation} \hat{\beta}^{\text{CMLE}}N \triangleq \arg\max{\beta \in \mathbb{R}^K} \ln L_N(\beta) \equiv \frac{1}{N} \sum_{i=1}^N \ell_i(\beta_0) = \frac{1}{N}\sum_{i=1}^N \ln \left{\exp\left(\sum_{t=1}^Ty_{it}X_{it}\beta_0 \right)\left[\sum_{a \in R_i} \exp \left(\sum_{t=1}^Ta_tX_{it}\beta_0 \right)\right]^{-1}\right} \end{equation}

where $L_N(\beta) = \prod_{i=1}^N\mathbb{P}(y_{i1} = y_1, \ldots, y_{iT} = y_T | X_i, c_i, n_i = n)$ and $R_i := {a \in {0,1}^T : \sum_{t=1}^Ta_t = n_i }$. The Newton-Raphson algorithm is particularly suited to the task since the hessian will be in general negative-definite when the parameter is identified (concave log-likelihood). By denoting $$U_i(\beta_0) : = \exp\left(\sum_{t=1}^Ty_{it}X_{it}\beta_0 \right)\left[\sum_{a \in R_i} \exp \left(\sum_{t=1}^Ta_tX_{it}\beta_0 \right)\right]^{-1}$$ , one has :

\begin{equation} \begin{split} \nabla_{\beta}U_i(\beta_0) = & \left(\sum_{t=1}^T y_{it}\textbf{X}{it} \right) \exp \left(\sum{t=1}^T y_{it}\textbf{X}{it}'\beta_0 \right) \left[ \sum{a \in R_i} \exp \left(\sum_{t=1}^T a_t \textbf{X}{it}'\beta_0 \right) \right]^{-1} \ & - \sum{a \in R_i} \left(\sum_{t=1}^T a_t \textbf{X}{it} \right)\exp \left(\sum{t=1}^T a_t \textbf{X}{it}'\beta_0 \right) \frac{\exp \left(\sum{t=1}^T y_{it}\textbf{X}{it}'\beta_0 \right)}{\left(\sum{a \in R_i} \exp \left(\sum_{t=1}^T a_t \textbf{X}_{it}'\beta_0 \right) \right)^2} \end{split} \end{equation}

So we have :

\begin{equation} \nabla_\beta \ell_i(\beta_0) = \frac{\nabla_\beta U_i(\beta_0)}{U_i(\beta_0)} = \sum_{t=1}^T y_{it}\textbf{X}{it} - \frac{\sum{a \in R_i} \left(\sum_{t=1}^T a_t \textbf{X}{it} \right)\exp \left(\sum{t=1}^T a_t \textbf{X}{it}'\beta_0 \right)}{\sum{a \in R_i} \exp \left(\sum_{t=1}^T a_t \textbf{X}_{it}'\beta_0 \right)} \end{equation}

Differentiating the above expression with respect to $\beta$ yields : \begin{equation} \begin{split} \nabla_\beta^2 \ell_i(\beta_0)= & \frac{\left[\sum_{a \in R_i} \left(\sum_{t=1}^T a_t \textbf{X}{it} \right) \exp \left(\sum{t=1}^T a_t \textbf{X}{it}'\beta_0 \right) \right]\left[\sum{a \in R_i} \left(\sum_{t=1}^T a_t \textbf{X}{it} \right)\exp \left(\sum{t=1}^T a_t \textbf{X}{it}'\beta_0 \right)\right]'}{\sum{a \in R_i} \exp \left(\sum_{t=1}^T a_t \textbf{X}{it}'\beta_0 \right)^2} \ & - \frac{\sum{a \in R_i} \left(\sum_{t=1}^T a_t \textbf{X}{it} \right)\left(\sum{t=1}^T a_t \textbf{X}{it}\right)' \exp \left(\sum{t=1}^T a_t \textbf{X}{it}'\beta_0 \right)}{\sum{a \in R_i} \exp \left(\sum_{t=1}^T a_t \textbf{X}_{it}'\beta_0 \right)} \end{split} \end{equation}

We report $$\hat{\beta}{t+1} = \hat{\beta}t - \left[\frac{1}{N}\sum{i=1}^N \nabla\beta^2 \ell_i(\hat{\beta}t)\right]^{-1} \left[\frac{1}{N}\sum{i=1}^N\nabla_\beta \ell_i(\hat{\beta}_t)\right] $$

$\hat{\beta}^{\text{CMLE}}$ is asymptotically normal ($\sqrt{N}(\hat{\beta}^{\text{CMLE}} - \beta_0) \overset{(d)}{\longrightarrow} \mathcal{N}\left(0, A_0^{-1} \right)$) with $A_0 = - \mathbb{E}[H_i(\beta_0)]$ so that the asymptotic variance of the estimator can be consistently estimated by $$ \left[\sum_{i=1}^N \nabla_\beta^2 \ell_i(\hat{\beta}^{\text{CMLE}})\right]^{-1} \text{ or } \left[\sum_{i=1}^N \nabla_{\beta}U_i(\hat{\beta}^{\text{CMLE}}) \nabla_{\beta}U_i(\hat{\beta}^{\text{CMLE}})'\right]^{-1} $$
