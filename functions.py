"""
The following functions compute the main quantities presented in the paper: 
  sharpe_ratio_variance(SR, T, γ₃, γ₄, K)
  minimum_track_record_length(SR, SR₀, γ₃, γ₄, α)
  probabilistic_sharpe_ratio(SR, SR₀, T, γ₃, γ₄)
  critical_sharpe_ratio(SR₀, T, γ₃, γ₄, α, K)
  sharpe_ratio_power(SR₀, SR₁, T, γ₃, γ₄, α)
  pFDR(p₁, α, β)
  oFDR(SR, SR₀, SR₁, T, p₁, γ₃, γ₄)
  expected_maximum_sharpe_ratio(K, σ₀², SR₀)
  FDR_critical_value(q, SR₀, SR₁, σ₀, σ₁, p₁)
  control_for_FDR(q, SR₀, SR₁, p₁, T, γ₃, γ₄, K)

We use the following notations: 
  SR: observed Sharpe ratio
  γ₃: skewness of the returns
  γ₄: (non-excess) kurtosis of the returns
  ρ: autocorrelation of the returns, assumed to be AR(1)
  T: number of observations
  SR₀: Sharpe ratio under H₀
  SR₂: Sharpe ratio under H₁
  K: number of trials (we only see the maximum Sharpe ratio of K trials)
  α: significance level, P[SR>SR_c|H₀]
  1-β: power, P[reject H₀ | H₁] = P[SR>SR_c|H₁]
  β: type II error, P[SR<SR_c|H₁]
  p₁ = P[H₁]
  σ₀: standard deviation of the Sharpe ratios under H₀
  σ₁: standard deviation of the Sharpe ratios under H₁
  q: desired false discovery date (FDR)

To account for multiple testing (K>1), the following may be useful:
  effective_rank(C)
  number_of_clusters(C)
  robust_covariance_inverse(V)
  minimum_variance_weights_for_correlated_assets(V)
  variance_of_the_clustered_trials(X, clusters)

The following function reproduces the numeric example in the paper:
  test_numeric_example()

The following functions were used to generate sample data:
  get_random_correlation_matrix(n, k, T, ε)
  generate_non_gaussian_data(T, n, SR₀, name)
  generate_autocorrelated_non_gaussian_data(T, n, SR₀, name, ρ)

The following functions compute the classical multiple testing adjustments: 
  adjusted_p_values_bonferroni(ps)
  adjusted_p_values_sidak(ps)
  adjusted_p_values_holm(ps)
  scipy.stats.false_discovery_control(ps), statsmodels.stats.multitest.fdrcorrection(ps, q)

The following functions were used to compute the variance of the maximum of k Sharpe ratios
  moments_Mk(k, ρ)
  make_expectation_gh(N)

The following functions are used to pretty-print p-values
  significance_stars(p)
  round_p_value(p)
"""

from deprecated import deprecated
import warnings
import math
import scipy.stats
import scipy.interpolate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def make_expectation_gh(n_nodes=200):
    """
    Expectation under Z ~ N(0,1) via Gauss–Hermite:
        E[g(Z)] = (1/√π) * Σ w_i * g(√2 * t_i), 
    where (t_i, w_i) are GH nodes/weights
    
    Inputs:
    - n_nodes: int, number of nodes
    Outputs:
    - E: function to compute the expectation, f↦E[f(X)]
    """
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    scale = np.sqrt(2.0)
    norm = 1.0 / np.sqrt(np.pi)
    x = scale * nodes
    def E(g):
        vals = g(x)
        return norm * np.dot(weights, vals)
    return E

E_under_normal = make_expectation_gh(n_nodes=200)

def moments_Mk(k, *, rho=0):
    """
    Moments of M_k = Max(Z_1, Z_2, ..., Z_k), where the Z_i are i.i.d. N(0,1)
    
    The density is 
        f_M(x) = k φ(x) Φ(x)^(k-1),
    so the moments are 
        E[M_k^r] = k * E[ Z^r * Φ(Z)^(k-1) ].

    For the correlated case, we assume equi-correlation,
        Zᵢ = √ρX + √(1-ρ)Yᵢ
        M = √ρX + √(1-ρ)Max(Yᵢ)

    Inputs:
    - k: int, number of variables
    - rho: float, correlation coefficient
    Outputs:
    - Ez: float, expectation of M_k
    - Ez2: float, expectation of M_k^2
    - var: float, variance of M_k
    """
    Phi = scipy.stats.norm.cdf
    Ez  = E_under_normal(lambda z: k * z    * (Phi(z) ** (k - 1)))
    Ez2 = E_under_normal(lambda z: k * z*z  * (Phi(z) ** (k - 1)))
    var = Ez2 - Ez**2

    Ez = (1-rho) * Ez
    var = rho + (1-rho) * var
    Ez2 = var + Ez**2

    return Ez, Ez2, var


def significance_stars(p: float) -> str:
    """
    Significance stars for a p-value
    """
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '.' if p < .1 else ''


def round_p_value(p: float) -> str:
    """
    Round a p-value to 2 decimal places if above 0.01, 3 if above 0.001, and zero if below.
    """
    if p == 0: 
        return "0"
    if p > .01:
        return f"{p:.2f}"
    if p > .001:
        return f"{p:.3f}"
    return f"10^{int(math.log10(p))}"


def effective_rank(C: np.ndarray) -> float:
    """
    Effective rank of a correlation matrix.

    The effective rank of a positive semi-definite matrix is computed as follows: 
    - Compute the eigenvalues; they are non-negative
    - Discard zeros
    - Normalize the remaining eigenvalues to sum to 1, so we can interpret them as a probability distribution
    - Compute its entropy
    - "Invert" the entropy to have an effective number of items 
      (the number of items for which the uniform distribution would have the same entropy)

    Input:  C: np.ndarray, square, a positive semi-definite matrix
    Output: float, effective rank

    [1] The effective rank: a measure of effective dimensionality
        O. Roy and M. Vetterli (2007)
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.177.2721
    """
    p = np.linalg.eigvalsh(C)
    p = p[ p > 0 ]
    p = p / sum(p)
    H = np.sum( - p * np.log(p) )
    return math.exp(H)


def test_effective_rank():
    np.random.seed(1)
    x = np.random.normal( size=(10,3) )
    C = np.corrcoef(x.T)
    effective_rank(C)  # Almost 3
    assert abs( effective_rank( np.eye(3) ) - 3 ) < 1e-12
    C = np.array([[10,1,7],[1,10,8],[7,8,10]]) / 10
    assert abs( effective_rank(C[:2,:2]) - 2 ) < .02
    assert abs( effective_rank(C) - 1.84 ) < .01


def number_of_clusters(
        C: np.ndarray,
        *,
        retries: int = 10,
        max_clusters: int = 100,
        plot: bool = False,
) -> tuple[int, pd.Series, np.ndarray]:
    """
    Compute the optimal number of clusters, from a correlation matrix

    Algorithm in section 8.1 of [1], without the third point (i.e., no recursive re-clustering of low-quality clusters): 
    - Convert the correlation matrix into a distance matrix
    - Using the columns of the distance matrix as features, run the k-means algorithm, for all k, 
      and compute the "quality" of the clustering
    - Keep the clustering with the highest quality
    The quality is computed as the mean of the silhouette scores, divided by their standard deviation.

    References: 
    [1] Detection of false investment strategies using unsupervised learning methods
        M. Lopez de Prado (2018)
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017

    Inputs: 
    - C: np.ndarray, square, a correlation matrix (symmetric, positive semi-definite, with ones on the diagonal)
    - retries: int, number of times to run the k-means algorithm
    - max_clusters: int, maximum number of clusters to consider
    - plot: bool, whether to plot the quality of the clustering
    Outputs:
    - number_of_clusters: int, the optimal number of clusters
    - qualities: pd.Series, quality of the clusterings, indexed by the number of clusters
    - clusters: np.ndarray, cluster assignment for each observation, for the optimal number of clusters
    """

    # Check this looks like a correlation matrix
    assert isinstance( C, np.ndarray )
    assert np.all( -1 <= C )
    assert np.all( C <= 1 )
    assert np.all( np.diag(C) == 1 )
    assert np.all( np.isfinite( C ) )

    max_clusters = min( max_clusters, C.shape[0]-1 )

    # Compute the distances
    D = np.sqrt( (1-C)/2 )
    assert np.all( np.isfinite( D ) )

    # For all values of k:
    # - run the k-means algorithm on D
    # - compute the silhouette score of each observation, S[i],
    # - compute the quality of the clustering, q = E[S]/Std[S]
    # - Do that several times and keep the maximum quality
    qualities = {}
    clusters = {}
    for k in range(2, max_clusters+1):
        qualities[k] = - np.inf
        for _ in range(retries):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(D)  # Use the distances as features, i.e., compute distances between columns of the distance matrix
            labels = kmeans.labels_
            silhouette_vals = silhouette_samples(D, labels)
            q = silhouette_vals.mean() / silhouette_vals.std()
            if q > qualities[k]:
                qualities[k] = max(qualities[k], q)
                clusters[k] = labels
    qualities = pd.Series( qualities )
    number_of_clusters = qualities.idxmax()
    clusters = clusters[number_of_clusters]

    if plot:
        fig, ax = plt.subplots( figsize = (4,3), layout = 'constrained')
        ax.plot( qualities )
        i = np.argmax(qualities)
        x, y = qualities.index[i], qualities.iloc[i]
        ax.scatter( x, y )
        ax.text( x, y, f"  {qualities.index[i]}", va='center', ha='left' )
        ax.set_xlabel( "Number of clusters" )
        ax.set_ylabel( "Quality" )
        plt.show()

    return number_of_clusters, qualities, clusters


def get_random_correlation_matrix(
    number_of_trials: int = 100,
    effective_number_of_trials: int = 10,
    number_of_observations: int = 200,
    noise: float = .1,
):
    """
    Generate a correlation matrix with a block structure

    Inputs: 
    - number_of_trials: int, number of time series to generate; size of the correlation matrix
    - effective_number_of_trials: int, number of clusters
    - number_of_observations: int, number of observations to generate
    - noise: float, noise level
    Outputs:
    - C: np.ndarray, square, a correlation matrix
    - X: np.ndarray, matrix of observations, with number_of_observations rows and number_of_trials columns
    - clusters: np.ndarray, cluster assignment for each trial (column)
    """
    while True:
        block_positions = [0] + sorted( np.random.choice( number_of_trials, effective_number_of_trials-1, replace = True ) ) + [number_of_trials]
        block_sizes = np.diff( block_positions )
        if np.all(block_sizes > 0):
            break

    clusters = np.array( [
        block_number
        for block_number, size in enumerate( block_sizes )
        for _ in range(size)
    ] )
    X0 = np.random.normal( size = (number_of_observations, effective_number_of_trials ) )
    X = np.zeros( shape = (number_of_observations, number_of_trials) )
    for i, cluster in enumerate( clusters ):
        X[:,i] = X0[:,cluster] + noise * np.random.normal(size = number_of_observations)
    C = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(C, 1)  # rounding errors
    C = np.clip( C, -1, 1 )
    return C, X, clusters


def generate_non_gaussian_data( 
    nr: int, 
    nc: int, 
    *, 
    SR0: float = 0, 
    name: str = 'severe', 
) -> np.ndarray:
    """
    Generate non-Gaussian data

    Inputs:
    - nr: int, number of rows
    - nc: int, number of columns
    - SR0: float, the target Sharpe ratio
    - name: str, distribution (gaussian, mild, moderate, severe)
    Outputs:
    - X: np.ndarray, matrix of observations, shape (nr, nc)
    """
    configs = {
        "gaussian": (0,    0,     0.015, 0.010),
        "mild":     (0.04, -0.03, 0.015, 0.010),
        "moderate": (0.03, -0.045, 0.020, 0.010),
        "severe":   (0.02, -0.060, 0.025, 0.010),
    }
    assert name in configs

    def mixture_variance(p_tail, mu_tail, sigma_tail, mu_core, sigma_core):
        w = 1.0 - p_tail
        mu = w*mu_core + p_tail*mu_tail
        m2 = w*(sigma_core**2 + mu_core**2) + p_tail*(sigma_tail**2 + mu_tail**2)
        return m2 - mu**2

    def gen_with_true_SR0(reps, T, cfg, SR0):
        p, mu_tail, sig_tail, sig_core = cfg
        # Zero-mean baseline mixture (choose mu_core so mean=0)
        mu_core0 = - p*mu_tail/(1.0 - p)
        std0 = np.sqrt(mixture_variance(p, mu_tail, sig_tail, mu_core0, sig_core))
        mu_shift = SR0 * std0  # sets population Sharpe to SR0, preserves skew/kurt
        mask = np.random.uniform(size=(reps, T)) < p
        X = np.random.normal(mu_core0 + mu_shift, sig_core, size=(reps, T))
        X[mask] = np.random.normal(mu_tail + mu_shift, sig_tail, size=mask.sum())
        return X

    return gen_with_true_SR0( nr, nc, configs[name], SR0 )


def generate_autocorrelated_Gaussian_data( nr, nc, rho = .5 ): 
    X = np.random.normal( size = (nr, nc) )
    for i in range(1, nr): 
        X[i,:] = rho*X[i-1,:] + X[i,:]
    return X

def enforce_marginals( gaussian, marginals ): 
    assert gaussian.shape == marginals.shape
    shape = gaussian.shape
    gaussian = gaussian.flatten()
    marginals = marginals.flatten()
    i = np.argsort( gaussian )
    i = np.argsort( gaussian )
    j = np.argsort( marginals )
    ii = np.argsort( i )
    #jj = np.argsort( j )
    result = marginals[j[ii]]
    assert np.all( np.array( sorted( result ) ) == np.array( sorted( marginals ) ) )
    assert np.all( np.argsort( gaussian ) == np.argsort( result ) )
    return result.reshape( shape )

def autocorrelation( X ): 
    nr, nc = X.shape
    ac = np.zeros( nc )
    for i in range( nc ): 
        ac[i] = np.corrcoef( X[1:,i], X[:-1,i] )[0,1]
    return ac.mean()

"""
The mapping between gaussian and non-gaussian autocorrelations was obtained as follows: 
    N = 100_000
    values = []
    for name in [ "gaussian", "mild", "moderate", "severe" ]:
        for rho in tqdm(np.linspace( -1, 1, 21 )[1:-1]):
            marginals = generate_non_gaussian_data( N, 3, SR0 = 0, name = name )
            gaussian = generate_autocorrelated_Gaussian_data( N, 3, rho = rho )
            X = enforce_marginals( gaussian, marginals )
            values.append( { 
                'name': name, 
                'rho_input': rho, 
                'rho_output': autocorrelation( X ),
            } )
    values = pd.DataFrame( values )
    values.pivot( index = 'rho_input', columns = 'name', values = 'rho_output').plot()
"""
values = """name,rho_input,rho_output
gaussian,-0.9,-0.8988
gaussian,-0.8,-0.8005
gaussian,-0.7,-0.6995
gaussian,-0.6,-0.5993
gaussian,-0.5,-0.5006
gaussian,-0.4,-0.3987
gaussian,-0.3,-0.3009
gaussian,-0.2,-0.2019
gaussian,-0.1,-0.0987
gaussian,0.0,0.0004
gaussian,0.1,0.0999
gaussian,0.2,0.2
gaussian,0.3,0.3
gaussian,0.4,0.3991
gaussian,0.5,0.5018
gaussian,0.6,0.599
gaussian,0.7,0.7002
gaussian,0.8,0.8008
gaussian,0.9,0.9007
mild,-0.9,-0.8519
mild,-0.8,-0.7583
mild,-0.7,-0.6661
mild,-0.6,-0.57
mild,-0.5,-0.4722
mild,-0.4,-0.3807
mild,-0.3,-0.2855
mild,-0.2,-0.1891
mild,-0.1,-0.098
mild,0.0,0.0003
mild,0.1,0.0979
mild,0.2,0.1928
mild,0.3,0.2889
mild,0.4,0.3859
mild,0.5,0.4871
mild,0.6,0.5884
mild,0.7,0.6884
mild,0.8,0.7904
mild,0.9,0.8922
moderate,-0.9,-0.7798
moderate,-0.8,-0.6966
moderate,-0.7,-0.6078
moderate,-0.6,-0.5197
moderate,-0.5,-0.436
moderate,-0.4,-0.3489
moderate,-0.3,-0.2625
moderate,-0.2,-0.1768
moderate,-0.1,-0.0888
moderate,0.0,0.0005
moderate,0.1,0.0873
moderate,0.2,0.1767
moderate,0.3,0.2717
moderate,0.4,0.3661
moderate,0.5,0.4624
moderate,0.6,0.5627
moderate,0.7,0.6628
moderate,0.8,0.7671
moderate,0.9,0.8804
severe,-0.9,-0.7334
severe,-0.8,-0.646
severe,-0.7,-0.5692
severe,-0.6,-0.4897
severe,-0.5,-0.4104
severe,-0.4,-0.3269
severe,-0.3,-0.2451
severe,-0.2,-0.1661
severe,-0.1,-0.0799
severe,0.0,0.0018
severe,0.1,0.0832
severe,0.2,0.1663
severe,0.3,0.258
severe,0.4,0.349
severe,0.5,0.4419
severe,0.6,0.5443
severe,0.7,0.6368
severe,0.8,0.7422
severe,0.9,0.8656
"""
values = pd.read_csv(StringIO(values))

def compute_input_rho(name, rho_output):
    subset = values[values['name'] == name].sort_values('rho_output')
    x = subset['rho_output']
    y = subset['rho_input']
    f = scipy.interpolate.interp1d(x, y, bounds_error=False, fill_value="extrapolate")
    return float(f(rho_output))

def generate_autocorrelated_non_gaussian_data( N, n, SR0 = 0, name = "gaussian", rho = 0 ): 
    rho_input = compute_input_rho( name, rho )
    marginals = generate_non_gaussian_data( N, n, SR0 = SR0, name = name )
    gaussian = generate_autocorrelated_Gaussian_data( N, n, rho = rho_input )
    X = enforce_marginals( gaussian, marginals )
    return X


def sharpe_ratio_variance( 
    SR: float, 
    T: int, 
    *, 
    gamma3: float = 0., 
    gamma4: float = 3.,
    rho: float = 0.,
    K: int = 1,
) -> float:
    """
    Asymptotic variance of the Sharpe ratio

    Inputs:
    - SR: float, Sharpe ratio
    - T: int, number of observations
    - gamma3: float, skewness
    - gamma4: float, (non-excess) kurtosis
    - rho: float, autocorrelation
    - K: int, number of strategies whose Sharpe ratios we take the maximum of -- larger K means smaller variance
    Outputs:
    - float, the variance of the Sharpe ratio
    """
    A = 1
    B = rho / ( 1 - rho )
    C = rho**2 / ( 1 - rho**2 )
    a = A + 2 * B
    b = A + B + C
    c = A + 2 * C
    V = ( a*1 - b * gamma3 * SR + c * (gamma4-1)/4 * SR**2 ) / T
    return V * moments_Mk(K)[2]

def test_sharpe_ratio_variance():
    assert round( math.sqrt( sharpe_ratio_variance( SR = .036 / .079, gamma3 = -2.448, gamma4 = 10.164, T = 24 ) ), 3 ) == .329
    assert round( math.sqrt( sharpe_ratio_variance( SR = .036 / .079, gamma3 = 0,      gamma4 = 3,      T = 24 ) ), 3 ) == .214


def minimum_track_record_length( 
    SR: float, 
    SR0: float, 
    *, 
    gamma3: float = 0., 
    gamma4: float = 3., 
    rho: float = 0.,
    alpha: float = 0.05,
) -> float:
    """
    Minimum track record length for the Sharpe ratio to be significantly greater than SR0, at the confidence level alpha.

    Inputs:
    - SR: float, observed Sharpe ratio
    - SR0: float, Sharpe ratio under H0
    - gamma3: float, skewness
    - gamma4: float, (non-excess) kurtosis
    - alpha: float, confidence level
    Outputs:
    - float, minimum track record length
    """
    return sharpe_ratio_variance( SR0, T = 1, gamma3 = gamma3, gamma4 = gamma4, rho = rho, K = 1 ) * ( scipy.stats.norm.ppf(1-alpha) / (SR - SR0) ) ** 2
    #return ( 1 - gamma3 * SR0 + (gamma4-1)/4*SR0**2 ) * ( scipy.stats.norm.ppf(1-alpha) / (SR - SR0) ) ** 2

def test_minimum_track_record_length():
    assert round( minimum_track_record_length( SR = .036 / .079, SR0 = 0, gamma3 = -2.448, gamma4 = 10.164, alpha = .05 ), 3 ) == 13.029


def probabilistic_sharpe_ratio( 
    SR: float, 
    SR0: float, 
    *,
    variance: float = None,
    T: int = None, 
    gamma3: float = 0., 
    gamma4: float = 3.,
    rho: float = 0.,
    K: int = 1,
) -> float:
    """
    Probabilistic Sharpe Ratio (PSR)

    This is 1-p, where p is the p-value of the test  H0: SR=SR0  vs  H1: SR>SR0.
    It can be interpreted as a Sharpe ratio "on a probability scale", i.e., in [0,1].

    TODO: In case of multiple testing, we currently expect SR0 to be already adjusted; this function will only adjust the variance

    Inputs: 
    - SR: float, observed Sharpe ratio
    - SR0: float, Sharpe ratio under H0
    - T: int, number of observations
    - gamma3: float, skewness
    - gamma4: float, (non-excess) kurtosis
    - rho: float, autocorrelation
    - K: int, number of strategies whose Sharpe ratios we take the maximum of -- larger K means smaller variance
    Outputs:
    - float, probabilistic Sharpe ratio
    """
    if variance is None:
        variance = sharpe_ratio_variance( SR0, T, gamma3 = gamma3, gamma4 = gamma4, rho = rho, K = K )
    else: 
        assert T is None, "Provide either the variance or (T, gamma3, gamma4, rho)"
    return scipy.stats.norm.cdf( (SR - SR0) / math.sqrt(variance) )


def test_probabilistic_sharpe_ratio():
    assert round( probabilistic_sharpe_ratio( SR = .036 / .079, SR0 = 0,  T = 24, gamma3 = -2.448, gamma4 = 10.164), 3 ) == .987
    assert round( probabilistic_sharpe_ratio( SR = .036 / .079, SR0 = .1, T = 24, gamma3 = -2.448, gamma4 = 10.164), 3 ) == .939


def critical_sharpe_ratio(
    SR0: float, 
    T: int, 
    *, 
    gamma3: float = 0., 
    gamma4: float = 3., 
    rho: float = 0.,
    alpha: float = 0.05,
    K: int = 1,
) -> float:
    """
    Critical value for the test  H0: SR=SR0  vs  H1: SR>SR0.

    Inputs:
    - SR0: float, Sharpe ratio under H0
    - T: int, number of observations
    - gamma3: float, skewness
    - gamma4: float, (non-excess) kurtosis
    - rho: float, autocorrelation
    - alpha: float, confidence level
    - K: int, number of strategies whose Sharpe ratios we take the maximum of -- larger K means smaller variance
    Outputs:
    - float, critical value
    """
    variance = sharpe_ratio_variance(SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)
    return SR0 + scipy.stats.norm.ppf(1 - alpha) * math.sqrt(variance)


def sharpe_ratio_power( 
    SR0: float, 
    SR1: float, 
    T: int, 
    *, 
    gamma3: float = 0., 
    gamma4: float = 3., 
    rho: float = 0.,
    alpha: float = 0.05,
    K: int = 1,
) -> float:
    """
    Power (1-β) of the test  H0: SR=SR0  vs  H1: SR=SR1.

    Remarks: 
    - To compute the power, we need to know more about the alternative hypothesis: 
      SR1 could be the average Sharpe ratio of strategies with positive excess returns
    - "Power" is the same thing as "recall" in classification:
            Power = P[ reject H0 | H1 ]
                  = P[ H1 ∧ reject H0 | H1 ]
                  = TP / (TP + FN )
                  = recall

    Inputs:
    - SR0: float, Sharpe ratio under H0
    - SR1: float, Sharpe ratio under H1
    - T: int, number of observations
    - gamma3: float, skewness
    - gamma4: float, (non-excess) kurtosis
    - rho: float, autocorrelation
    - alpha: float, confidence level
    - K: int, number of strategies whose Sharpe ratios we take the maximum of -- larger K means smaller variance
    Outputs:
    - float, power
    """
    critical_SR = critical_sharpe_ratio(SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, alpha=alpha)
    variance = sharpe_ratio_variance(SR1, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K)
    beta = scipy.stats.norm.cdf((critical_SR - SR1) / math.sqrt(variance))
    return 1 - beta

def test_sharpe_ratio_power():
    assert round( 1 - sharpe_ratio_power( SR0=0, SR1 = .5, T = 24, gamma3 = -2.448, gamma4 = 10.164 ), 3 ) == .315


def pFDR( 
    p_H1: float, 
    alpha: float, 
    beta: float,
) -> float:
    """
    Posterior probability of H0, given that SR>SR_c

    Remarks: 
    - Needs beta=1-power (which needs, SR1), and p[H1], perhaps estimated as the proportion of deployed profitable strategies
    - This does not use the observed Sharpe ratio, only the critical value for the test at level alpha; see oFDR

    Inputs:
    - p_H1: float, probability that H1 is true
    - alpha: float, confidence level
    - beta: float, 1 - power, i.e., type II error, i.e., the probability of not rejecting H0 when H1 is true
    Outputs:
    - float, posterior probability of H0
    """
    p_H0 = 1 - p_H1
    return 1 / ( 1 + (1-beta) * p_H1 / alpha / p_H0 )

@deprecated(reason="This was renamed to pFDR")
def bayesian_fdr( *args, **kwargs ):
    return pFDR( *args, **kwargs )

def test_pFDR():
    assert round( pFDR( .05, .05, .315 ), 3 ) == .581


def oFDR( 
    SR: float, 
    SR0: float,
    SR1: float, 
    T: int, 
    p_H1: float, 
    *, 
    gamma3: float = 0., 
    gamma4: float = 3.,
    rho: float = 0.,
    K: int = 1,
) -> float:
    """
    Posterior probability of H0, given that SR>SR_obs

    In case of multiple testing, we currently expect SR0 and SR1 to be already adjusted; this function will only adjust the variance

    Inputs:
    - SR: float, observed Sharpe ratio
    - SR0: float, Sharpe ratio under H0
    - SR1: float, Sharpe ratio under H1
    - T: int, number of observations
    - p_H1: float, probability that H1 is true
    - gamma3: float, skewness
    - gamma4: float, (non-excess) kurtosis
    - rho: float, autocorrelation
    - K: int, number of strategies whose Sharpe ratios we take the maximum of -- larger K means smaller variance
    Outputs:
    - float, posterior probability of H0
    """
    p0 = 1 - probabilistic_sharpe_ratio( SR, SR0, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K )
    p1 = 1 - probabilistic_sharpe_ratio( SR, SR1, T=T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K )
    p_H0 = 1 - p_H1
    return p0 * p_H0 / ( p0 * p_H0 + p1 * p_H1 )

@deprecated(reason="This was renamed to oFDR")
def posterior_p_value( *args, **kwargs ):
    return oFDR( *args, **kwargs )

def test_oFDR():
    assert round( oFDR( SR = .036 / .079, SR0=0, SR1=.5, T=24, p_H1=.05, gamma3 = -2.448, gamma4 = 10.164 ), 3 ) == .306


def robust_covariance_inverse( V: np.ndarray ) -> np.ndarray:
    r"""
    Inverse of a constant-correlation covariance matrix, using the Sherman–Morrison formula

    Assume $V = \rho \sigma \sigma' + (1-\rho) \text{diag}(\sigma^2)$ (variance matrix, with constant correlations).
    Its inverse is $V^{-1} = A^{-1} - \dfrac{ A^{01} \rho \sigma \sigma' A^{-1} }{ 1 + \rho \sigma' A^{-1} \sigma }$.

    Input:
    - V: np.ndarray, variance matrix
    Output:
    - np.ndarray, inverse of the variance matrix
    """
    sigma = np.sqrt( np.diag(V) )
    C = (V.T/sigma).T/sigma
    rho = np.mean( C[ np.triu_indices_from(C,1) ] )
    A = np.diag( 1 / sigma**2 ) / (1-rho)
    sigma = sigma.reshape( -1, 1 )
    return A - ( rho * A @ sigma @ sigma.T @ A ) / ( 1 + rho * sigma.T @ A @ sigma )

def test_robust_covariance_inverse():
    np.random.seed(0)
    rho = .5
    C = rho * np.ones( shape=(10,10) )
    np.fill_diagonal(C, 1)
    sigma = np.random.lognormal( size=10 ).reshape( -1, 1 )
    V = (C*sigma).T*sigma
    assert np.all( np.abs( np.linalg.inv(V) - robust_covariance_inverse(V) ) < 1e-12 )


def minimum_variance_weights_for_correlated_assets(V: np.ndarray) -> np.ndarray:
    """
    Weights of the minimum variance portfolio, for correlated assets, assuming a constant-correlation covariance matrix

    Input:
    - V: np.ndarray, variance matrix, shape (n,n)
    Output:
    - np.ndarray, weights of the minimum variance portfolio, shape (n,)
    """
    ones = np.ones( shape = V.shape[0] )
    S = robust_covariance_inverse(V)
    w = S @ ones
    w = w / np.sum(w)
    return w

def test_minimum_variance_weights_for_correlated_assets():
    np.random.seed(0)
    rho = .5
    C = rho * np.ones( shape=(10,10) )
    np.fill_diagonal(C, 1)
    sigma = np.random.lognormal( size=10 ).reshape( -1, 1 )
    V = (C*sigma).T*sigma
    w = minimum_variance_weights_for_correlated_assets(V)

    import cvxpy as cp
    W = cp.Variable( shape = V.shape[0] )
    problem = cp.Problem(
        cp.Minimize( cp.quad_form(W, V) ),
        [W.sum() == 1]
    )
    problem.solve()
    assert np.all( np.abs( W.value - w ) < 1e-10 )


def variance_of_the_clustered_trials(X: np.ndarray, clusters: np.ndarray) -> tuple[float, np.ndarray, pd.DataFrame]:
    """
    Compute the returns of a minimum variance portfolio in each cluster, 
    the corresponding Sharpe ratios, 
    and then the variance of those Sharpe ratios.

    Inputs:
    - X: numpy array or returns, one column per strategy
    - cluster: cluster assignment, list (or array) with one element per strategy
    Outputs:
    - the variance of the Sharpe ratios fo the cluster portfolios
    - The Sharpe ratios of the cluster portfolios
    - The time series of returns of the cluster portfolios
    """
    assert X.shape[1] == len(clusters)
    # Minimum variance portfolio in each cluster (assuming constant correlation)
    y = {}
    for i in np.unique( clusters ):
        j = clusters == i
        if j.sum() == 1:
            y[i] = X[:,j][:,0]
        else:
            Y = X[:,j]
            V = np.cov(Y.T)
            w = minimum_variance_weights_for_correlated_assets(V)
            y[i] = np.sum( Y * w, axis = 1 )
    y = pd.DataFrame( y )

    # Sharpe ratios
    SRs = y.mean() / y.std()

    return SRs.var(), SRs, y


def expected_maximum_sharpe_ratio( number_of_trials: int, variance: float, SR0: float = 0 ) -> float:
    """
    Expected maximum Sharpe ratio; add SR0 if non-zero

    Inputs:
    - number_of_trials: int, number of trials
    - variance: float, variance of the Sharpe ratios
    Outputs:
    - float, expected maximum Sharpe ratio
    """
    return SR0 + (
        np.sqrt( variance ) * (
            ( 1 - np.euler_gamma ) * scipy.stats.norm.ppf( 1 - 1 / number_of_trials ) +
            np.euler_gamma * scipy.stats.norm.ppf( 1 - 1 / number_of_trials / np.exp(1) )
        )
    )

def variance_of_the_maximum_of_k_Sharpe_ratios( number_of_trials: int, variance: float ) -> float:
    """
    Variance of the maximum of K Sharpe ratios
    """
    return variance * moments_Mk(number_of_trials)[2]


@deprecated(reason="Not used")
def deflated_sharpe_ratio(X: np.ndarray, *, verbose: bool = False, cluster: bool = True, details: bool = False) -> float:
    gamma3 = scipy.stats.skew(X.flatten())                    # Skewness
    gamma4 = scipy.stats.kurtosis(X.flatten(), fisher=False)  # Kurtosis (not excess kurtosis)
    T = X.shape[0]

    if cluster:

        # Correlation matrix
        C = np.corrcoef( X.T )
        C = np.clip( C, -1, +1 )
        np.fill_diagonal(C, 1)

        # Number of clusters
        number_of_trials, qualities, clusters = number_of_clusters(C)
        assert number_of_trials == clusters.max() + 1

        variance, SRs, y = variance_of_the_clustered_trials(X, clusters)

        # Use the pooled data instead of the individual (aggregated) time series:
        # since y contains returns averaged over several strategies, they are more Gaussian
        ps = np.array( [
            1 - probabilistic_sharpe_ratio( u, SR0 = 0, T = X.shape[0], gamma3 = gamma3, gamma4 = gamma4 )
            for u in SRs
        ] )

    else:

        number_of_trials = X.shape[1]
        SRs = X.mean(axis=0) / X.std(axis=0)
        variance = np.var( SRs )

        ps = np.array( [
            1 - probabilistic_sharpe_ratio(
                X[:,i].mean() / X[:,i].std(),
                SR0 = 0,
                T = X.shape[0],
                gamma3 = scipy.stats.skew( X[:,i] ),                     # TODO: Should we use the pooled values of gamma3 and gamma4?
                gamma4 = scipy.stats.kurtosis( X[:,i], fisher=False )
            )
            for i in range(X.shape[1])
        ] )

    SR0 = expected_maximum_sharpe_ratio( number_of_trials, variance )
    DSR = probabilistic_sharpe_ratio(SRs.max(), SR0 = SR0, T = T, gamma3 = gamma3, gamma4 = gamma4 )

    if verbose:
        print( f"gamma3            = {gamma3:.3f}" )
        print( f"gamma4            = {gamma4:.3f}" )
        print( f"T                 = {T}" )
        print( f"number_of_trials  = {number_of_trials}" )
        print( f"Var[ SR ]         = {math.sqrt(variance):.3f}²" )
        print( f"SR0 = E[ max SR ] = {SR0:.3f}" )
        print( f"max SR            = {SRs.max():.3f}" )
        print( f"max SR  (raw)     = {np.max( X.mean(axis=0) / X.std(axis=0) ):.3f}" )
        print( f"DSR               = {DSR:.3f}" )
        print( f"1 - DSR           = {1-DSR:.3f} {'***' if DSR>.999 else '**' if DSR>.99 else '*' if DSR>.95 else '.' if DSR>.9 else ''}" )
        print( f"min p             = {ps.min():.3f}" )
        print( f"p (Bonferroni)    = {adjusted_p_values_bonferroni(ps).min():.3f}")
        print( f"p (Sidak)         = {adjusted_p_values_sidak(ps).min():.3f}")
        print( f"p (Holm)          = {adjusted_p_values_holm(ps).min():.3f}  (the *minimum* Bonferroni and Holm p-values are the same)")

    if details:
        return {
            "gamma3": gamma3,
            "gamma4": gamma4,
            "T": T,
            "number_of_trials": number_of_trials,
            "Var[ SR ]": math.sqrt(variance),
            "SR0 = E[ max SR ]": SR0,
            "max SR": SRs.max(),
            "max SR (raw)": np.max( X.mean(axis=0) / X.std(axis=0) ),
            "DSR": DSR,
            "1 - DSR": 1-DSR,
            "min p": ps.min(),
            "p (Bonferroni)": adjusted_p_values_bonferroni(ps).min(),
            "p (Sidak)": adjusted_p_values_sidak(ps).min(),
            "p (Holm)": adjusted_p_values_holm(ps).min()
        }

    return DSR


def adjusted_p_values_bonferroni(ps: np.ndarray) -> np.ndarray:
    """
    Adjust p-values using the Bonferroni correction (to control the FWER)

    Inputs:
    - ps: np.ndarray, p-values
    Outputs:
    - np.ndarray, adjusted p-values
    """
    M = len(ps) 
    return np.minimum( 1, M * ps )

def adjusted_p_values_sidak(ps: np.ndarray) -> np.ndarray:
    """
    Adjust p-values using the Šidák correction (to control the FWER)

    Inputs:
    - ps: np.ndarray, p-values
    Outputs:
    - np.ndarray, adjusted p-values
    """
    M = len(ps)
    return 1 - (1 - ps) ** M

def adjusted_p_values_holm(ps: np.ndarray, *, variant: str = 'bonferroni') -> np.ndarray:
    """
    Adjust p-values using the Holm correction (to control the FWER)

    Inputs:
    - ps: np.ndarray, p-values
    - variant: str, variant of the Holm correction (bonferroni or sidak)
    Outputs:
    - np.ndarray, adjusted p-values
    """
    assert variant in ['bonferroni', 'sidak']
    i = np.argsort(ps)
    M = len(ps)
    p_adjusted = np.zeros(M)
    previous = 0
    for j, idx in enumerate(i):
        if variant == 'bonferroni':
            candidate = min(1, ps[idx] * (M - j))
        else:
            candidate = 1 - (1 - ps[idx]) ** (M - j)
        p_adjusted[idx] = max(previous, candidate)
        previous = p_adjusted[idx]
    return p_adjusted



def FDR_critical_value( q, SR0, SR1, sigma0, sigma1, p_H1 ):
    """
    Given
        H ~ Bern(p₁)
        X₀ ~ N(μ₀,σ₀²)
        X₁ ~ N(μ₁,σ₁²)
    compute c such that
        P[H=0|X_H>c] = q
    
    Inputs: 
        q: float, desired FDR
        SR0, SR1: float, mean of X0 and X1
        sigma0, sigma1: float, standard deviation of X0 and X1
        p_H1: float, probability of H=1
    Returns: 
        c: float, critical value
    """

    assert SR0 < SR1
    assert 0 < q < 1
    assert 0 < p_H1 < 1
    assert 0 < sigma0
    assert 0 < sigma1

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
        warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")

        def f(c): 
            a = 1/( 
                1 + 
                scipy.stats.norm.sf( (c - SR1) / sigma1 ) / 
                scipy.stats.norm.sf( (c - SR0) / sigma0 ) * 
                p_H1 / (1-p_H1)
            )
            return np.where( np.isfinite(a), a, 0 )

        if f(-10) < q:  # Solution outside of the search interval
            return -np.inf

        return scipy.optimize.brentq( lambda c: f(c) - q, -10, 10 )


def control_for_FDR( 
    q: float, 
    *, 
    SR0: float = 0, 
    SR1: float = .5, 
    p_H1: float = .05, 
    T: int = 24, 
    gamma3: float = 0., 
    gamma4: float = 3.,
    rho: float = 0.,
    K: int = 1,
    # The other arguments are no longer used
    grid_size: int = 10_000,
    max_iterations: int = 1000,
    epsilon: float = 1e-14,
) -> tuple[float, float, float, float]:
    """
    Compute the critical value to test for multiple Sharpe ratios, while controling the false discovery rate (FDR)
    
    Return the solution (alpha, beta, SR_c) of
        alpha = q / (1-q) * (1-p_H0) / p_H0 * (1-beta)
        SR_c = SR0 + sigma_SR0 * Z_inv( 1 - alpha )
        beta = Z( (SR_c - SR1 ) / sigma_SR1 )

    We first use grid search to find an approximate solution, then do a few more iterations to have a more precise result.
    (This is inefficient, but robust.)

    Inputs:
    - q: float, FDR level
    - SR0: float, Sharpe ratio under H0
    - SR1: float, Sharpe ratio under H1
    - p_H1: float, probability that H1 is true
    - T: int, number of observations
    - gamma3: float, skewness
    - gamma4: float, (non-excess) kurtosis
    - rho: float, autocorrelation
    - grid_size: int, number of points in the grid search
    - max_iterations: int, maximum number of fixed point iterations after the grid search
    - epsilon: float, tolerance for the fixed point iterations
    - K: int, number of strategies whose Sharpe ratios we take the maximum of -- larger K means smaller variance
    Outputs:
    - alpha: float, significance level, P[SR>SR_c|H0]
    - beta: float, type II error, P[SR<=SR_c|H1]; the power is 1-beta=P[SR>SR_c|H1]
    - SR_c: float, critical value
    - q_hat: float, estimated FDR; should be close to q
    """

    Z_inv = scipy.stats.norm.ppf
    Z = scipy.stats.norm.cdf

    s0 = math.sqrt( sharpe_ratio_variance( SR0, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K ) )
    s1 = math.sqrt( sharpe_ratio_variance( SR1, T, gamma3=gamma3, gamma4=gamma4, rho=rho, K=K ) )

    if False:  # Old implementation (slow)
            
        def one_iteration( alpha, *, q, s0, s1, p_H1 = .05, T = 24 ):
            SRc = SR0 + s0 * Z_inv( 1 - alpha )
            beta = Z( (SRc - SR1 ) / s1 )
            alpha = q/(1-q) * p_H1/(1-p_H1) * (1-beta)
            return alpha, beta, SRc

        # Grid search
        xs = np.linspace( 0, 1, grid_size )[1:-1]
        ys = [ one_iteration( alpha, q=q, s0=s0, s1=s1, p_H1=p_H1, T=T )[0] for alpha in xs ]
        alpha = xs[ np.argmin( np.abs( xs - ys ) ) ]

        previous_alpha = previous_beta = previous_SRc = np.inf
        for _ in range(max_iterations):
            alpha, beta, SRc = one_iteration( alpha, q=q, s0=s0, s1=s1, p_H1=p_H1, T=T )
            error = np.abs(alpha - previous_alpha) + np.abs(beta - previous_beta) + np.abs(SRc - previous_SRc)
            if np.isnan(error) or ( error < epsilon ): 
                break
            previous_alpha = alpha
            previous_beta = beta
            previous_SRc = SRc
        assert np.isnan(error) or ( error < epsilon ), f"Error: {error:.3g} > {epsilon:.3g}"

    else: 
        SRc = FDR_critical_value( q, SR0, SR1, s0, s1, p_H1 )

    beta = Z( (SRc - SR1 ) / s1 )
    alpha = q/(1-q) * p_H1/(1-p_H1) * (1-beta)
    q_hat = 1/( 1 + (1-beta)/alpha * p_H1/(1-p_H1) )

    return alpha, beta, SRc, q_hat


def test_numeric_example():

    for rho in [0, 0.2]: 
        print( "----------" )
        mu     = .036
        sigma  = .079
        T      = 24
        gamma3 = -2.448
        gamma4 = 10.164
        SR0    = 0
        SR1    = .5
        p_H1   = .10
        alpha  = .10
        SR = mu / sigma
        print( f"SR0                    = {SR0:.3f}" )
        print( f"SR1                    = {SR1:.3f}" )
        print( f"μ                      = {mu:.3f}" )
        print( f"σ                      = {sigma:.3f}" )
        print( f"γ3                     = {gamma3:.3f}" )
        print( f"γ4                     = {gamma4:.3f}" )
        print( f"ρ                      = {rho:.3f}" )
        print( f"T                      = {T}" )
        print( f"SR                     = {SR:.3f}" )
        print( f"σ_SR                   = {math.sqrt( sharpe_ratio_variance( SR = mu / sigma, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T ) ):.3f} (non-Gaussian)" )
        print( f"σ_SR                   = {math.sqrt( sharpe_ratio_variance( SR = mu / sigma, gamma3 = 0,      gamma4 = 3,      T = T ) ):.3f} (Gaussian, iid)" )
        print( f"MinTRL                 = {minimum_track_record_length( SR = mu / sigma, SR0 = 0, gamma3 = gamma3, gamma4 = gamma4, rho = rho, alpha = alpha ):.3f}" )
        print( f"MinTRL(SR0=.1)         = {minimum_track_record_length( SR = mu / sigma, SR0 = .1, gamma3 = gamma3, gamma4 = gamma4, rho = rho, alpha = alpha ):.3f}" )
        print( f"p = 1 - PSR(SR0=0)     = {1-probabilistic_sharpe_ratio( SR = mu / sigma, SR0 = 0,  T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho):.3f}" )
        print( f"PSR(SR0=0)             = {  probabilistic_sharpe_ratio( SR = mu / sigma, SR0 = 0,  T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho):.3f}" )
        print( f"PSR(SR0=.1)            = {probabilistic_sharpe_ratio( SR = mu / sigma, SR0 = .1, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho):.3f}" )
        print( f"SR0                    = {SR0:.3f}" )
        print( f"SR_c                   = {critical_sharpe_ratio(SR0, T, gamma3=0.,     gamma4=3.,     rho = 0,   alpha=alpha):.3f} (Gaussian, iid)" )
        print( f"SR_c                   = {critical_sharpe_ratio(SR0, T, gamma3=gamma3, gamma4=gamma4, rho = rho, alpha=alpha):.3f} (non-Gaussian -- unchanged if iid, SR0=0)" )
        print( f"SR1                    = {SR1:.3f}" )
        print( f"Power = 1 - β          = {sharpe_ratio_power( SR0=SR0, SR1 = SR1, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho, alpha = alpha ):.3f}" )
        print( f"β                      = {1-sharpe_ratio_power( SR0=SR0, SR1 = SR1, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho, alpha = alpha ):.3f}" )
        print( f"P[H1]                  = {p_H1:.3f}" )
        print( f"pFDR = P[H0|SR>SR_c]   = {pFDR( p_H1, alpha, 1 - sharpe_ratio_power( SR0=SR0, SR1 = SR1, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho, alpha = alpha ) ):.3f}" )
        print( f"oFDR = P[H0|SR>SR_obs] = {oFDR( SR = mu / sigma, SR0=SR0, SR1=SR1, T=T, p_H1=p_H1, gamma3 = gamma3, gamma4 = gamma4, rho = rho ):.3f}" )

        print( "\nFWER" )
        number_of_trials = 10
        variance = .1
        E_max_SR = expected_maximum_sharpe_ratio( number_of_trials, variance )   
        SR0_adj = SR0 + E_max_SR
        SR1_adj = SR1 + E_max_SR
        sigma_SR0_adj_single = math.sqrt( sharpe_ratio_variance( SR = SR0_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T ) )
        sigma_SR0_adj =  math.sqrt( sharpe_ratio_variance( SR = SR0_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T, K = number_of_trials ) )
        sigma_SR1_adj_single = math.sqrt( sharpe_ratio_variance( SR = SR1_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T ) )
        sigma_SR1_adj =  math.sqrt( sharpe_ratio_variance( SR = SR1 + SR0_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T, K = number_of_trials ) )
        DSR_single = probabilistic_sharpe_ratio(SR, SR0 = SR0_adj, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho )
        DSR        = probabilistic_sharpe_ratio(SR, SR0 = SR0_adj, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho, K = number_of_trials )
        print( f"K                         = {number_of_trials}" )
        print( f"Var[SR_k]                 = {variance:.3f}  (only used to compute E[max SR])" )
        print( f"E[max SR]                 = {E_max_SR:.3f}" )
        print( f"SR0_adj = SR0 + E[max SR] = {SR0_adj:.3f}" )
        print( f"SR1_adj = SR1 + E[max SR] = {SR1_adj:.3f}" )
        print( f"σ_SR0_adj                 = {sigma_SR0_adj:.3f}" )
        print( f"σ_SR1_adj                 = {sigma_SR1_adj:.3f}" )
        print( f"DSR                       = {DSR:.3f}" )
        print( f"oFDR = P[H0|SR>SR_obs]    = {oFDR( SR = mu / sigma, SR0=SR0_adj, SR1=SR1+SR0_adj, T=T, p_H1=p_H1, gamma3 = gamma3, gamma4 = gamma4, rho = rho, K = number_of_trials ):.3f}" )

        print( f"σ_SR0_adj (K=1)              = {sigma_SR0_adj_single:.3f}" )
        print( f"σ_SR1_adj (K=1)              = {sigma_SR1_adj_single:.3f}" )
        print( f"DSR (K=1)                    = {DSR_single:.3f}" )
        print( f"oFDR = P[H0|SR>SR_obs] (K=1) = {oFDR( SR = mu / sigma, SR0=SR0_adj, SR1=SR1+SR0_adj, T=T, p_H1=p_H1, gamma3 = gamma3, gamma4 = gamma4, rho = rho ):.3f}" )

        print( "\nFDR" )
        q    = .25
        alpha_, beta_, SR_c, q_hat = control_for_FDR( q, SR0 = SR0, SR1 = SR1, p_H1 = p_H1, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho )
        print( f"P[H1]                  = {p_H1:.3f}" )
        print( f"q                      = {q:.3f}" )
        print( f"α                      = {alpha_:.4f}" )
        print( f"β                      = {beta_:.3f}" )
        print( f"SR_c                   = {SR_c:.3f}" )
        print( f"σ_SR0                  = {math.sqrt( sharpe_ratio_variance( SR = SR0, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T ) ):.3f}" )
        print( f"σ_SR1                  = {math.sqrt( sharpe_ratio_variance( SR = SR1, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T ) ):.3f}" )

        print( "\nFWER-FDR" )
        alpha_W, beta_W, SR_c_W, q_hat_W = control_for_FDR( q, SR0 = SR0+SR0_adj, SR1 = SR1+SR0_adj, p_H1 = p_H1, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho )
        alpha_, beta_, SR_c, q_hat = control_for_FDR( q, SR0 = SR0+SR0_adj, SR1 = SR1+SR0_adj, p_H1 = p_H1, T = T, gamma3 = gamma3, gamma4 = gamma4, rho = rho, K = number_of_trials )
        print( f"σ_SR0                  = {math.sqrt( sharpe_ratio_variance( SR = SR0+SR0_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T, K = number_of_trials ) ):.3f}" )
        print( f"σ_SR1                  = {math.sqrt( sharpe_ratio_variance( SR = SR1+SR0_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T, K = number_of_trials ) ):.3f}" )
        print( f"α                      = {alpha_:.5f}" )
        print( f"β                      = {beta_:.3f}" )
        print( f"SR_c                   = {SR_c:.3f}" )
        
        print( f"σ_SR0 (K=1)          = {math.sqrt( sharpe_ratio_variance( SR = SR0+SR0_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T ) ):.3f}" )
        print( f"σ_SR1 (K=1)          = {math.sqrt( sharpe_ratio_variance( SR = SR1+SR0_adj, gamma3 = gamma3, gamma4 = gamma4, rho = rho, T = T ) ):.3f}" )
        print( f"α (K=1)              = {alpha_W:.5f}" )
        print( f"β (K=1)              = {beta_W:.3f}" )
        print( f"SR_c (K=1)           = {SR_c_W:.3f}" )


if __name__ == '__main__':
    test_effective_rank()
    test_sharpe_ratio_variance()
    test_minimum_track_record_length()
    test_probabilistic_sharpe_ratio()
    test_sharpe_ratio_power()
    test_pFDR()
    test_oFDR()
    test_robust_covariance_inverse()
    test_minimum_variance_weights_for_correlated_assets()
    test_numeric_example()
    print("All tests passed.")
