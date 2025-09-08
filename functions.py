from math import sqrt
import math
import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


def significance_stars(p: float) -> str:
    """
    Significance stars for a p-value
    """
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else '.' if p < .1 else ''


def round_p_value(p: float) -> str:
    """
    Round a p-value to 2 decimal places if above 0.01, 3 if above 0.001, and zero if below.
    """
    if p > .01:
        return f"{p:.2f}"
    if p > .001:
        return f"{p:.3f}"
    return f"10^{int(math.log10(p))}"


def effective_rank(C):
    """
    Effective rank of a correlation matrix.

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
        retries: int = 10,
        max_clusters: int = 100,
        plot: bool = False,
) -> tuple[int, pd.Series, np.ndarray]:
    """
    Compute the optimal number of clusters, from a correlation matrix

    Algorithm in section 8.1 of [1], without the third point (i.e., no recursive re-clustering of low-quality clusters)

    [1] Detection of false investment strategies using unsupervised learning methods
        M. Lopez de Prado (2018)
        https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167017
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
            kmeans.fit(D)
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
    number_of_trials = 100,
    effective_number_of_trials = 10,
    number_of_observations = 200,
    noise = .1,
):
    """
    Generate a correlation matrix with a block structure

    Returns the correlation matrix, the return matrix, and the cluster labels.
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


def generate_non_gaussian_data( nr, nc, *, SR0 = 0, name = 'severe', seed = None ):

    configs = {
        "gaussian": (0,    0,     0.015, 0.010),
        "mild":     (0.04, -0.03, 0.015, 0.010),
        "moderate": (0.03, -0.045, 0.020, 0.010),
        "severe":   (0.02, -0.060, 0.025, 0.010),
    }

    def mixture_variance(p_tail, mu_tail, sigma_tail, mu_core, sigma_core):
        w = 1.0 - p_tail
        mu = w*mu_core + p_tail*mu_tail
        m2 = w*(sigma_core**2 + mu_core**2) + p_tail*(sigma_tail**2 + mu_tail**2)
        return m2 - mu**2

    def gen_with_true_SR0(reps, T, cfg, SR0, seed):
        p, mu_tail, sig_tail, sig_core = cfg
        # Zero-mean baseline mixture (choose mu_core so mean=0)
        mu_core0 = - p*mu_tail/(1.0 - p)
        std0 = np.sqrt(mixture_variance(p, mu_tail, sig_tail, mu_core0, sig_core))
        mu_shift = SR0 * std0  # sets population Sharpe to SR0, preserves skew/kurt
        rng = np.random.default_rng(seed)
        mask = rng.random((reps, T)) < p
        X = rng.normal(mu_core0 + mu_shift, sig_core, size=(reps, T))
        X[mask] = rng.normal(mu_tail + mu_shift, sig_tail, size=mask.sum())
        return X

    return gen_with_true_SR0( nr, nc, configs[name], SR0, seed )


def sharpe_ratio_variance( SR, T, *, gamma3=0., gamma4=3. ):
    """
    Asymptotic variance of the Sharpe ratio
    """
    return ( 1 - gamma3 * SR + (gamma4-1)/4 * SR**2 ) / T

def test_sharpe_ratio_variance():
    assert round( sqrt( sharpe_ratio_variance( SR = .036 / .079, gamma3 = -2.448, gamma4 = 10.164, T = 24 ) ), 3 ) == .329
    assert round( sqrt( sharpe_ratio_variance( SR = .036 / .079, gamma3 = 0,      gamma4 = 3,      T = 24 ) ), 3 ) == .214


def minimum_track_record_length( SR, SR0, *, gamma3=0., gamma4=3., alpha=0.05 ):
    """
    Minimum track record length for the Sharpe ratio to be significantly greater than SR0, at the confidence level alpha.
    """
    return ( 1 - gamma3 * SR0 + (gamma4-1)/4*SR0**2 ) * ( norm.ppf(1-alpha) / (SR - SR0) ) ** 2

def test_minimum_track_record_length():
    assert round( minimum_track_record_length( SR = .036 / .079, SR0 = 0, gamma3 = -2.448, gamma4 = 10.164, alpha = .05 ), 3 ) == 13.029


def probabilistic_sharpe_ratio( SR, SR0, T, *, gamma3=0., gamma4=3. ):
    """
    This is 1-p, where p is the p-value of the test  H0: SR=SR0  vs  H1: SR>SR0.
    """
    variance = sharpe_ratio_variance( SR0, T, gamma3 = gamma3, gamma4 = gamma4 )
    return norm.cdf( (SR - SR0) / sqrt(variance) )

def test_probabilistic_sharpe_ratio():
    assert round( probabilistic_sharpe_ratio( SR = .036 / .079, SR0 = 0,  T = 24, gamma3 = -2.448, gamma4 = 10.164), 3 ) == .987
    assert round( probabilistic_sharpe_ratio( SR = .036 / .079, SR0 = .1, T = 24, gamma3 = -2.448, gamma4 = 10.164), 3 ) == .939


def critical_sharpe_ratio(SR0, T, *, gamma3=0., gamma4=3., alpha=0.05):
    """
    Critical value for the test  H0: SR=SR0  vs  H1: SR>SR0.
    """
    variance = sharpe_ratio_variance(SR0, T, gamma3=gamma3, gamma4=gamma4)
    return SR0 + norm.ppf(1 - alpha) * sqrt(variance)


def sharpe_ratio_power( SR0, SR1, T, *, gamma3=0., gamma4=3., alpha=0.05):  # Needs SR1, e.g., average Sharpe ratio of strategies with positive excess returns
    """
    Power (1-β) of the test  H0: SR=SR0  vs  H1: SR=SR1.

    Note that "power" is the same thing as "recall" in classification:
      Power = P[ reject H0 | H1 ]
            = P[ H1 ∧ reject H0 | H1 ]
            = TP / (TP + FN )
            = recall
    """
    critical_SR = critical_sharpe_ratio(SR0, T, gamma3=gamma3, gamma4=gamma4, alpha=alpha)
    variance = sharpe_ratio_variance(SR1, T, gamma3=gamma3, gamma4=gamma4)
    beta = norm.cdf((critical_SR - SR1) / sqrt(variance))
    return 1 - beta

def test_sharpe_ratio_power():
    assert round( 1 - sharpe_ratio_power( SR0=0, SR1 = .5, T = 24, gamma3 = -2.448, gamma4 = 10.164 ), 3 ) == .315


def bayesian_fdr( p_H1, alpha, beta ):  # Needs beta=1-power (which needs, SR1), and p_H1, the proportion of deployed profitable strategies
    p_H0 = 1 - p_H1
    return 1 / ( 1 + (1-beta) * p_H1 / alpha / p_H0 )

def test_bayesian_fdr():
    assert round( bayesian_fdr( .05, .05, .315 ), 3 ) == .581


def posterior_p_value( SR, SR0, SR1, T, p_H1, *, gamma3=0., gamma4=3.):
    p0 = 1 - probabilistic_sharpe_ratio( SR, SR0, T, gamma3=gamma3, gamma4=gamma4 )
    p1 = 1 - probabilistic_sharpe_ratio( SR, SR1, T, gamma3=gamma3, gamma4=gamma4 )
    p_H0 = 1 - p_H1
    return p0 * p_H0 / ( p0 * p_H0 + p1 * p_H1 )

def test_posterior_p_value():
    assert round( posterior_p_value( SR = .036 / .079, SR0=0, SR1=.5, T=24, p_H1=.05, gamma3 = -2.448, gamma4 = 10.164 ), 3 ) == .306


def robust_covariance_inverse( V ):
    r"""
    Sherman–Morrison formula
    Assume $V = \rho \sigma \sigma' + (1-\rho) \text{diag}(\sigma^2)$ (variance matrix, with constant correlations).
    Its inverse is $V^{-1} = A^{-1} - \dfrac{ A^{01} \rho \sigma \sigma' A^{-1} }{ 1 + \rho \sigma' A^{-1} \sigma }$.
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


def minimum_variance_weights_for_correlated_assets(V):
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


def variance_of_the_clustered_trials(X, clusters):
    """
    Inputs:
    - X: numpy array or returns, one column per strategy
    - cluster: cluster assignment, list (or array) with one element per startegy
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


def expected_maximum_sharpe_ratio( number_of_trials, variance ):
    return (
        sqrt( variance ) * (
            ( 1 - np.euler_gamma ) * norm.ppf( 1 - 1 / number_of_trials ) +
            np.euler_gamma * norm.ppf( 1 - 1 / number_of_trials / np.exp(1) )
        )
    )

def deflated_sharpe_ratio(X, *, verbose = False, cluster = True, details = False):
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


def adjusted_p_values_bonferroni(ps):
    M = len(ps)
    return np.minimum( 1, M * ps )

def adjusted_p_values_sidak(ps):
    M = len(ps)
    return 1 - (1 - ps) ** M

def adjusted_p_values_holm(ps, *, variant = 'bonferroni'):
    assert variant in ['bonferroni', 'sidak']
    i = np.argsort(ps)
    M = len(ps)
    p_adjusted = np.zeros(M)
    previous = 0
    for j, idx in enumerate(i):
        if variant == 'bonferroni':
            candidate = min(1, ps[idx] * (M - j))
        else:
            candidate = 1 - (1 - ps[idx]) ** M
        p_adjusted[idx] = max(previous, candidate)
        previous = p_adjusted[idx]
    return p_adjusted


def control_for_FDR( q, *, SR0 = 0, SR1 = .5, p_H1 = .05, T = 24, gamma3 = 0., gamma4 = 3. ):
    """
    Return the solution (alpha, beta, SR_c) of
        alpha = q / (1-q) * (1-p_H0) / p_H0 * (1-beta)
        SR_c = SR0 + sigma_SR0 * Z_inv( 1 - alpha )
        beta = Z( (SR_c - SR1 ) / sigma_SR1 )

    We first use grid search to find an approximate solution, then do a few more iterations to have a more precise result.
    (This is inefficient, but robust.)
    """

    Z_inv = scipy.stats.norm.ppf
    Z = scipy.stats.norm.cdf

    def one_iteration( alpha, *, q, SR0 = 0, SR1 = .5, p_H1 = .05, T = 24, gamma3 = 0., gamma4 = 3. ):
        SRc = SR0 + math.sqrt( sharpe_ratio_variance( SR0, T, gamma3=gamma3, gamma4=gamma4 ) ) * Z_inv( 1 - alpha )
        beta = Z( (SRc - SR1 ) / math.sqrt( sharpe_ratio_variance( SR1, T, gamma3=gamma3, gamma4=gamma4 ) ) )
        alpha = q / ( 1 - q ) * p_H1 / (1-p_H1) * ( 1 - beta )
        return alpha, beta, SRc

    # Grid search
    xs = np.linspace( 0, 1, 10_000 )[1:-1]
    ys = [ one_iteration( alpha, q=q, SR0=SR0, SR1=SR1, p_H1=p_H1, T=T, gamma3=gamma3, gamma4=gamma4 )[0] for alpha in xs ]
    alpha = xs[ np.argmin( np.abs( xs - ys ) ) ]

    for _ in range(100):
        alpha, beta, SRc = one_iteration( alpha, q=q, SR0=SR0, SR1=SR1, p_H1=p_H1, T=T, gamma3=gamma3, gamma4=gamma4 )

    return alpha, beta, SRc


def test_numeric_example():
    mu     = .036
    sigma  = .079
    T      = 24
    gamma3 = -2.448
    gamma4 = 10.164
    SR0    = 0
    SR1    = .5
    p_H1   = .05
    alpha  = .05

    SR = mu / sigma
    print( f"μ                      = {mu:.3f}" )
    print( f"σ                      = {sigma:.3f}" )
    print( f"γ3                     = {gamma3:.3f}" )
    print( f"γ4                     = {gamma4:.3f}" )
    print( f"T                      = {T}" )
    print( f"SR                     = {SR:.3f}" )
    print( f"σ_SR                   = {sqrt( sharpe_ratio_variance( SR = mu / sigma, gamma3 = gamma3, gamma4 = gamma4, T = T ) ):.3f} (non-Gaussian)" )
    print( f"σ_SR                   = {sqrt( sharpe_ratio_variance( SR = mu / sigma, gamma3 = 0,      gamma4 = 3,      T = T ) ):.3f} (Gaussian)" )
    print( f"MinTRL                 = {minimum_track_record_length( SR = mu / sigma, SR0 = 0, gamma3 = gamma3, gamma4 = gamma4, alpha = alpha ):.3f}" )
    print( f"MinTRL(SR0=.1)         = {minimum_track_record_length( SR = mu / sigma, SR0 = .1, gamma3 = gamma3, gamma4 = gamma4, alpha = alpha ):.3f}" )
    print( f"PSR(SR0=0)             = {probabilistic_sharpe_ratio( SR = mu / sigma, SR0 = 0,  T = T, gamma3 = gamma3, gamma4 = gamma4):.3f}" )
    print( f"PSR(SR0=.1)            = {probabilistic_sharpe_ratio( SR = mu / sigma, SR0 = .1, T = T, gamma3 = gamma3, gamma4 = gamma4):.3f}" )
    print( f"SR0                    = {SR0:.3f}" )
    print( f"SR_c                   = {critical_sharpe_ratio(SR0, T, gamma3=0.,     gamma4=3.,     alpha=alpha):.3f} (Gaussian)" )
    print( f"SR_c                   = {critical_sharpe_ratio(SR0, T, gamma3=gamma3, gamma4=gamma4, alpha=alpha):.3f} (non-Gaussian -- unchanged if SR0=0)" )
    print( f"SR1                    = {SR1:.3f}" )
    print( f"Power = 1 - β          = {sharpe_ratio_power( SR0=SR0, SR1 = SR1, T = T, gamma3 = gamma3, gamma4 = gamma4, alpha = alpha ):.3f}" )
    print( f"β                      = {1-sharpe_ratio_power( SR0=SR0, SR1 = SR1, T = T, gamma3 = gamma3, gamma4 = gamma4, alpha = alpha ):.3f}" )
    print( f"P[H1]                  = {p_H1:.3f}" )
    print( f"pFDR = P[H0|SR>SR_c]   = {bayesian_fdr( p_H1, alpha, 1 - sharpe_ratio_power( SR0=SR0, SR1 = SR1, T = T, gamma3 = gamma3, gamma4 = gamma4, alpha = alpha ) ):.3f}" )
    print( f"oFDR = P[H0|SR>SR_obs] = {posterior_p_value( SR = mu / sigma, SR0=SR0, SR1=SR1, T=T, p_H1=p_H1, gamma3 = gamma3, gamma4 = gamma4 ):.3f}" )

    print( "\nFWER" )
    number_of_trials = 10
    variance = .1
    SR0_adj = expected_maximum_sharpe_ratio( number_of_trials, variance )
    sigma_SR0_adj = sqrt( sharpe_ratio_variance( SR = SR0_adj, gamma3 = gamma3, gamma4 = gamma4, T = T ) )
    DSR = probabilistic_sharpe_ratio(SR, SR0 = SR0_adj, T = T, gamma3 = gamma3, gamma4 = gamma4 )
    print( f"K                      = {number_of_trials}" )
    print( f"Var[ SR ]              = {variance:.3f}" )
    print( f"SR0_adj = E[max SR]    = {SR0_adj:.3f}" )
    print( f"σ_SR0_adj              = {sigma_SR0_adj:.3f}" )
    print( f"DSR                    = {DSR:.3f}" )
    print( f"SR1_adj                = {SR1 + SR0_adj:.3f}" )
    print( f"σ_SR1_adj              = {sqrt( sharpe_ratio_variance( SR = SR1 + SR0_adj, gamma3 = gamma3, gamma4 = gamma4, T = T ) ):.3f}" )
    print( f"oFDR = P[H0|SR>SR_obs] = {posterior_p_value( SR = mu / sigma, SR0=SR0_adj, SR1=SR1+SR0_adj, T=T, p_H1=p_H1, gamma3 = gamma3, gamma4 = gamma4 ):.3f}" )

    print( "\nFDR" )
    q    = .25
    p_H1 = .10  # Was 5% in the previous examples
    alpha_, beta_, SR_c = control_for_FDR( q, SR0 = SR0, SR1 = SR1, p_H1 = p_H1, T = T, gamma3 = gamma3, gamma4 = gamma4 )
    print( f"P[H1]                  = {p_H1:.3f}" )
    print( f"q                      = {q:.3f}" )
    print( f"α                      = {alpha_:.4f}" )
    print( f"β                      = {beta_:.3f}" )
    print( f"SR_c                   = {SR_c:.3f}" )
    print( f"σ_SR0                  = {sqrt( sharpe_ratio_variance( SR = SR0, gamma3 = gamma3, gamma4 = gamma4, T = T ) ):.3f}" )
    print( f"σ_SR1                  = {sqrt( sharpe_ratio_variance( SR = SR1, gamma3 = gamma3, gamma4 = gamma4, T = T ) ):.3f}" )

    print( "\nFWER-FDR" )
    alpha_, beta_, SR_c = control_for_FDR( q, SR0 = SR0+SR0_adj, SR1 = SR1+SR0_adj, p_H1 = p_H1, T = T, gamma3 = gamma3, gamma4 = gamma4 )
    print( f"σ_SR0                  = {sqrt( sharpe_ratio_variance( SR = SR0+SR0_adj, gamma3 = gamma3, gamma4 = gamma4, T = T ) ):.3f}" )
    print( f"σ_SR1                  = {sqrt( sharpe_ratio_variance( SR = SR1+SR0_adj, gamma3 = gamma3, gamma4 = gamma4, T = T ) ):.3f}" )
    print( f"α                      = {alpha_:.5f}" )
    print( f"β                      = {beta_:.3f}" )
    print( f"SR_c                   = {SR_c:.3f}" )


if __name__ == '__main__':
    test_effective_rank()
    test_sharpe_ratio_variance()
    test_minimum_track_record_length()
    test_probabilistic_sharpe_ratio()
    test_sharpe_ratio_power()
    test_bayesian_fdr()
    test_posterior_p_value()
    test_robust_covariance_inverse()
    test_minimum_variance_weights_for_correlated_assets()
    test_numeric_example()
    print("All tests passed.")
