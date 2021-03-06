# import math
import numpy as np
import jax.numpy as jnp
from scipy.stats import norm
import jax
# from scipy.special._ufuncs import gammainc
# from scipy.special._ufuncs import gamma
# from scipy.special._ufuncs import iv

from jax.scipy.special import gammainc
from scipy.special._ufuncs import gamma
# from jax.scipy.special import iv

from scipy.stats import gaussian_kde
from scipy.optimize import newton
from scipy.optimize import minimize
from scipy.stats import ncx2
# from matplotlib import pyplot

r"""Unbiased SABR model simulation in the manner of Bin Chen, Cornelis W. Oosterlee and Hans van der Weide (2011).
The Stochastic Alpha Beta Rho model first designed by Hagan & al. is a very popular model use extensively by practitioners
for interest rates derivatives. In this framework, volatility is stochastic, following a geometric brownian motion
with no drift, whereas the forward rate dynamics are modeled with a CEV process.However, despite the simplicity of its formulation, 
it does not allow for closed form analytical solutions. 
Moreover as pointed by early authors Andersen (1995) and Andersen & Andreasen (2000) Euler-Maruyama and Milstein discretization scheme 
are biased for the CEV process, and monte carlo simulations will exhibit significant bias even with a high number of simulated paths.
Chen & al. (2011) extend the methodologies of Willard (1997), Broadie & Kaya (2006), Andersen (2008)and  Islah (2009) to provide an unbiased
scheme to simulate and discretize the SABR process. This method is a mix of  multiple techniques :a direct inversion scheme of the non central
 chi-squared distribution, the QE method of andersen and small disturbance expansion.   
The implementation I have provided, tries to vectorize the problem as much as possible, but some amount of iteration is required when dealing
with the conditional application of the QE scheme or direct inversion. It also does not implement the so-called "Enhanced direct inversion procedure"
of formula (3.12). I leave this for a later time.
References
----------
 * "Efficient Techniques for Simulation of Interest Rate Models Involving Non-Linear Stochastic Differential Equations"
   Leif B. G. Andersen (1995)
 * "Volatility skews and extensions of the libor market model"
   L. Andersen, J. Andreasen (2000)
 * "Managing Smile Risk",
   Patrick S. Hagan, Deep Kumar, Andrew S. Lesniewski,and Diana E. Woodward (2002)
 * "Efficient simulation of the heston stochastic volatility model"
   Andersen L. Journal of Computational Finance 11:3 (2008) 1???22.
 * "Simulation of the CEV process and the local martingale property."
    A. E. Lindsay, D. R. Brecher (2010)
 * "Efficient unbiased simulation scheme for the SABR stochastic volatility model"
       Bin Chen, Cornelis W. Oosterl, Hans van der Weide (2011)
"""

__author__ = 'Lionel Ouaknin'
__credit__ = 'Bin Chen, Cornelis W. Oosterlee, Hans van der Weide and Lionel Ouaknin'
__status__ = 'beta'
__version__ = '0.1.0'


######################## direct inversion ######################################
def root_chi2(u, b, a, t):
    ''' inversion of the non central chi-square distribution '''
    x0 = a
    # root = newton(func, x0, args=(u, b, a))
    # return root * t
    res = minimize(func, x0, args=(u, b, a))
    return res.x[0] * t



def func(x, u, b, a):  # page 13
    return jnp.abs(0.5 + 0.5 * jnp.sign(x) * ncx2.cdf(jnp.abs(x), 2 - b, a)
                  - 0.5 * jnp.sign(a) * ncx2.cdf(a, b, jnp.abs(x)) - u)


# def fprime(x, u, b, a):
#     x_ = np.abs(x)
#     return 0.25 * np.sign(x) * np.exp(-(a+x_)/2) * (x_/a) ** (-b/4) * iv(-b/2, np.sqrt(a*x_)) \
#            - 0.25 * np.sign(a) * np.exp(-(a+x_)/2) * (a/x_) ** ((b-2)/4) * iv(b/2-1, np.sqrt(a*x_))


######################## Absorption probability #################################

def AbsorptionConditionalProb(a, b):
    ''' probability that F_ti+1 is absorbed by the 0 barrier conditional on inital value S0  '''
    cprob = 1. - gammainc(b / 2, a / 2)  # formula (2.10), scipy gammainc is already normalized by gamma(b)
    return cprob


######################## volatility GBM simulation ##############################

def simulate_Wt(dW, T, N):
    ''' Simulates brownian paths. Vectorization inspired by Scipy cookbook '''
    return jnp.cumsum(dW, axis=0)


def simulate_sigma(Wt, sigma0, alpha, t):
    ''' 'Exact' simulation of GBM with mu=0 '''
    return sigma0 * jnp.exp(alpha * Wt - 0.5 * (alpha ** 2) * t[1:])


######################## integrated variance ####################################
def integrated_variance_small_disturbances(N, rho, alpha, sigmat, dt, dW, U):
    ''' Small disturbance expansion Chen B. & al (2011).'''
    # formula (3.18)
    dW_2, dW_3, dW_4 = jnp.power(dW, 2), jnp.power(dW, 3), jnp.power(dW, 4)

    m1 = alpha * dW
    m2 = (1. / 3) * (alpha ** 2) * (2 * dW_2 - dt / 2)
    m3 = (1. / 3) * (alpha ** 3) * (dW_3 - dW * dt)
    m4 = (1. / 5) * (alpha ** 4) * ((2. / 3) * dW_4 - (3. / 2) * dW_2 * dt + 2 * jnp.power(dt, 2))
    m = (sigmat ** 2) * dt * (1. + m1 + m2 + m3 + m4)

    v = (1. / 3) * (sigmat ** 4) * (alpha ** 2) * jnp.power(dt, 3)
    # step 3 & 4 of 3.6 discretization scheme
    mu = jnp.log(m) - (1. / 2) * jnp.log(1. + v / m ** 2)
    sigma2 = jnp.log(1. + v / (m ** 2))
    A_t = jnp.exp(jnp.sqrt(sigma2) * norm.ppf(U) + mu)
    v_t = (1. - rho ** 2) * A_t
    return v_t


def integrated_variance_trapezoidal(rho, sigma_t, dt):
    sigma2_ti = sigma_t ** 2
    sigma2_ti_1 = shift(sigma_t, -1, fill_value=0.) ** 2
    A_t = ((dt / 2) * (sigma2_ti + sigma2_ti_1))
    v_t = (1. - rho ** 2) * A_t
    return v_t


def shift(arr, num, fill_value=jnp.nan):
    arr = jnp.roll(arr, num)
    if num < 0:
        arr[num:] = fill_value
    elif num > 0:
        arr[:num] = fill_value
    return arr


def andersen_QE(ai, b):
    ''' Test for Andersen L. (2008) Quadratic exponential Scheme (Q.E.) '''
    k = 2. - b
    lbda = ai
    s2 = (2 * (k + 2 * lbda))
    m = k + lbda
    psi = s2 / m ** 2
    return m, psi

def compute_Ft(Ft_1, sigma2dt, vt_1, beta, rho, alpha, sigma_t, sigma_t_1, b, psi_threshold, zt_1, Ut_1):
    if Ft_1 == jnp.nan:
        return jnp.nan

    if Ft_1 == 0.:
        return 0.
    
    a = (1. / vt_1) * (((jnp.abs(Ft_1) ** (1. - beta)) / (1. - beta) + (rho / alpha) * (
            sigma_t - sigma_t_1)) ** 2)

    x = a * sigma2dt

    sign = jnp.sign(Ft_1)
    # first moment
    m1 = (x + (2 - b) * sigma2dt) * gammainc(b / 2, a / 2) + x * jnp.exp(-a / 2) * (
                a / 2 ** (b / 2 - 1)) / gamma(b / 2)
    # # second moment
    m2 = x ** 2 + 4 * (2 - b / 2) * sigma2dt * x + 4 * (1 - b / 2) * (2 - b / 2) * sigma2dt ** 2 - m1 ** 2

    m, psi = m1, m2 / m1 ** 2

    if m >= 0 and psi_threshold >= psi > 0:
        # Formula 3.9: simulation for high values
        e2 = (2. / psi) - 1. + jnp.sqrt(2. / psi) * jnp.sqrt((2. / psi) - 1.)
        d = m / (1. + e2)
        x_t_next = d * ((jnp.sqrt(e2) + zt_1) ** 2)
        return sign * jnp.power(((1. - beta) ** 2) * x_t_next, 1 / (2. * (1. - beta)))

    else:
        # direct inversion for small values
        x_t_next = root_chi2(Ut_1, b, a, sigma2dt) * sign
        return jnp.sign(x_t_next) * jnp.power(((1. - beta) ** 2) * jnp.abs(x_t_next),
                                                 1 / (2. * (1. - beta)))
    return 0

def compute_Ft_np(Ft_1, sigma2dt, vt_1, beta, rho, alpha, sigma_t, sigma_t_1, b, psi_threshold, zt_1, Ut_1):
    if Ft_1 == np.nan:
        return np.nan

    if Ft_1 == 0.:
        return 0.
    
    a = (1. / vt_1) * (((np.abs(Ft_1) ** (1. - beta)) / (1. - beta) + (rho / alpha) * (
            sigma_t - sigma_t_1)) ** 2)

    x = a * sigma2dt

    sign = np.sign(Ft_1)
    # first moment
    m1 = (x + (2 - b) * sigma2dt) * gammainc(b / 2, a / 2) + x * np.exp(-a / 2) * (
                a / 2 ** (b / 2 - 1)) / gamma(b / 2)
    # # second moment
    m2 = x ** 2 + 4 * (2 - b / 2) * sigma2dt * x + 4 * (1 - b / 2) * (2 - b / 2) * sigma2dt ** 2 - m1 ** 2

    m, psi = m1, m2 / m1 ** 2

    if m >= 0 and psi_threshold >= psi > 0:
        # Formula 3.9: simulation for high values
        e2 = (2. / psi) - 1. + np.sqrt(2. / psi) * np.sqrt((2. / psi) - 1.)
        d = m / (1. + e2)
        x_t_next = d * ((np.sqrt(e2) + zt_1) ** 2)
        return sign * jnp.power(((1. - beta) ** 2) * x_t_next, 1 / (2. * (1. - beta)))

    else:
        # direct inversion for small values
        x_t_next = root_chi2(Ut_1, b, a, sigma2dt) * sign
        return np.sign(x_t_next) * jnp.power(((1. - beta) ** 2) * jnp.abs(x_t_next),
                                                 1 / (2. * (1. - beta)))
    return 0

def sabrMC(F0=1, sigma0=0.25, alpha=0.001, beta=0.999, rho=0.001, psi_threshold=2., n_years=1, T=100000, N=100000,
           trapezoidal_integrated_variance=False):
    # print(F0)
    """Simulates a SABR process with absoption at 0 with the given parameters.
       The Sigma, Alpha, Beta, Rho (SABR) model originates from Hagan S. et al. (2002).
       The simulation algorithm is taken from Chen B., Osterlee C. W. and van der Weide H. (2011)
       Parameters
       ----------
       F0: Underlying (most often a forward rate) initial value

       sigma0: Initial stochastic volatility
       alpha: Vol-vol parameter of SABR

       beta: Beta parameter of SABR

       rho: Stochastic process correlation

       psi_threshold: Refers to the threshold of applicability of Andersen L. (2008)
           Quadratic Exponential (QE) algorithm.

       n_years: Number of year fraction for the simulation

       T: Number of steps

       N: Number of simulated paths

       trapezoidal_integrated_variance: use trapezoidal integrated variance instead of small disturbances
       Returns
       -------
       Ft: type numpy.ndarray, shape (T+1, N)
           An array with each path stored in a column.
       Reference
       ---------
       * "Managing Smile Risk",
       Patrick S. Hagan, Deep Kumar, Andrew S. Lesniewski,and Diana E. Woodward (2002)
       * "Efficient simulation of the heston stochastic volatility model"
       Andersen L. Journal of Computational Finance 11:3 (2008) 1???22.
       * "Simulation of the CEV process and the local martingale property."
       A. E. Lindsay, D. R. Brecher (2010)
       * "Efficient unbiased simulation scheme for the SABR stochastic volatility model"
       Bin Chen, Cornelis W. Oosterl, Hans van der Weide (2011)
    """

    tis = jnp.linspace(1E-10, n_years, T + 1)  # grid - vector of time steps - starts at 1e-10 to avoid unpleasantness
    t = jnp.expand_dims(tis, axis=-1)  # for numpy broadcasting
    dt = n_years / (T)



    # Distributions samples
    dW2 = jax.random.normal(jax.random.PRNGKey(0), (T, N)) * jnp.sqrt(dt)
    U1 = jax.random.uniform(jax.random.PRNGKey(0), (T, N))
    U = jax.random.uniform(jax.random.PRNGKey(1), (T, N))
    Z = jax.random.normal(jax.random.PRNGKey(1), (T, N))
    W2t = simulate_Wt(dW2, T, N)

    # vol process
    sigma_t = simulate_sigma(W2t, sigma0, alpha, t)

    # integrated variance- values are integrals between ti-1 and ti
    # not integrals over the whole interval [0,ti] distribution is approx. log normal
    if trapezoidal_integrated_variance:
        v_t = integrated_variance_trapezoidal(rho, sigma_t, dt)
    else:
        v_t = integrated_variance_small_disturbances(N, rho, alpha, sigma_t, dt, dW2, U1)

    b = 2. - ((1. - 2. * beta - (1. - beta) * (rho ** 2)) / ((1. - beta) * (1. - rho ** 2)))

    # initialize underlying values
    Ft_arr = [F0 * jnp.ones(N)]
    
    for ti in range(1, T):
        row = [] #compute_Ft(Ft_arr[ti - 1], v_t[ti - 1], v_t[ti - 1], beta, rho, alpha, sigma_t[ti], sigma_t[ti - 1], b, psi_threshold, Z[ti - 1])

        for n in range(0, N):
            row.append(compute_Ft(Ft_arr[ti - 1][n], v_t[ti - 1, n], v_t[ti - 1, n], beta, rho, alpha, sigma_t[ti, n], sigma_t[ti - 1, n], b, psi_threshold, Z[ti - 1, n], U[ti - 1, n]))
        Ft_arr.append(row)

    return jnp.array(Ft_arr)


if __name__ == '__main__':
    # (F0=1+r, sigma0=0.25, alpha=0.001, beta=0.999, rho=0.001, psi_threshold=2., n_years=100000, T=100, N=100000, trapezoidal_integrated_variance=False)
    # np.random.seed(1)
    Ft = sabrMC(F0 = 1., N=1000, T=100, n_years=1)
    data = Ft[-1, :]
    print(jnp.mean(Ft[-1, :]))
    print(Ft[-1, :][:5])

    # np.savetxt("sample.csv", Ft, delimiter=",")

    # density = gaussian_kde(data)
    # xs = np.linspace(-0.1, 0.2, 500)
    # density.covariance_factor = lambda: .25
    # density._compute_covariance()
    # pyplot.plot(xs, density(xs))
    # pyplot.show()
    # pyplot.plot(Ft)
    # pyplot.show()
