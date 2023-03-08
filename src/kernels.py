"""Construction of covariance-kernels for source signals in time and frequency.

We consider radial basis function kernels and the matern family of kernels.
Since all of our kernels are stationary, their power spectrum is given by
  their fourier transform. See https://gaussianprocess.org/gpml/ for details
"""

import functools  # standard for jax, pylint: disable=g-importing-member
from typing import Callable, Sequence, TypeVar, Union, Any

import jax
from jax import numpy as jnp
from jax._src import prng
from jax._src import typing as jaxtyping
from jax.numpy import fft
from jax.scipy.special import gammaln


RealArray = jaxtyping.ArrayLike
IntegerArray = jaxtyping.ArrayLike
KeyArray = Union[jaxtyping.Array, prng.PRNGKeyArray]
Shape = Sequence[int]
PyTree = Union[dict[Any, Any], list[Any], jaxtyping.ArrayLike]

ModelEvidence = TypeVar("ModelEvidence", bound=Callable[[PyTree], float])

SpectralDensity = TypeVar("SpectralDensity", bound=Callable)
CovarianceKernel = TypeVar("CovarianceKernel", bound=Callable)


@jax.jit
def matern12_cov(t0: RealArray, t1: RealArray, marg_var: float,
                 lengthscale: float) -> RealArray:
  r"""Evaluates Matern 1/2 kernel / covariance at two timepoints.

  The Matern 1/2 kernel is given by:
  .. math::
  \mathrm{marg_var} \exp(- \frac{ \|t0-t1\|_2}{\mathrm{lengthscale}} )

  Args:
    t0: timepoints at which to evaluate kernel.
      Must be of compatible dimensions with t1.
    t1: timepoints at which to evaluate kernel.
      Must be of compatible dimensions with t0.
    marg_var: marginal variance, i.e. kernel value when t0 = t1
    lengthscale: determines distance at which two.
      timepoints are considered similar to each other. Inverse of bandwidth.

  Returns:
    A RealArray of possitive values of the broadcasted shape of t0 - t1.
  """
  return  marg_var * jnp.exp(-jnp.abs(t0 - t1) / lengthscale)


@jax.jit
def matern32_cov(t0: RealArray, t1: RealArray, marg_var: float,
                 lengthscale: float) -> RealArray:
  r"""Evaluates Matern 3/2 kernel / covariance at two timepoints.

  The Matern 3/2 kernel is given by:
    .. math::
    \mathrm{marg_var} \left( 1 + \frac{\sqrt{3}d}{\mathrm{lengthscale}} \right)
    \exp(- \frac{\sqrt{3}d}{\mathrm{lengthscale}} ) \textrm{ for } d =
    \|t0-t1\|_2

  Args:
    t0: timepoints at which to evaluate kernel. Must be of compatible dimensions
      with t1.
    t1: timepoints at which to evaluate kernel. Must be of compatible dimensions
      with t0.
    marg_var: marginal variance, i.e. kernel value when t0 = t1.
    lengthscale: determines distance at which two timepoints are considered
      similar to each other. Inverse of bandwidth.

  Returns:
    A RealArray of possitive values of the broadcasted shape of t0 - t1.
  """
  scaled_dist = jnp.sqrt(3.) * jnp.abs(t0 - t1) / lengthscale
  normaliser = 1 + scaled_dist
  exponential_term = jnp.exp(-scaled_dist)
  return marg_var * normaliser * exponential_term


@jax.jit
def matern52_cov(t0: RealArray, t1: RealArray, marg_var: float,
                 lengthscale: float) -> RealArray:
  r"""Evaluates Matern 5/2 kernel / covariance at two timepoints.

  The Matern 5/2 kernel is given by:
    .. math::
      \mathrm{marg_var} \left( 1 + \frac{\sqrt{5}d}{\mathrm{lengthscale}} +
      \frac{5 d^2}{3 \mathrm{lengthscale}^2} \right) \exp(-
      \frac{\sqrt{5}d}{\mathrm{lengthscale}} ) \textrm{ for } d = \|t0-t1\|_2


  Args:
    t0: timepoints at which to evaluate kernel. Must be of compatible dimensions
      with t1.
    t1: timepoints at which to evaluate kernel. Must be of compatible dimensions
      with t0.
    marg_var: marginal variance, i.e. kernel value when t0 = t1.
    lengthscale: determines distance at which two timepoints are considered
      similar to each other. Inverse of bandwidth.

  Returns:
    A RealArray of possitive values of the broadcasted shape of t0 - t1.
  """
  scaled_dist = jnp.sqrt(5.) * jnp.abs(t0 - t1) / lengthscale
  normaliser = (1 + scaled_dist + (1 / 3) * scaled_dist**2)
  exponential_term = jnp.exp(-scaled_dist)
  return marg_var * normaliser * exponential_term


@jax.jit
def rbf_cov(t0: RealArray, t1: RealArray, marg_var: float,
            lengthscale: float) -> jaxtyping.ArrayLike:
  r"""Evaluates Radial Basis Function (RBF) kernel at two timepoints.

  The RBF kernel is given by:
    .. math::
    \mathrm{marg_var} \left( 1 + \frac{\sqrt{5}d}{\mathrm{lengthscale}} +
    \frac{5 d^2}{3 \mathrm{lengthscale}^2} \right) \exp(-
    \frac{\sqrt{5}d}{\mathrm{lengthscale}} ) \textrm{ for } d = \|t0-t1\|_2

  Args:
    t0: timepoints at which to evaluate kernel. Must be of compatible dimensions
      with t1.
    t1: timepoints at which to evaluate kernel. Must be of compatible dimensions
      with t0.
    marg_var: marginal variance, i.e. kernel value when t0 = t1.
    lengthscale: determines distance at which two timepoints are considered
      similar to each other. Inverse of bandwidth.

  Returns:
    A RealArray of possitive values of the broadcasted shape of t0 - t1.
  """
  return marg_var * jnp.exp(-(jnp.abs(t0 - t1)**2) / (2 * lengthscale**2))


def gen_spectral_mixture_kernel_fn(
    base_kernel_fn: CovarianceKernel,
    sum_spectral_components: bool = True) -> CovarianceKernel:
  """Creates spectral mixture kernel from a baseband CovarianceKernel."""

  def spectral_mixture_kernels(t0: RealArray, t1: RealArray,
                               marg_var_vec: RealArray,
                               lengthscale_vec: RealArray,
                               carrier_vec: RealArray) -> RealArray:
    r"""Evaluates spectral mixture kernel at two timepoints.

    The spectral mixture kernel is built from a set of k base kernels K_1...K_k
      as:
    .. math::
    \sum_{i=1}^k \cos(2\pi f_i d) K_i(d) \textrm{ for } d = \|t0-t1\|_2

    Args:
      t0: timepoints at which to evaluate kernel.
        Must be of compatible dimensions with t1.
      t1: timepoints at which to evaluate kernel.
        Must be of compatible dimensions with t0.
      marg_var_vec: array of each kernel's marginal variance.
      lengthscale_vec: array of each kernel's lengthscale.
      carrier_vec: array of carrier frequency with which to modulate
        each kernel.

    Returns:
    TODO: explain effect of sum_spectral_components

      A RealArray of possitive values of the broadcasted shape of t0 - t1.
    """
    def modulated_kernel(marg_var: float, lengthscale: float, carrier_f: float):
      """Modulates base_kernel_fn by multiplying it with a cosine."""
      return jnp.cos(
          2 * jnp.pi * carrier_f * jnp.abs(t1 - t0)) * base_kernel_fn(
              t0, t1, marg_var, lengthscale)

    vmap_modulated_kernel = jax.jit(
        jax.vmap(modulated_kernel, in_axes=(0, 0, 0)))
    return vmap_modulated_kernel(
        marg_var_vec, lengthscale_vec,
        carrier_vec)

  def summed_spectral_mixture_kernels(t0: RealArray, t1: RealArray,
                                      marg_var_vec: RealArray,
                                      lengthscale_vec: RealArray,
                                      carrier_vec: RealArray) -> RealArray:
    return spectral_mixture_kernels(t0, t1, marg_var_vec, lengthscale_vec,
                                    carrier_vec).sum(axis=0)

  if sum_spectral_components:
    return jax.jit(summed_spectral_mixture_kernels)
  else:
    return jax.jit(spectral_mixture_kernels)


@jax.jit
def rbf_spectral_density(f: RealArray,
                         marg_var: float,
                         lengthscale: float,
                         carrier_f: float = 0) -> jaxtyping.ArrayLike:
  r"""Evaluates Radial Basis Function spectral density at given frequencies.

  The RBF spectral density is given by:
    .. math::
    \mathrm{marg_var} (2 \pi \mathrm{lengthscale})^{1/2} \exp(-2 \pi^2
    \mathrm{lengthscale}^2 (f - \mathrm{carrier_f})^2)

  Args:
    f: frequency array (corresponding to FFT of continuous time process). Given
      that kernels are possitive functions, a standard choice would be
      jnp.arange(0, int(Ns/2 + 1)) * fs  / Ns for Ns the number of samples and
      fs the sampling frequency.
    marg_var: marginal variance, i.e. kernel value when t0 = t1.
    lengthscale: determines distance at which two.
      timepoints are considered similar to each other. Inverse of bandwidth.
    carrier_f: frequency of carrier tone with which to modulate kernel.

  Returns:
    A RealArray of possitive values of the shape of f.
  """
  log_normaliser = jnp.log(2 * jnp.pi) + 2 * jnp.log(lengthscale)
  log_exponential_term = -2 * (jnp.pi**2) * (lengthscale**
                                             2) * (f - carrier_f)**2
  return marg_var * jnp.exp(0.5 * log_normaliser + log_exponential_term)


def gen_matern_spectral_density_fn(order: int) -> SpectralDensity:
  r"""Returns method that evaluates Matern Function spectral density.

  The Matern spectral density is given by:
    .. math::
    \mathrm{marg_var} \frac{ (2 \pi \Gamma(\mathrm{order} + 0.5)  (2
    \mathrm{order})^{\mathrm{order}}}{\Gamma(\mathrm{order})
    \mathrm{lengthscale}^{2\mathrm{order}} }
    \left(\frac{2\mathrm{order}}{\mathrm{lengthscale}^2} 4 \pi^2(f -
    \mathrm{carrier_f})^2 \right)^{- \mathrm{order} + 0.5}

  Args:
    order: Order of Matern function to use. Supported: [1,3,5].
  """
  if order not in [1, 3, 5]:
    raise ValueError("Only orders in [1, 3, 5] are supported.")

  log_gamma_v = gammaln(order)
  log_gamma_v_p_half = gammaln(order + 0.5)

  def matern_spectral_density(f: RealArray,
                              marg_var: float,
                              lengthscale: float,
                              carrier_f: float = 0) -> jaxtyping.ArrayLike:
    """Evaluates Matern Function spectral density at given frequencies.

    Args:
      f: frequency array (corresponding to FFT of continuous time process).
        Given that kernels are possitive functions, a standard choice would be
        jnp.arange(0, int(Ns/2 + 1)) * fs  / Ns for Ns the number of samples and
        fs the sampling frequency.
      marg_var: marginal variance, i.e. kernel value when t0 = t1.
      lengthscale: determines distance at which two
        timepoints are considered similar to each other. Inverse of bandwidth.
      carrier_f: frequency of carrier tone with which to modulate kernel.

    Returns:
      A RealArray of possitive values of the shape of f.
    """
    log_numerator = jnp.log(2) + 0.5 * jnp.log(
        jnp.pi) + log_gamma_v_p_half + order * jnp.log(2 * order)
    log_denominator = log_gamma_v + (2 * order) * jnp.log(lengthscale)
    log_power_term = (-order - 0.5) * jnp.log((2 * order / lengthscale**2) +
                                              4 * jnp.pi**2 *
                                              (f - carrier_f)**2)
    return marg_var * jnp.exp(log_numerator + log_power_term - log_denominator)

  return jax.jit(matern_spectral_density)


def parametrise_spectral_mixture_kernel(
    kernel_fun: SpectralDensity,
    params: PyTree) -> SpectralDensity:
  """Returns a spectral mixture kernel function with fixed parameters."""
  marg_vars = jnp.exp(params["log_marg_var"])
  lengthscales = jnp.exp(params["log_lengthscale"])
  carrier_freqs = jnp.exp(params["log_carrier_freq"])
  kernel = functools.partial(
      kernel_fun,
      marg_var_vec=marg_vars,
      lengthscale_vec=lengthscales,
      carrier_vec=carrier_freqs)
  return jax.jit(kernel)


def gen_spectral_mixture_density_fn(
    base_density_fn: SpectralDensity) -> SpectralDensity:
  """Creates spectral mixture kernel from a baseband CovarianceKernel."""

  def spectral_mixture_density(f: RealArray, marg_var_vec: RealArray,
                               lengthscale_vec: RealArray,
                               carrier_vec: RealArray) -> RealArray:
    """Creates spectral mixture kernel from a baseband CovarianceKernel.

    Args:
      f: frequency array (corresponding to FFT of continuous time process).
      Given that kernels are possitive functions, a standard choice would be
      jnp.arange(0, int(Ns/2 + 1)) * fs  / Ns for Ns the number of samples and
      fs the sampling frequency.
      marg_var_vec: array of each kernel's marginal variance.
      lengthscale_vec: array of each kernel's lengthscale.
      carrier_vec: array of carrier frequency with which to modulate
        each kernel.

    Returns:
      A RealArray of possitive values of the broadcasted shape of t0 - t1.
    """
    vmap_modulated_density = jax.jit(
        jax.vmap(base_density_fn, in_axes=(None, 0, 0, 0)))
    pos_freqs = vmap_modulated_density(f, marg_var_vec, lengthscale_vec,
                                       carrier_vec).sum(axis=0)
    neg_freqs = vmap_modulated_density(f, marg_var_vec, lengthscale_vec,
                                       -carrier_vec).sum(axis=0)
    return 0.5 * pos_freqs + 0.5 * neg_freqs

  return jax.jit(spectral_mixture_density)


@jax.jit
def stationary_solve(s: RealArray, v: RealArray) -> RealArray:
  r"""Efficiently approximates linear solve K^-1 v for stationary kernel matrix K.

  For a stationary kernel function k with power spectrum s evaluated at a
  sufficiently long sequence of equally spaced indices, we can approximate a
  linear solve with vector v as:
  .. math::
  K^{-1} v = FT^{-1} \frac{FT v}{s}
    \mathrm{ for } FT \mathrm{ the fourier operator }
  See appendix A.2 of http://www.gatsby.ucl.ac.uk/~turner/Publications/turner-2010.pdf  # link pylint: disable=line-too-long

  Args:
    s: Vector of size Ns / 2 + 1 containing possitive fourier coefficients of
    stationary covariance kernel.
    v: Vector of size Ns which to solve against.

  Returns:
    Vector of size Ns which approximates linear solve.
  """
  return fft.irfft(fft.rfft(v, norm="ortho") / s, norm="ortho")


def stationary_matsolve(s: RealArray, M: RealArray) -> RealArray:  # single capital for matrix notation,  pylint: disable=g-bad-name
  """Efficiently approximates linear solves K^-1 M for stationary kernel matrix K.

  Args:
    s: Vector of size (Ns/2+1,) containing possitive fourier coefficients of
    stationary covariance kernel.
    M: Matrix of size (Ns,d) which to solve against.

  Returns:
    Matrix of size (Ns,d) which approximates linear solve.
  """
  def fixed_s_solve(v):
    return stationary_solve(s=s, v=v)
  # vmap solve over vectors in M
  vmap_stationary_matsolve = jax.jit(jax.vmap(fixed_s_solve))
  return vmap_stationary_matsolve(M.T).T

