"""Learning objectives for simultaneous source localisation and separation.

Compute objective functions which we optimise to learn parameters for stationary
Gaussian process models of the source signals and source location parameters.
"""

import copy
import functools
from typing import Any, Callable, Sequence, Tuple, TypeVar, Union

import jax
from jax import numpy as jnp
from jax import random
from jax import typing as jaxtyping

from google3.util.compression.korvapuusti.up_mixing.src import kernels
from google3.util.compression.korvapuusti.up_mixing.src import position_model
from google3.util.compression.korvapuusti.up_mixing.src import utils

RealArray = jaxtyping.ArrayLike
IntegerArray = jaxtyping.ArrayLike
KeyArray = Union[jax.Array, random.PRNGKeyArray]
Shape = Sequence[int]
PyTree = Union[dict[Any, Any], list[Any], jaxtyping.ArrayLike]

ModelEvidence = TypeVar("ModelEvidence", bound=Callable[[PyTree], float])
SignalPredictor = TypeVar(
    "SignalPredictor", bound=Callable[[RealArray], Tuple[RealArray, RealArray]])

# Tools to build probabilistic model for signals at microphones


def construct_delayed_kernel_mat(t: RealArray,
                                 kernel_fun: Callable[[RealArray, RealArray],
                                                      RealArray],
                                 delay: float) -> Tuple[RealArray, RealArray]:
  r"""Constructs kernel matrix k00 and delayed kernel matrix k01.

  .. math::
    [k00]_{ij} = \mathrm{kernel_fun}(t_i, t_j) \mathrm{ and }
    [k01]_{ij} = \mathrm{kernel_fun}(t_i, t_j - \mathrm{delay})

  Args:
    t: [num_samples] RealArray containing timepoints at which to evaluate
      covariance functions.
    kernel_fun: Function that takes in two vectors of shape (s1, ) and (s2, ),
      respectively, and returns a matrix of shape (s1, s2).
    delay: expected time delay (in seconds) between function measured with two
      different time points.

  Returns:
    A tuple (k00, k01) where
    k00: Array of size (num_samples, num_samples) containing covariance function
      evaluated at timepoints t.
    k01: Array of size (num_samples, num_samples) containing cross covariance
      function evaluated at timepoints t and t - delay.
  """

  # Rectify delay such that a possitive delay is always applied to one of the
  # axis
  delay_0 = jnp.clip(delay, a_min=0)
  delay_1 = jnp.clip(-delay, a_min=0)

  t0 = t - delay_0  # size (num_samples,)
  t1 = t - delay_1

  k00 = kernel_fun(t0[:, None], t0[None, :])  # size (num_samples, num_samples)
  k01 = kernel_fun(t0[:, None], t1[None, :])

  return k00, k01


@jax.jit
def build_single_source_mic_covariance(source_params: PyTree,
                                       microphone_positions: RealArray,
                                       observed_t: RealArray, c: float,
                                       source_length: float) -> RealArray:
  """Returns the noiseless microphone covariance mat induced by a single source.

  For two observed signals (y_0, y_1), one at each microphone, stacked as
  [y_0^T, y_1^T]^T, the (noiseless) covariance matrix is given by
  [[k00, k01], [k01, K11] for k00 and K11 the covariance matrix of the signal
  emitted by a single source and observed at each individual microphone and
  k01=K10.T the signals' cross covariance matrices.
  In the noiseless two microphone setup, y_0 and y_1,
  and thus their covariance matrices, differ only in a delay and attenuation.

  Args:
    source_params: PyTree containing parameters for a single source model.
    microphone_positions: Array of size (2, 2) containing microphone positions
      in cartesian coordinates (meters) stacked as [[x_0, y_0],[x_1, y_1]].
    observed_t: [num_samples] RealArray containing the timepoints at which we
      measure the signal amplitude at the microphone.
    c: speed of sound (meters / second).
    source_length: minimum distance between sources and microphones in meters.

  Returns:
    Covariance matrix of size (2 num_samples, 2 num_samples) corresponding to
    the signal from a single source recorded at the stereo microphones.
  """

  r0 = utils.propagation_radius(
      utils.polar_params_to_cartesian(
          source_params["source_position"], source_length=source_length),
      microphone_positions[0])
  r1 = utils.propagation_radius(
      utils.polar_params_to_cartesian(
          source_params["source_position"], source_length=source_length),
      microphone_positions[1])

  # we work with relative delay between microphones
  relative_delay = (r0 - r1) / c

  spectral_mixture_kernel = kernels.gen_spectral_mixture_kernel_fn(
      kernels.matern32_cov)
  kernel = jax.jit(
      functools.partial(
          spectral_mixture_kernel,
          marg_var_vec=jnp.exp(source_params["log_marg_var"]),
          lengthscale_vec=jnp.exp(source_params["log_lengthscale"]),
          carrier_vec=jnp.exp(source_params["log_carrier_freq"])))

  # Get covariance blocks for sequence with itself and sequence with delayed
  #   sequence
  k00, k01 = construct_delayed_kernel_mat(t=observed_t, kernel_fun=kernel,
                                          delay=relative_delay)

  # remaining covariance blocks are obtained by symmetry
  k10 = k01.T
  k11 = k00

  return jnp.concatenate([
      jnp.concatenate([k00 / (r0 * r0), k01 / (r0 * r1)], axis=1),
      jnp.concatenate([k10 / (r0 * r1), k11 / (r1 * r1)], axis=1)
  ], axis=0)


@jax.jit
def build_mic_covariance(params: PyTree,
                         microphone_positions: RealArray,
                         observed_t: RealArray,
                         c: float,
                         source_length: float,
                         noise_eps: float = 1e-5) -> RealArray:
  """Returns mic covariance mat for a sum of independent sources and noise.

  For two observed signals (y_0, y_1), one at each microphone, stacked as
  [y_0^T, y_1^T]^T, the (noiseless) covariance matrix is given by
  [[k00, k01], [k01, K11] for k00 and K11 the covariance matrix of the sum of
  all sources' signals observed at each individual microphone and
  k01=K10.T the signals' cross covariance matrices. We add a scaled identity
  matrix to account for additive Gaussian noise which is independent across
  timepoints and microphones.

  Args:
    params: PyTree containing the parameters for all sources in our model and
      the variance of the noise at the microphones.
    microphone_positions: Array [[x0, y0], [x1, y1]] of microphone positions
      (in meters).
    observed_t: [num_samples] RealArray containing the timepoints at which we
      measure the signal amplitude at both microphones.
    c: speed of sound (meters / second).
    source_length: minimum distance between sources and microphones (meters).
    noise_eps: constant that is added to the diagonal of the observation
      covariance matrix to ensure it is PD. This represents the minimum noise
      floor. (Default: 1e-5)

  Returns:
    Covariance matrix of size (2 num_samples, 2 num_samples) corresponding to
    the sum of multiple sources' signals and noise recorded at the stereo
    microphones.
  """

  source_params = params["source_params"]
  log_noise_var = params["log_noise_var"]

  stacked_kyy = jax.jit(  # jax.vmap over number of sources in model
      jax.vmap(
          functools.partial(
              build_single_source_mic_covariance,
              microphone_positions=microphone_positions,
              observed_t=observed_t,
              c=c,
              source_length=source_length))
  )(source_params)  # size (num_sources, 2 num_samples, 2 num_samples)

  regulariser = jnp.ones(2 * len(observed_t)) * (
      jnp.exp(log_noise_var) + noise_eps)

  return stacked_kyy.sum(axis=0) + jnp.diag(
      regulariser)  # size (2 num_samples, 2 num_samples)


@jax.jit
def gp_evidence(y: RealArray, kyy: RealArray) -> float:
  r"""Computes model evidence of Gaussian process up to a constant.

  The log-probability density of y under a multivariate normal with covariance
  matrix Kyy is given up to a constant factor by
  .. math::
    - 0.5 * (y^T K_{yy}^{-1} y + \mathrm{logdet}(K_{yy}))

  Args:
    y: array of size (num_samples, ) representing a vector of observations.
    kyy: PSD matrix of size (num_samples, num_samples) representing covariance
      of a Gaussian process indexed at the observation points.

  Returns:
    A floating point number representing the log probability density of y
      under the GP model.
  """
  chol = jax.scipy.linalg.cholesky(kyy, lower=True)
  fit_term = jax.scipy.linalg.cho_solve(c_and_lower=(chol, True), b=y) @ y
  logdet = 2 * jnp.log(jnp.diag(chol)).sum()
  return 0.5*(-fit_term - logdet)


@jax.jit
def two_microphone_evidence(params: PyTree,
                            microphone_positions: RealArray,
                            y0: RealArray,
                            y1: RealArray,
                            observed_t: RealArray,
                            c: float,
                            source_length: float,
                            noise_eps: float = 1e-5) -> float:
  """Computes evidence for our source model given observations at two mics.

  Our model's evidence (also known as marginal likelihood) is the
  log-probability it assigns to the observed data when all latent
  variables---here the source signals---are marginalised out. This quantity
  can be seen as a score for the quality of our parameters: the kernel and
  source positions and can be used to optimise the values of these.

  Args:
    params: PyTree containing the parameters for all sources in our model and
      the variance of the noise at the microphones.
    microphone_positions: Array of size (2, 2) containing microphone positions
      in cartesian coordinates (meters) stacked as [[x_0, y_0],[x_1, y_1]].
    y0: Signal recorded at microphone 0.
    y1: Signal recorded at microphone 1.
    observed_t: [num_samples] RealArray containing the timepoints at which we
      measure the signal amplitude at both microphones.
    c: speed of sound (meters / second).
    source_length: minimum distance between sources and microphones (meters).
    noise_eps: constant that is added to the diagonal of the observation
      covariance matrix to ensure it is PD. This represents the minimum noise
      floor. (Default: 1e-5)

  Returns:
    Log probability density of observations at microphones under the specified
      source model.
  """

  kyy = build_mic_covariance(params, microphone_positions, observed_t, c,
                             source_length,
                             noise_eps)  # size (2 num_samples, 2 num_samples)

  y_stack = jnp.concatenate([y0, y1], axis=0)  # shape (2 num_samples,)

  return gp_evidence(y_stack, kyy)


def get_fully_learnable_evidence_fn(microphone_positions: RealArray,
                                    y0: RealArray,
                                    y1: RealArray,
                                    observed_t: RealArray,
                                    c: float,
                                    source_length: float,
                                    noise_eps: float = 1e-5) -> ModelEvidence:
  """Returns partially evaluated two_microphone_evidence.
  """

  def fully_learnable_evidence(params: PyTree) -> float:
    return two_microphone_evidence(
        params=params,
        microphone_positions=microphone_positions,
        y0=y0,
        y1=y1,
        observed_t=observed_t,
        c=c,
        source_length=source_length,
        noise_eps=noise_eps)

  return jax.jit(fully_learnable_evidence)


def get_single_source_learnable_evidence_fn(
    base_params: PyTree,
    learnable_source_idx: int,
    microphone_positions: RealArray,
    y0: RealArray,
    y1: RealArray,
    observed_t: RealArray,
    c: float,
    source_length: float,
    noise_eps: float = 1e-5) -> ModelEvidence:
  """Returns view of two_microphone_evidence with a single learnable source.

  The returned function stops gradients by substituting the parameters for
  sources with indices different from learnable_source_idx with the parameters
  for the corresponding sources from base_params.

  TODO(antoran): re-implement this method using jax.lax.stop_gradient(x)
  https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.stop_gradient.html

  Args:
    base_params: Parameter PyTree that will be used for sources different than
      the one specified by learnable_source_idx.
    learnable_source_idx: index of source for which to learn parameters.
    microphone_positions: Array containing microphone positions
      in cartesian coordinates (meters) stacked as [[x_0, y_0],[x_1, y_1]].
    y0: Signal recorded at microphone 0.
    y1: Signal recorded at microphone 1.
    observed_t: [num_samples] RealArray containing the timepoints at which we
      measure the signal amplitude at both microphones.
    c: speed of sound (meters / second).
    source_length: minimum distance between sources and microphones (meters).
    noise_eps: constant that is added to the diagonal of the observation
      covariance matrix to ensure it is PD. This represents the minimum noise
      floor. (Default: 1e-5)

  Returns:
    A functools.partial view of two_microphone_evidence
    that takes in a full params PyTree but stops gradients with respect to the
    parameters of all but one source model, specified by learnable_source_idx.
  """

  def source_set(x, y):
    return x.at[learnable_source_idx].set(y[learnable_source_idx])

  def single_source_learnable_evidence(params: PyTree) -> float:
    new_params = copy.deepcopy(base_params)
    new_params["source_params"] = jax.tree_util.tree_map(
        source_set, new_params["source_params"], params["source_params"])
    new_params["log_noise_var"] = params["log_noise_var"]
    return two_microphone_evidence(
        params=new_params,
        microphone_positions=microphone_positions,
        y0=y0,
        y1=y1,
        observed_t=observed_t,
        c=c,
        source_length=source_length,
        noise_eps=noise_eps)

  return jax.jit(single_source_learnable_evidence)


# Objectives for learning stochastic source position model


def gen_single_sample_elbo_fn(
    likelihood_func: Callable[[PyTree], jaxtyping.ArrayLike],
    source_length: float) -> Callable[[KeyArray, PyTree], jaxtyping.ArrayLike]:
  """Returns function that evaluates a single sample expectation of the elbo.

  Args:
    likelihood_func: function that computes log-probability of observations
      under the model with parameters set by its input.
    source_length: minimum distance between sources and microphones (meters).
  """

  # jax.vmap over random keys and position model parameters
  single_sample_source_positions = jax.jit(
      functools.partial(
          sample_source_positions,
          num_samples=1,
          source_length=source_length))

  def single_sample_elbo(key: KeyArray, variational_params: PyTree):
    r"""Evaluates single sample estimate of the elbo.

    Computes a single sample estimate of a standard variational lower bound,
    see
    https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf
    where an approximate distribution q is placed over each source's radius r
    and angle A.
    .. math::
      p(y) \geq \log p(Y=y| {r_i, a_i}_{i=1}^{\mathrm{num_sources}}) +
      \sum_{i=1}^{\mathrm{num_sources}} \left( \log p(R=r_i) - \log q(R=r_i)
      - \log q(R=a_i) \right) \mathrm{ with }
      {r_i, a_i}_{i=1}^{\mathrm{num_sources}} \sim q

    Args:
      key: jax random key.
      variational_params: PyTree containing parameters for the variational
        distribution over positions and likelihood function.
    Returns:
      A scalar representing a single sample estimate of the elbo.
    """
    (radius_params, radius, radius_log_probs, angle_params, _,
     angle_log_probs) = single_sample_source_positions(
         key, variational_params)

    # params["source_params"] is a new dictionary to which we can add entries
    # safely
    params = {"source_params": copy.copy(variational_params["source_params"]),
              "log_noise_var": variational_params["log_noise_var"]}
    params["source_params"]["source_position"] = jnp.stack(
        [radius_params[:, 0], angle_params[:, 0]], axis=1)

    radius_log_priors = position_model.radius_log_prior(
        radius)  # size (num_sources,),

    sample_log_likelihood = likelihood_func(params)  # scalar

    elbo_estimate = sample_log_likelihood + (
        -radius_log_probs + radius_log_priors -
        angle_log_probs).sum()  # sum across sources
    return elbo_estimate

  return jax.jit(single_sample_elbo)


def gen_multi_sample_estimator_fn(
    single_sample_estimator: Callable[[KeyArray, PyTree], jaxtyping.ArrayLike],
    num_samples: int) -> Callable[[KeyArray, PyTree], jaxtyping.ArrayLike]:
  """Returns a function to evaluate multi-sample estimator of an objective func.

  Args:
    single_sample_estimator: method that evaluates single-sample estimator of
      the objective function.
    num_samples: number of samples with which to estimate objective function.
  """

  def multi_sample_estimator(key: KeyArray, params: PyTree):
    r"""Returns multi-sample estimator of expectation over specified function.

    Computes
    .. math::
      \frac{1}{\mathrm{num_samples}} \sum_{i=1}^{\mathrm{num_samples}}
      \mathrm{single_sample_estimator}(r, \mathrm{params}) \mathrm{ for } r_i
      \sim p(r)
    where r refers to a random variable over which we take the expectation

    Args:
      key: jax random key to use for sample generation.
      params: non-random parameters of the function.
    """
    vmap_single_sample_estimator = jax.jit(
        jax.vmap(
            functools.partial(
                single_sample_estimator, variational_params=params)))
    keys = jax.random.split(key, num_samples)
    samples = vmap_single_sample_estimator(keys)
    return samples.mean()

  return jax.jit(multi_sample_estimator)


# Parameter PyTree generation and manipulation


def initialise_spectral_mixture_params(n_sources: int,
                                       radius_param: float,
                                       angle_param_range: float,
                                       log_marg_var_vec: RealArray,
                                       log_lengthscale_vec: RealArray,
                                       log_carrier_vec: RealArray,
                                       log_noise_var: float) -> PyTree:
  """Returns PyTree with position and kernel parameters for sources.

  All sources are assigned the same spectral mixture kernel parameters at
  initialisation. All sources are assigned the same radius parameter but their
  angle parameters are different.

  Args:
    n_sources: number of sources for which to initialise parameters.
    radius_param: scalar parameter representing the log the radius minus the
      source_length parameter.
    angle_param_range: The means of different sources' distribution over the
      angle parameter (i.e. the logit of the angle divided by pi) are
      initialised uniformly separated in the [-angle_param_range,
      angle_param_range] range.
    log_marg_var_vec: jnp.ndarray containing the log-marginal-variances (i.e.
      power) of each spectral mixture component. The number of mixture
      components is inferred from the length of the array.
    log_lengthscale_vec: jnp.ndarray containing the log-lengthscales (i.e.
      inverse bandwidth) of each spectral mixture component. The number of
      mixture components is inferred from the length of the array.
    log_carrier_vec: jnp.ndarray containing the carrier frequency of each
      spectral mixture component. The number of mixture components is inferred
      from the length of the array and must match the length of
      log_lengthscale_vec and log_marg_var_vec.
    log_noise_var: the log of the variance (i.e. power) of the noise floor at
      initialisation.

  Returns:
    A PyTree with contents compatible with the learning objective function
    two_microphone_evidence.
  """

  if not len(log_marg_var_vec) == len(log_lengthscale_vec) == len(
      log_carrier_vec):
    raise ValueError("log_marg_var_vec, log_lengthscale_vec and"
                     " log_carrier_vec must be the same length")

  single_source_params = {
      "source_position": jnp.array([radius_param, 0.]),
      "log_lengthscale": log_lengthscale_vec,
      "log_marg_var": log_marg_var_vec,
      "log_carrier_freq": log_carrier_vec,
  }

  source_params = jax.tree_util.tree_map(
      utils.repeat_for_pytree(n_sources), single_source_params)
  # We are permitted to adjust these outer layers on the newly allocated PyTree.
  source_params["source_position"] = source_params[
      "source_position"].at[:, 1].set(
          jnp.linspace(-angle_param_range, angle_param_range,
                       n_sources).astype(float))
  return dict(log_noise_var=log_noise_var, source_params=source_params)


def initialise_spectral_mixture_variational_params(
    n_sources: int, radius_mean: float, radius_logvar: float,
    angle_param_range: float, angle_param_logvar: float,
    log_marg_var_vec: RealArray, log_lengthscale_vec: RealArray,
    log_carrier_vec: RealArray, log_noise_var: float) -> PyTree:
  """Returns PyTree with sources' kernel parameters and variational parameters.

  All sources are assigned the same spectral mixture kernel parameters at
  initialisation. All sources are assigned the same distribution over radius
  parameter at initialisation. The mean of each source's distirbution over angle
  is chosen uniformly from within the range specified by angle_param_range. The
  initial variance of the distribution over angles is shared among all sources.

  Args:
    n_sources: number of sources for which to initialise parameters.
    radius_mean: the mean of all sources' distribution over their radius.
    radius_logvar: the log of the variance of all sources' distribution over
      radius.
    angle_param_range: The means of different sources' distribution over the
      angle parameter (i.e. the logit of the angle divided by pi) are
      initialised uniformly separated in the [-angle_param_range,
      angle_param_range] range.
    angle_param_logvar: the log of the variance of all sources' distribution
      over the angle parameter (i.e. the logit of the angle divided by pi).
    log_marg_var_vec: jnp.ndarray containing the log-marginal-variances (i.e.
      power) of each spectral mixture component. The number of mixture
      components is inferred from the length of the array.
    log_lengthscale_vec: jnp.ndarray containing the log-lengthscales (i.e.
      inverse bandwidth) of each spectral mixture component. The number of
      mixture components is inferred from the length of the array.
    log_carrier_vec: jnp.ndarray containing the carrier frequency of each
      spectral mixture component. The number of mixture components is inferred
      from the length of the array.
    log_noise_var: the log of the variance (i.e. power) of the noise floor at
      initialisation.

  Returns:
    A PyTree with contents compatible with the learning objective function
    returned by gen_single_sample_elbo_fn.
  """

  if not len(log_marg_var_vec) == len(log_lengthscale_vec) == len(
      log_carrier_vec):
    raise ValueError("log_marg_var_vec, log_lengthscale_vec and"
                     " log_carrier_vec must be the same length")

  params = initialise_spectral_mixture_params(
      n_sources=n_sources,
      radius_param=jnp.nan,
      angle_param_range=1,
      log_marg_var_vec=log_marg_var_vec,
      log_lengthscale_vec=log_lengthscale_vec,
      log_carrier_vec=log_carrier_vec,
      log_noise_var=log_noise_var)

  source_params = copy.copy(params["source_params"])
  del source_params["source_position"]

  radius_logscale = radius_logvar - jnp.log(jnp.clip(radius_mean, a_min=1e-4))
  radius_logshape = radius_logvar - 2 * radius_logscale

  single_source_dist_params = {
      "radius_logshape": radius_logshape,
      "radius_logscale": radius_logscale,
      "angle_logvar": angle_param_logvar
  }

  source_dist_params = jax.tree_util.tree_map(
      utils.repeat_for_pytree(n_sources), single_source_dist_params)

  source_dist_params["angle_mean"] = jnp.linspace(
      -angle_param_range, angle_param_range, n_sources).astype(float)

  source_params["source_dist"] = source_dist_params
  variational_params = copy.copy(params)
  variational_params["source_params"] = source_params

  return variational_params


def sample_source_positions(
    key: KeyArray,
    variational_params: PyTree,
    num_samples: int,
    source_length: float,
) -> Tuple[RealArray, RealArray, RealArray, RealArray, RealArray, RealArray]:
  """Returns samples and log-probs from distribution over sources' positions.

  Args:
    key: jax random key used for sample generation.
    variational_params: PyTree containing the parameters of the distribution of
      source positions.
    num_samples: number of samples to draw of each parameter.
    source_length: minimum distance between sources and microphones.

  Returns:
    A tuple (radius_params, radius, radius_log_probs, angle_params, angles,
    angle_log_probs) where:
    radius_params: [num_sources, num_samples]-jnp.ndarray containing samples of
    the radius parameter in reparametrised form.
    radius: [num_sources, num_samples]-jnp.ndarray containing samples of the
      source radius.
    radius_log_probs:  [num_sources, num_samples]-jnp.ndarray containing the
      log-probability density of each radius sample under the sampling
      distribution.
    angle_params: [num_sources, num_samples]-jnp.ndarray containing samples of
      the angle parameter in reparametrised form
    angles: [num_sources, num_samples]-jnp.ndarray containing samples of the
      source angle.
    angle_log_probs: [num_sources, num_samples]-jnp.ndarray containing the
      log-probability density of each angle sample under the sampling
      distribution.
  """

  # jax.vmap over sources in our model
  vmap_angle_sample_and_log_prob = jax.jit(
      jax.vmap(
          functools.partial(
              position_model.angle_sample_and_log_prob,
              num_samples=num_samples),
          in_axes=(0, 0, 0)))
  vmap_radius_sample_and_log_prob = jax.jit(
      jax.vmap(
          functools.partial(
              position_model.radius_sample_and_log_prob,
              num_samples=num_samples,
              source_length=source_length),
          in_axes=(0, 0, 0)))

  num_sources = len(
      variational_params["source_params"]["source_dist"]["angle_mean"])

  # 2 keys per source, one for raius and one for angle
  split_keys = jax.random.split(key, 2 * num_sources)

  radius_params, radius, radius_log_probs = vmap_radius_sample_and_log_prob(
      split_keys[num_sources:],
      variational_params["source_params"]["source_dist"]["radius_logshape"],
      variational_params["source_params"]["source_dist"]["radius_logscale"]
  )  # sizes (num_sources, num_samples),
  # (num_sources, num_samples), (num_sources, num_samples)

  angle_params, angles, angle_log_probs = vmap_angle_sample_and_log_prob(
      split_keys[:num_sources],
      variational_params["source_params"]["source_dist"]["angle_mean"],
      variational_params["source_params"]["source_dist"]["angle_logvar"]
  )  # sizes (num_sources, num_samples),
  # (num_sources, num_samples), (num_sources, num_samples)

  return radius_params, radius, radius_log_probs, angle_params, angles, angle_log_probs


def position_params_from_variational_params(
    variational_params: PyTree,
    source_length: float,
    num_samples: int = 300) -> Tuple[PyTree, RealArray]:
  """Returns deterministic position parameter tree from a variational parameter PyTree.

  Extracts position distribution means and samples from distribution over
  positions. Useful for plotting progress of optimisation over source
  positions.

  Args:
    variational_params: PyTree containing parameters for the variational
      distribution and likelihood function.
    source_length: minimum distance between sources and microphones
      num_samples: number of samples to draw.

  Returns:
    A tuple (params, source_param_samples) where:
    params: PyTree containing deterministic sources' position parameters under
      params["source_params"]["source_position"].
    source_param_samples: [num_samples, num_sources, 2]-jnp.array containing
      polar parameter samples for all sources in the model.
  """

  key = jax.random.PRNGKey(0)

  radius_params, _, _, angle_params, _, _ = sample_source_positions(
      key=key,
      variational_params=variational_params,
      num_samples=num_samples,
      source_length=source_length)

  source_param_samples = jnp.stack(
      [radius_params.T, angle_params.T],
      axis=-1)  # size (num_samples, num_sources, 2)

  radius_param = radius_params.mean(axis=1)  # shape (num_sources,)
  angle_param = angle_params.mean(axis=1)  # shape (num_sources,)

  # construct deterministic position parameter PyTree
  new_source_params = variational_params["source_params"].copy()
  new_source_params["source_position"] = jnp.stack(
      [radius_param, angle_param], axis=1)  # shape (num_sources, 2)

  del new_source_params["source_dist"]

  params = variational_params.copy()
  params["source_params"] = new_source_params

  return params, source_param_samples


## source and up-mixed signal synthesis code


@jax.jit
def build_source_self_covariance(source_params: PyTree,
                                 eval_t: RealArray) -> RealArray:
  r"""Returns self-covariance matrix for the signal emitted by a single source.

  Constructs source covariance kernel using parameters and evaluates it at all
  pairs of points contained in the provided eval_t vector to obtain Gram matrix.

  That is
  .. math::
    [K]_{ij} = k(\mathrm{eval_t}_i, \, \mathrm{eval_t}_i)

  Args:
    source_params: PyTree containing the source signal parameters for a single
      source.
    eval_t: (num_samples_eval,)-RealArray containing
      points in time at which to evaluate covariance functions.

  Returns:
    [num_samples_eval, num_samples_eval] RealArray representing source self
    covariance matrix.
  """

  spectral_mixture_kernel = kernels.gen_spectral_mixture_kernel_fn(
      kernels.matern32_cov)
  kernel_fun = jax.jit(
      functools.partial(
          spectral_mixture_kernel,
          marg_var_vec=jnp.exp(source_params["log_marg_var"]),
          lengthscale_vec=jnp.exp(source_params["log_lengthscale"]),
          carrier_vec=jnp.exp(source_params["log_carrier_freq"])))

  kxx = kernel_fun(eval_t[:, None], eval_t[None, :])
  return kxx


@jax.jit
def build_source_mic_cross_covariance(source_params: PyTree,
                                      microphone_position: RealArray,
                                      eval_t: RealArray, observed_t: RealArray,
                                      c: float,
                                      source_length: float) -> RealArray:
  r"""Returns cross covariance mat between a single source and measured signals.

  Constructs source covariance kernel using parameters, applies linear
  propagation operator, consisting of delay and attenuation that signals
  experience when transmitted to a microphone, to the right hand side of
  the kernel and evaluates the updated kernel at all pairs of points between
  the provided observed_t and eval_t vectors to obtain Gram matrix.

  That is
  .. math::
    [K]_{ij} = k(\mathrm{eval_t}_i, \, \mathrm{observed_t}_j - r c^{-1}) / r
  for the radius r between the source and microphone.

  Args:
    source_params: PyTree containing the source signal parameters for a single
      source.
    microphone_position: RealArray of size (2,) containing a microphone's
      position in cartesian coordinates (meters) stacked as [x_0, y_0].
    eval_t: (num_samples_eval,)-RealArray containing
      points in time at which to evaluate covariance functions.
    observed_t: [num_samples] RealArray containing time
      points at which we recorded signal amplitude at the stereo microphones.
    c: speed of sound (meters / second).
    source_length: minimum distance between sources and microphones (meters).

  Returns:
    [num_samples_eval, num_samples] RealArray representing
      source / stereo microphone cross covariance matrix.
  """

  r = utils.propagation_radius(
      utils.polar_params_to_cartesian(
          source_params["source_position"], source_length=source_length),
      microphone_position)

  # we work with relative delay between microphones
  delay = r / c

  spectral_mixture_kernel = kernels.gen_spectral_mixture_kernel_fn(
      kernels.matern32_cov)
  kernel_fun = jax.jit(
      functools.partial(
          spectral_mixture_kernel,
          marg_var_vec=jnp.exp(source_params["log_marg_var"]),
          lengthscale_vec=jnp.exp(source_params["log_lengthscale"]),
          carrier_vec=jnp.exp(source_params["log_carrier_freq"])))

  observed_t_delay = observed_t - delay

  return kernel_fun(eval_t[:, None], observed_t_delay[None, :]) / r


@jax.jit
def build_mic_cross_covariance(
    source_params: PyTree, microphone0_position: RealArray,
    microphone1_position: RealArray, t0: RealArray, t1: RealArray, c: float,
    source_length: float) -> RealArray:
  r"""Returns cross covariance mat between 2 mics induced by a single source.

  Constructs source covariance kernel k using parameters, applies linear
  propagation operator, consisting of the delay and attenuation that signals
  experience
  when transmitted from the source to a microphone, to the left hand side of the
  kernel for microphone0 and the right for microphone1 and
  evaluates the updated kernel at all pairs of points between the provided
  t0 (left kernel argument) and t1 (right kernel argument) vectors to obtain
  Gram matrix.

  That is
  .. math::
    [K]_{ij} =  k(t0_i - r_0 c^{-1}, \, t1_j - r_1 c^{-1}) / (r_0, r_1)
  for the radius r0 between the source and microphone 0 and r1
  the radius between the source and microphone 1.


  This function can generate the self covariance matrix of the signal recorded
  at a single microphone by passing values of
  microphone0_position = microphone1_position and t0 = t1.

  Args:
    source_params: PyTree containing the source signal parameters for a single
      source.
    microphone0_position: RealArray containing [x0, y0], first microphone's
      position (in meters).
    microphone1_position: RealArray containing [x1, y1], second microphone's
      position (in meters).
    t0: [num_samples0] RealArray containing points in
      time at which to evaluate the left argument of the covariance function.
    t1: [num_samples1] RealArray containing points in
      time at which to evaluate the right argument of the covariance function.
    c: speed of sound (meters / second).
    source_length: minimum distance between sources and microphones (meters).

  Returns:
    [num_samples0, num_samples1] RealArray representing
       cross covariance matrix between a single source's signals at two
       microphones.
  """

  r0 = utils.propagation_radius(
      utils.polar_params_to_cartesian(
          source_params["source_position"], source_length=source_length),
      microphone0_position)

  r1 = utils.propagation_radius(
      utils.polar_params_to_cartesian(
          source_params["source_position"], source_length=source_length),
      microphone1_position)

  delay0 = r0 / c
  delay1 = r1 / c

  spectral_mixture_kernel = kernels.gen_spectral_mixture_kernel_fn(
      kernels.matern32_cov)
  kernel_fun = jax.jit(
      functools.partial(
          spectral_mixture_kernel,
          marg_var_vec=jnp.exp(source_params["log_marg_var"]),
          lengthscale_vec=jnp.exp(source_params["log_lengthscale"]),
          carrier_vec=jnp.exp(source_params["log_carrier_freq"])))

  t0_delay = t0 - delay0  # size (n_samples,)
  t1_delay = t1 - delay1  # size (n_samples,)

  return kernel_fun(t0_delay[:, None], t1_delay[None, :]) / (r0 * r1)


def get_source_cov_mat_generator(
    params: PyTree, microphone_positions: RealArray, observed_t: RealArray,
    c: float,
    source_length: float) -> Callable[[RealArray], Tuple[RealArray, RealArray]]:
  """Returns function that constructs cov mats between all sources and mics.

  Args:
    params: PyTree containing the parameters for all sources in our model and
      the variance of the noise at the microphones.
    microphone_positions: Array of size (2, 2) containing
      microphone positions in cartesian coordinates (meters) stacked as [[x_0,
      y_0],[x_1, y_1]].
    observed_t: [num_samples] RealArray containing time
      points at which we recorded signal amplitude at the stereo microphones.
    c: speed of sound (meters / second).
    source_length: minimum distance between sources and microphones (meters).
  """

  # vmap self covariance generating function across sources
  get_source_self_covariance_mat_fn = jax.jit(
      jax.vmap(build_source_self_covariance, in_axes=(0, None)))

  @jax.jit
  def _gen_source_self_covariance_mats(eval_t):
    return get_source_self_covariance_mat_fn(params["source_params"], eval_t)

  # vmap cross covariance generating function across sources and microphones
  get_cross_covariance_mat_fn = jax.jit(
      jax.vmap(
          jax.vmap(
              functools.partial(
                  build_source_mic_cross_covariance,
                  observed_t=observed_t,
                  c=c,
                  source_length=source_length),
              in_axes=(0, None, None)),
          in_axes=(None, 0, None)))

  @jax.jit
  def _gen_source_observation_covariance_mats(eval_t):
    return get_cross_covariance_mat_fn(params["source_params"],
                                       microphone_positions, eval_t)

  def construct_source_cov_mats(
      eval_t: RealArray) -> Tuple[RealArray, RealArray]:
    """Returns sources' self-cov mats and cross-cov mats for source-mic pairs.

    Args:
      eval_t: (num_samples_eval,)-RealArray containing
        points in time at which to evaluate source signals' covariance
        functions.

    Returns:
      A tuple (kxy, kxx) consisting of
      kxy: [num_sources, num_samples_eval, 2 * num_samples] RealArray
        representing the cross covariance between the signal from each source
        and the stereo signal recorded at the microphones.
      kxx: [num_sources, num_samples_eval, num_samples_eval] RealArray
        representing each source's self
        covariance matrix.
    """
    kxy = _gen_source_observation_covariance_mats(eval_t=eval_t)
    # shape (num_microphones, num_sources, num_samples_eval, num_samples)

    # concatenate observed signals' covariances
    kxy = jnp.concatenate(list(kxy), axis=-1)
    # shape (num_sources, num_samples_eval, 2*num_samples)

    kxx = _gen_source_self_covariance_mats(eval_t=eval_t)
    # shape (num_sources, num_samples_eval, num_samples_eval)

    return kxy, kxx

  return jax.jit(construct_source_cov_mats)


def get_upmix_cov_mat_generator(
    params: PyTree, microphone_positions: RealArray,
    up_mix_array_positions: RealArray, observed_t: RealArray, c: float,
    source_length: float
) -> Callable[[RealArray], Tuple[RealArray, RealArray]]:
  """Returns covariance-computing function.

    The returned function constructs self and cross cov mats for all
    real-virtual mic pairs.

  Args:
    params: PyTree containing the parameters for all sources in our model and
      the variance of the noise at the microphones.
    microphone_positions: Array [[x0, y0], [x1, y1]] of recording microphone
      positions (in meters).
    up_mix_array_positions: Array of size (num_mics, 2) containing positions of
      the up-mixing, or virtual, microphones for which we will infer the
      signals, in cartesian coordinates (meters) stacked as [[x_i, y_i], ...]
      for i < num_mics.
    observed_t: [num_samples] RealArray containing time
      points at which we recorded signal amplitude at the stereo microphones.
    c: speed of sound (meters / second).
    source_length: minimum distance between sources and microphones (meters).
  """

  # vmap cross covariance matrix generating function across sources and virtual
  # microphones on both sides to obtain a function that generates virtual
  # microphone self covariance matrices

  get_mic_self_covariance_mat_fn = jax.jit(
      jax.vmap(
          jax.vmap(
              functools.partial(
                  build_mic_cross_covariance, c=c, source_length=source_length),
              in_axes=(0, None, None, None, None)),
          in_axes=(None, 0, 0, None, None)))

  @jax.jit
  def _gen_mic_self_covariance_mats(eval_t):
    return get_mic_self_covariance_mat_fn(params["source_params"],
                                          up_mix_array_positions,
                                          up_mix_array_positions, eval_t,
                                          eval_t)

  # vmap cross covariance generating function across sources, up-mixing
  # microphones and real microphones
  get_mic_cross_covariance_mat_fn = jax.jit(
      jax.vmap(
          jax.vmap(
              jax.vmap(
                  functools.partial(
                      build_mic_cross_covariance,
                      t1=observed_t,
                      c=c,
                      source_length=source_length),
                  in_axes=(0, None, None, None)),
              in_axes=(None, 0, None, None)),
          in_axes=(None, None, 0, None)))

  @jax.jit
  def _gen_mic_observation_covariance_mats(eval_t):
    return get_mic_cross_covariance_mat_fn(params["source_params"],
                                           up_mix_array_positions,
                                           microphone_positions, eval_t)

  def construct_upmix_cov_mats(
      eval_t: RealArray) -> Tuple[RealArray, RealArray]:
    """Returns virtual mics' self and cross cov mats for real-virtual mic pairs.

    Args:
      eval_t: (num_samples_eval,)-RealArray containing
        points in time at which to evaluate virtual microphone signals'
        covariance functions.

    Returns:
      A tuple (kmy, kmm) consisting of
      kmy: [num_sources, num_samples_eval, 2 * num_samples] RealArray
      representing the cross covariance between the signal at each virtual
      microphone and the stereo signal recorded at the real microphones.
        source / stereo microphone cross covariance matrix.
      kmm: [num_sources, num_samples_eval, num_samples_eval] RealArray
        representing each virtual microphone's self
        covariance matrix.
    """
    kmy = _gen_mic_observation_covariance_mats(eval_t=eval_t)
    # shape (num_recording_microphones, num_virtual_microphones, num_sources,
    # num_samples_eval, num_samples)

    # sum over sources since the signals from them are assumed independent
    kmy = kmy.sum(
        axis=2
    )  # shape (num_recording_microphones, num_virtual_microphones,
       # num_samples_eval, num_samples)

    # concatenate observed signals' covariances
    kmy = jnp.concatenate(list(kmy),
                          axis=-1)
    # shape (num_virtual_microphones, num_samples_eval, 2*num_samples)

    kmm = _gen_mic_self_covariance_mats(eval_t=eval_t)
    # shape (num_virtual_microphones, num_sources, num_samples_eval,
    # num_samples_eval)

    # sum over sources since the signals from them are assumed independent
    kmm = kmm.sum(
        axis=1
    )
    # shape (num_virtual_microphones, num_samples_eval, num_samples_eval)

    return kmy, kmm

  return jax.jit(construct_upmix_cov_mats)


@jax.jit
def gp_reconstruction(y: RealArray, kyy: RealArray,
                      kxy: RealArray) -> RealArray:
  """Returns (optionally batched) predicted mean of Gaussian process regression.

  Computes:
  .. math::
    kxy kyy^{-1} y

  Args:
    y: [num_samples_y] RealArray consisting of the observed signal
    kyy: [num_samples_y, num_samples_y] RealArray consisting of self covariance
      matrix of observed signals.
    kxy: [num_samples_x, num_samples_y] or [num_tergets, num_samples_x,
      num_samples_y] RealArray consisting of cross covariance matrix between
      signals we want to predict at (each) target location and observed signals.

  Returns:
    A [num_samples_x] or [num_targets, num_samples_x] RealArray containing
    the posterior mean for the signal(s) we want to predict.
  """
  solve = jax.scipy.linalg.solve(kyy, b=y, assume_a="pos")
  return kxy @ solve


@jax.jit
def batched_gp_predictive_covariance(kyy: RealArray, kxx: RealArray,
                                     kxy: RealArray) -> RealArray:
  """Returns batched predictive covariance of Gaussian process regression.

  Computes:
  .. math::
    kxx - kxy kyy^{-1} kxy^T

  Args:
    kyy: [num_samples_y, num_samples_y] RealArray consisting of self covariance
      matrix of observed signals.
    kxx: [num_targets, num_samples_x, num_samples_x] RealArray consisting of
      prior self covariance matrix over over signals we want to predict at each
      target location.
    kxy: [num_tergets, num_samples_x, num_samples_y] RealArray consisting of
      cross covariance matrix between signals we want to predict at each target
      location and observed signals.

  Returns:
    A [num_targets, num_samples_x, num_samples_x] RealArray containing the
    posterior self covariance for the signals we want to predict at
    each target location.
  """

  def _solve(kyx):
    """Closure of linear solve that allows for easy vmapping."""
    return jax.scipy.linalg.solve(kyy, b=kyx, assume_a="pos")

  return kxx - kxy @ jax.jit(jax.vmap(_solve))(kxy.transpose((0, 2, 1)))


@jax.jit
def batched_gp_reconstruction_and_predictive_covariance(
    y: RealArray, kyy: RealArray, kxx: RealArray,
    kxy: RealArray) -> Tuple[RealArray, RealArray]:
  """Returns batched predicted mean & covariance of Gaussian process regression.

  .. math::
    kxy kyy^{-1} y
    kxx - kxy kyy^{-1} kxy^T

  Args:
    y: [num_samples_y] RealArray consisting of the observed signal
    kyy: [num_samples_y, num_samples_y] RealArray consisting of self covariance
      matrix of observed signals.
    kxx: [num_targets, num_samples_x, num_samples_x] RealArray consisting of
      prior self covariance matrix over over signals we want to predict at each
      target location.
    kxy: [num_tergets, num_samples_x, num_samples_y] RealArray consisting of
      cross covariance matrix between signals we want to predict at each target
      location and observed signals.

  Returns:
    A tuple (mu, cov) where
    mu:  A [num_targets, num_samples_x] RealArray containing the posterior
      mean for the signals we want to predict.
    cov: A [num_targets, num_samples_x, num_samples_x] RealArray containing the
      posterior self covariance for the signals we want to predict at each
    target location.
  """
  f = gp_reconstruction(y, kyy, kxy)
  cov = batched_gp_predictive_covariance(kyy, kxx, kxy)
  return f, cov


def gen_reconstruct_sources_fns(
    params: PyTree, microphone_positions: RealArray, y_stack: RealArray,
    observed_t: RealArray, c: float, source_length: float, noise_eps: float
) -> Tuple[SignalPredictor, SignalPredictor, SignalPredictor]:
  """Returns functions to reconstruct sources' signals with uncertainty.

  The three returned functions are ordered in terms of how much information they
  provide on reconstruction uncertainty and thus computational cost.

  Args:
    params: PyTree containing the parameters for all sources in our model and
      the variance of the noise at the microphones.
    microphone_positions: Array of size (2, 2) containing
      microphone positions in cartesian coordinates (meters) stacked as [[x_0,
      y_0],[x_1, y_1]].
    y_stack: [2*num_samples] RealArray containing the concatenation of the
      signals observed at the stereo recording microphones.
    observed_t: [num_samples] RealArray containing time
      points at which we recorded signal amplitude at the stereo microphones.
    c: speed of sound (meters / second).
    source_length: minimum distance between sources and microphones (meters).
    noise_eps: constant that is added to the diagonal of the observation
      covariance matrix to ensure it is PD. This represents the minimum noise
      floor.

  Returns:
    A tuple (reconstruct_source_signals,
    reconstruct_source_signals_with_block_uncertainty,
    reconstruct_source_signals_with_full_uncertainty) where
    reconstruct_source_signals returns the mean prediction for each source's
    signal.

    reconstruct_source_signals_with_block_uncertainty returns the mean
    prediction for each source's signal and its self-covariance.

    reconstruct_source_signals_with_full_uncertainty returns the mean prediction
    for each source's signal and the full covariance matrix capturing
    interactions among the reconstructed signals for all sources.
  """

  construct_source_cov_mats = get_source_cov_mat_generator(
      params, microphone_positions, observed_t, c, source_length)

  kyy = build_mic_covariance(params, microphone_positions, observed_t, c,
                             source_length,
                             noise_eps)  # size (2 num_samples, 2 num_samples)

  def reconstruct_source_signals(eval_t: RealArray) -> RealArray:
    """Returns the mean prediction at all sources.

    Args:
      eval_t: A (num_samples_eval,)-RealArray containing the timepoints at which
        to evaluate the source signals.

    Returns:
      A [num_sources, num_samples_eval] RealArray.
    """
    kxy, _ = construct_source_cov_mats(eval_t)
    return gp_reconstruction(y_stack, kyy, kxy)

  def reconstruct_source_signals_with_block_uncertainty(
      eval_t: RealArray) -> Tuple[RealArray, RealArray]:
    """Returns mean prediction and source-wise covariance for source's signals.

    Args:
      eval_t: A (num_samples_eval,)-RealArray containing the timepoints at which
        to evaluate the source signals and their covariance.

    Returns:
      A tuple (mu, cov) where
      mu: A [num_sources, num_samples_eval] RealArray containing the
        mean predicted signal for each source.
      cov:  A [num_sources, num_samples_eval, num_samples_eval]
        RealArray containing the self-covariance matrix for the signal predicted
        at each source.
    """
    kxy, kxx = construct_source_cov_mats(eval_t)
    return batched_gp_reconstruction_and_predictive_covariance(
        y_stack, kyy, kxx, kxy)

  def reconstruct_source_signals_with_full_uncertainty(
      eval_t: RealArray) -> Tuple[RealArray, RealArray]:
    """Returns the mean prediction and full covariance among all source signals.

    Args:
      eval_t: A (num_samples_eval,)-RealArray containing the timepoints at which
        to evaluate the source signals and their covariance.

    Returns:
      A tuple (mu, cov) where
      mu: A [num_sources, num_samples_eval] RealArray containing the
        mean predicted signal for each source.
      cov:  A [num_sources*num_samples_eval,
        num_sources*num_samples_eval] RealArray containing the
        covariance matrix for the signal formed by concatenating the signals
        predicted at each source. This matrix includes each
        source's self covariance in the block diagonal entries and
        cross covariances in the off-diagonal blocks.
    """
    kxy, kxx = construct_source_cov_mats(eval_t)
    # stack cross covariance matrices
    kxy_full = jnp.concatenate([k for k in kxy], axis=0)
    # Construct prior covariance over source signals as a block diagonal
    # matrix.
    kxx_full = jax.scipy.linalg.block_diag(*[k for k in kxx])
    mean, cov = batched_gp_reconstruction_and_predictive_covariance(
        y_stack, kyy, kxx_full[jnp.newaxis, ...], kxy_full[jnp.newaxis, ...])
    return jnp.squeeze(mean), jnp.squeeze(cov)

  return jax.jit(reconstruct_source_signals), jax.jit(
      reconstruct_source_signals_with_block_uncertainty), jax.jit(
          reconstruct_source_signals_with_full_uncertainty)


def gen_upmixing_fns(
    params: PyTree, microphone_positions: RealArray,
    up_mix_array_positions: RealArray, y_stack: RealArray,
    observed_t: RealArray, c: float, source_length: float, noise_eps: float
) -> Tuple[SignalPredictor, SignalPredictor, SignalPredictor]:
  """Returns functions to up-mix signals with uncertainty.

  The three returned functions are ordered in terms of how much information they
  provide on reconstruction uncertainty and thus computational cost.

  Args:
    params: PyTree containing the parameters for all sources in our model and
      the variance of the noise at the microphones.
    microphone_positions: Array [[x0, y0], [x1, y1]] of recording microphone
      positions (in meters).
    up_mix_array_positions: Array of size (num_mics, 2) containing positions of
      the up-mixing, or virtual, microphones for which we will infer the
      signals, in cartesian coordinates (meters) stacked as [[x_i, y_i], ...]
      for i < num_mics.
    y_stack: [2*num_samples] RealArray containing the concatenation of the
      signals observed at the stereo recording microphones.
    observed_t: [num_samples] RealArray containing time
      points at which we recorded signal amplitude at the stereo microphones.
    c: speed of sound (meters / second).
    source_length: minimum distance between sources and microphones (meters).
    noise_eps: constant that is added to the diagonal of the observation
      covariance matrix to ensure it is PD. This represents the minimum noise
      floor.

  Returns:
    A tuple (upmixed_signals, upmixed_signals_with_block_uncertainty,
      upmixed_signals_with_full_uncertainty) where

    upmixed_signals returns the mean prediction for each virtual microphone's
    signal.

    upmixed_signals_with_block_uncertainty returns the mean prediction for each
    virtual microphone's signal and its self-covariance.

    upmixed_signals_with_full_uncertainty returns the mean prediction for each
    virtual microphone's signal and the full covariance matrix capturing
    interactions among the reconstructed signals for all virtual microphones.
  """

  construct_upmix_cov_mats = get_upmix_cov_mat_generator(
      params, microphone_positions, up_mix_array_positions, observed_t, c,
      source_length)

  kyy = build_mic_covariance(params, microphone_positions, observed_t, c,
                             source_length,
                             noise_eps)  # size (2 num_samples, 2 num_samples)

  def upmixed_signals(eval_t: RealArray) -> RealArray:
    """Returns the mean prediction at all virtual microphones.

    Args:
      eval_t: A (num_samples_eval,)-RealArray containing the timepoints at which
        to evaluate the up-mixed signals.

    Returns:
      A (num_virtual_microphones, num_samples_eval)-RealArray containing the
        mean predicted signal for each virtual microphone.
    """
    kmy, _ = construct_upmix_cov_mats(eval_t)
    return gp_reconstruction(y_stack, kyy, kmy)

  def upmixed_signals_with_block_uncertainty(
      eval_t: RealArray) -> Tuple[RealArray, RealArray]:
    """Returns the mean prediction and mic-wise covariance for all virtual mics.

    Args:
      eval_t: A (num_samples_eval,)-RealArray containing the timepoints at which
        to evaluate the up-mixed signals and their covariance.

    Returns:
      A tuple (mu, cov) where
      mu: A (num_virtual_microphones, num_samples_eval)-RealArray containing the
        mean predicted signal for each virtual microphone.
      cov:  A [num_virtual_microphones, num_samples_eval, num_samples_eval]
        RealArray containing the self-covariance matrix for the signal predicted
        at each virtual microphone.
    """
    kmy, kmm = construct_upmix_cov_mats(eval_t)
    return batched_gp_reconstruction_and_predictive_covariance(
        y_stack, kyy, kmm, kmy)

  def upmixed_signals_with_full_uncertainty(
      eval_t: RealArray) -> Tuple[RealArray, RealArray]:
    """Returns the mean prediction and full covariance among all virtual mics.

    Args:
      eval_t: A (num_samples_eval,)-RealArray containing the timepoints at which
        to evaluate the up-mixed signals and their covariance.

    Returns:
      A tuple (mu, cov) where
      mu: A (num_virtual_microphones, num_samples_eval)-RealArray containing the
        mean predicted signal for each virtual microphone.
      cov:  A [num_virtual_microphones*num_samples_eval,
        num_virtual_microphones*num_samples_eval] RealArray containing the
        covariance matrix for the signal formed by concatenating the signals
        predicted at each virtual microphone. This matrix includes each
        microphone's signal's self covariance in the block diagonal entries and
        cross covariances in the off-diagonal blocks.
    """
    kmy, kmm = construct_upmix_cov_mats(eval_t)
    # stack cross covariance matrices
    kmy_full = jnp.concatenate([k for k in kmy], axis=0)
    # Construct prior covariance over up-mixed signals as a block diagonal
    # matrix.
    kmm_full = jax.scipy.linalg.block_diag(*[k for k in kmm])
    mean, cov = batched_gp_reconstruction_and_predictive_covariance(
        y_stack, kyy, kmm_full[jnp.newaxis, ...], kmy_full[jnp.newaxis, ...])
    return jnp.squeeze(mean), jnp.squeeze(cov)

  return jax.jit(upmixed_signals), jax.jit(
      upmixed_signals_with_block_uncertainty), jax.jit(
          upmixed_signals_with_full_uncertainty)
