"""TODO(antoran): DO NOT SUBMIT without one-line documentation for gp_inference_fast.

TODO(antoran): DO NOT SUBMIT without a detailed description of gp_inference_fast.
"""

import copy
import functools
from typing import Any, Callable, Sequence, Tuple, TypeVar, Union

import jax
from jax import numpy as jnp
from jax._src import prng
from jax._src import typing as jaxtyping

from google3.util.compression.korvapuusti.up_mixing.src import kernels
from google3.util.compression.korvapuusti.up_mixing.src import position_model
from google3.util.compression.korvapuusti.up_mixing.src import utils
# from google3.util.compression.korvapuusti.up_mixing.gp_inference import

RealArray = jaxtyping.ArrayLike
IntegerArray = jaxtyping.ArrayLike
KeyArray = Union[jaxtyping.Array, prng.PRNGKeyArray]
Shape = Sequence[int]
PyTree = Union[dict[Any, Any], list[Any], jaxtyping.ArrayLike]

ModelEvidence = TypeVar("ModelEvidence", bound=Callable[[PyTree], float])
SignalPredictor = TypeVar(
    "SignalPredictor", bound=Callable[[RealArray], Tuple[RealArray, RealArray]])





def gen_stereo_block_toeplitz_matvec_fn(filter00, filter01, filter10, filter11):

  @jax.jit
  def irfft(x):
    return jnp.fft.irfft(
        x, n=filter00.shape[-1] * 2 - 1, axis=-1)[..., :filter00.shape[-1]]

  new_dim_idxs = list(jnp.arange(filter00.ndim - 1).astype(int))

  def _matvec(vec):

    vec_length_half = int(len(vec) / 2)

    padded_vec0 = jnp.concatenate(
        [vec[:vec_length_half],
         jnp.zeros(vec_length_half - 1)], axis=0)
    padded_vec1 = jnp.concatenate(
        [vec[vec_length_half:],
         jnp.zeros(vec_length_half - 1)], axis=0)

    fft_vec0 = jnp.fft.rfft(padded_vec0)
    fft_vec1 = jnp.fft.rfft(padded_vec1)

    expanded_fft_vec0 = jnp.expand_dims(fft_vec0, axis=new_dim_idxs)
    expanded_fft_vec1 = jnp.expand_dims(fft_vec1, axis=new_dim_idxs)

    matvec0 = irfft(filter00 * expanded_fft_vec0) + irfft(
        filter01 * expanded_fft_vec1)
    matvec1 = irfft(filter10 * expanded_fft_vec0) + irfft(
        filter11 * expanded_fft_vec1)

    return jnp.concatenate([matvec0, matvec1], axis=-1)

  return jax.jit(_matvec)




def kernel_first_column_and_row(t, delay, kernel_fun):

  # Rectify delay such that a possitive delay is always applied to one of the
  # axis
  delay_0 = jnp.clip(delay, a_min=0)
  delay_1 = jnp.clip(-delay, a_min=0)

  t0 = t - delay_0  # size (num_samples,)
  t1 = t - delay_1

  column = kernel_fun(t0, t1[0])  # size (num_samples, num_samples)
  row = kernel_fun(t0[0], t1)

  return column, row




@jax.jit
def get_reflected_FFT_filters(first_col,
                              first_row):
  mirrored_k = jnp.concatenate([first_col, first_row[1:][::-1]], axis=0)

  # filter for transpose matrix looks the same except elements 1: are reversed
  mirrored_k_transpose = jnp.concatenate([mirrored_k[:1],
                                          mirrored_k[1:][::-1]], axis=0)
  return jnp.fft.rfft(mirrored_k), jnp.fft.rfft(mirrored_k_transpose)


@jax.jit
def single_source_unweighed_base_FFT_kernels(microphone_positions,
                                            source_length,
                                            time_vec,
                                            source_position,
                                            log_lengthscales,
                                            log_carrier_freqs,
                                            c):

  vmap_get_reflected_fft_filters = jax.jit(
      jax.vmap(get_reflected_FFT_filters, in_axes=(0, 0)) )

  spectral_mixture_kernel = kernels.gen_spectral_mixture_kernel_fn(
      kernels.matern32_cov, sum_spectral_components=False)

  r0 = utils.propagation_radius(
      utils.polar_params_to_cartesian(
          source_position, source_length=source_length),
      microphone_positions[0])
  r1 = utils.propagation_radius(
      utils.polar_params_to_cartesian(
          source_position, source_length=source_length),
      microphone_positions[1])

  # we work with relative delay between microphones
  relative_delay = (r0 - r1) / c

  kernel = jax.jit(
      functools.partial(
          spectral_mixture_kernel,
          marg_var_vec=jnp.ones((log_lengthscales.shape)),
          lengthscale_vec=jnp.exp(log_lengthscales),
          carrier_vec=jnp.exp(log_carrier_freqs)))

  get_kernel_first_column_and_row = jax.jit(
      functools.partial(kernel_first_column_and_row, kernel_fun=kernel))

  (self_cov_first_columns,
   self_cov_first_rows) = get_kernel_first_column_and_row(time_vec, 0)
  # (num_filters, len(t0))  (num_filters, len(t1))

  (cross_cov_first_columns,
   cross_cov_first_rows) = get_kernel_first_column_and_row(
       time_vec, relative_delay)
  # this is for K01 -- shape: num_filters, len(t0), len(t1)

  self_cov_filters, _ = vmap_get_reflected_fft_filters(
      self_cov_first_columns, self_cov_first_rows)

  cross01_cov_filters, cross10_cov_filters = vmap_get_reflected_fft_filters(
      cross_cov_first_columns, cross_cov_first_rows)


  return self_cov_filters / (r0**2), cross01_cov_filters / (
      r0 * r1), cross10_cov_filters / (r0 * r1), self_cov_filters / (
          r1**2)


def gen_regularised_kernel_matvec_fn(unweighed_filters00,
                              unweighed_filters01,
                              unweighed_filters10,
                              unweighed_filters11,
                              log_marg_var_params,
                              noise_var):

  filter00 = (unweighed_filters00 *
              jnp.exp(log_marg_var_params)[..., None]).sum(axis=(0, 1))
  filter01 = (unweighed_filters01 *
              jnp.exp(log_marg_var_params)[..., None]).sum(axis=(0, 1))
  filter10 = (unweighed_filters10 *
              jnp.exp(log_marg_var_params)[..., None]).sum(axis=(0, 1))
  filter11 = (unweighed_filters11 *
              jnp.exp(log_marg_var_params)[..., None]).sum(axis=(0, 1))

  # Gaussian noise is flat in frequency
  regularised_filter00 = filter00 +  noise_var
  regularised_filter11 = filter11 +  noise_var
  return gen_stereo_block_toeplitz_matvec_fn(regularised_filter00, filter01,
                                             filter10, regularised_filter11)
