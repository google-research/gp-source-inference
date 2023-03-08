"""Tests for parameter initialisation, learning & reconstruction with GP model.

This file is split into three test classes ParameterGenerationTest,
DeterministicPossitionTest, StochasticPossitionTest. The functions tested in
each of these classes depend on the ones tested in the previous class.
Additionally, all functions may depend on functions from src/utils and
src/position_models. In this sense, the tests implemented in this file should be
seen as integration tests rather than unit tests.
"""

import functools
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from google3.util.compression.korvapuusti.up_mixing.src import gp_inference
from google3.util.compression.korvapuusti.up_mixing.src import kernels
from google3.util.compression.korvapuusti.up_mixing.src import testbench
from google3.util.compression.korvapuusti.up_mixing.src import utils


def get_random_variational_params(key, num_sources, n_mixture_components,
                                  log_noise_var):
  """Randomly initialises variational params PyTree."""
  split_keys = jax.random.split(key, 7)
  log_marg_var_vec = jax.random.normal(
      split_keys[0], shape=(n_mixture_components,))
  log_lengthscale_vec = jax.random.normal(
      split_keys[1], shape=(n_mixture_components,))
  log_carrier_vec = jax.random.normal(
      split_keys[2], shape=(n_mixture_components,))
  radius_mean = jax.random.normal(
      split_keys[3])
  radius_logvar = jax.random.normal(
      split_keys[4])
  angle_param_range = jax.random.normal(
      split_keys[5])
  angle_param_logvar = jax.random.normal(
      split_keys[6])
  return gp_inference.initialise_spectral_mixture_variational_params(
      num_sources,
      radius_mean,
      radius_logvar,
      angle_param_range,
      angle_param_logvar,
      log_marg_var_vec,
      log_lengthscale_vec,
      log_carrier_vec,
      log_noise_var=log_noise_var)


def gen_fixed_params_spectral_mixture_kernel_fn(
    base_kernel_name, source_params):
  """Returns spectral mixture kernel function with fixed parameter."""
  if base_kernel_name == "rbf":
    base_kernel_fn = kernels.rbf_cov
  elif base_kernel_name == "matern12_cov":
    base_kernel_fn = kernels.matern12_cov
  elif base_kernel_name == "matern32_cov":
    base_kernel_fn = kernels.matern32_cov
  elif base_kernel_name == "matern52_cov":
    base_kernel_fn = kernels.matern52_cov
  else:
    raise ValueError("Only values supported for base_kernel_name"
                     " are [rbf, matern12_cov, matern52_cov, matern72_cov].")
  spectral_mixture_kernel = kernels.gen_spectral_mixture_kernel_fn(
      base_kernel_fn)
  return jax.jit(
      functools.partial(
          spectral_mixture_kernel,
          marg_var_vec=jnp.exp(source_params["log_marg_var"]),
          lengthscale_vec=jnp.exp(source_params["log_lengthscale"]),
          carrier_vec=jnp.exp(source_params["log_carrier_freq"])))


def get_random_params(key, num_sources, n_mixture_components, source_length,
                      log_noise_var):
  """Returns randomly initialised source parameter PyTree."""
  key = jax.random.PRNGKey(0)
  variational_params = get_random_variational_params(
      key,
      num_sources,
      n_mixture_components,
      log_noise_var=log_noise_var)
  (params, _) = gp_inference.position_params_from_variational_params(
      variational_params, source_length, 100)
  return params


def sample_functions(k, n_samples=100):
  """Samples function from Gaussian process with given kernel."""
  key = jax.random.PRNGKey(0)
  test_indices = jax.random.normal(key, shape=(k.shape[0], n_samples))
  chol = jnp.linalg.cholesky(k)
  return (chol @ test_indices).T  # shape (n_samples, k.shape[0])


class ParameterGenerationTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for parameter initialisation and manipulation functions."""
  source_model_size_list = [
      dict(
          testcase_name="single_source_single_component",
          num_sources=1,
          n_mixture_components=1,
          ),
      dict(
          testcase_name="multiple_sources_single_component",
          num_sources=4,
          n_mixture_components=1,
          ),
      dict(
          testcase_name="single_source_multiple_components",
          num_sources=1,
          n_mixture_components=4,
          ),
      dict(
          testcase_name="multiple_sources_multiple_components",
          num_sources=4,
          n_mixture_components=4,
      ),
  ]
  log_noise_var = np.log(1e-3)

  @parameterized.named_parameters(*source_model_size_list)
  def test_initialise_spectral_mixture_params(self, num_sources,
                                              n_mixture_components):
    """Checks the sizes of initialised parameter arrays are correct."""
    key = jax.random.PRNGKey(0)
    split_keys = jax.random.split(key, 5)
    log_marg_var_vec = jax.random.normal(
        split_keys[0], shape=(n_mixture_components,))
    log_lengthscale_vec = jax.random.normal(
        split_keys[1], shape=(n_mixture_components,))
    log_carrier_vec = jax.random.normal(
        split_keys[2], shape=(n_mixture_components,))
    radius_param = jax.random.normal(
        split_keys[3])
    angle_param_range = jax.random.normal(
        split_keys[4])
    params = gp_inference.initialise_spectral_mixture_params(
        num_sources,
        radius_param,
        angle_param_range,
        log_marg_var_vec,
        log_lengthscale_vec,
        log_carrier_vec,
        log_noise_var=self.log_noise_var)
    self.assertAllEqual(params["source_params"]["source_position"].shape,
                        (num_sources, 2))
    self.assertAllEqual(params["source_params"]["log_lengthscale"].shape,
                        (num_sources, n_mixture_components))
    self.assertAllEqual(params["source_params"]["log_marg_var"].shape,
                        (num_sources, n_mixture_components))
    self.assertAllEqual(params["source_params"]["log_carrier_freq"].shape,
                        (num_sources, n_mixture_components))
    self.assertAllEqual(params["log_noise_var"], self.log_noise_var)

  @parameterized.named_parameters(*source_model_size_list)
  def test_initialise_spectral_mixture_variational_params(
      self, num_sources, n_mixture_components):
    """Checks the sizes of initialised variational parameter arrays are correct."""
    key = jax.random.PRNGKey(0)
    params = get_random_variational_params(
        key,
        num_sources,
        n_mixture_components,
        log_noise_var=self.log_noise_var)
    self.assertAllEqual(
        params["source_params"]["source_dist"]["radius_logshape"].shape,
        (num_sources,))
    self.assertAllEqual(
        params["source_params"]["source_dist"]["radius_logscale"].shape,
        (num_sources,))
    self.assertAllEqual(
        params["source_params"]["source_dist"]["angle_mean"].shape,
        (num_sources,))
    self.assertAllEqual(
        params["source_params"]["source_dist"]["angle_logvar"].shape,
        (num_sources,))
    self.assertNotIn("source_position", params["source_params"].keys())
    self.assertAllEqual(params["source_params"]["log_lengthscale"].shape,
                        (num_sources, n_mixture_components))
    self.assertAllEqual(params["source_params"]["log_marg_var"].shape,
                        (num_sources, n_mixture_components))
    self.assertAllEqual(params["source_params"]["log_carrier_freq"].shape,
                        (num_sources, n_mixture_components))
    self.assertAllEqual(params["log_noise_var"], self.log_noise_var)

  @parameterized.named_parameters(*source_model_size_list)
  def test_sample_source_positions(self, num_sources, n_mixture_components):
    """Checks correctness of the shape and the domain of the sampled data."""
    key = jax.random.PRNGKey(0)
    variational_params = get_random_variational_params(
        key,
        num_sources,
        n_mixture_components,
        log_noise_var=self.log_noise_var)
    source_length = 2.1
    num_samples = 100
    (radius_params, radius, radius_log_probs, angle_params, angles,
     angle_log_probs) = gp_inference.sample_source_positions(
         key, variational_params, num_samples, source_length=source_length)
    self.assertAllEqual(radius_params.shape,
                        (num_sources, num_samples))
    self.assertAllEqual(radius.shape,
                        (num_sources, num_samples))
    self.assertAllEqual(radius_log_probs.shape,
                        (num_sources, num_samples))
    self.assertAllEqual(angle_params.shape,
                        (num_sources, num_samples))
    self.assertAllEqual(angles.shape,
                        (num_sources, num_samples))
    self.assertAllEqual(angle_log_probs.shape,
                        (num_sources, num_samples))
    self.assertAllGreaterEqual(radius, source_length-1e-3)
    self.assertAllGreaterEqual(angles, 0.)

  @parameterized.named_parameters(*source_model_size_list)
  def test_position_params_from_variational_params(self, num_sources,
                                                   n_mixture_components):
    """Checks correctness of shapes of returned parameter PyTree and samples."""
    key = jax.random.PRNGKey(0)
    variational_params = get_random_variational_params(
        key,
        num_sources,
        n_mixture_components,
        log_noise_var=self.log_noise_var)
    source_length = 2.1
    num_samples = 100
    (params, source_param_samples
    ) = gp_inference.position_params_from_variational_params(
        variational_params, source_length, num_samples)
    self.assertAllEqual(source_param_samples.shape,
                        (num_samples, num_sources, 2))
    self.assertAllEqual(params["source_params"]["source_position"].shape,
                        (num_sources, 2))


class DeterministicPossitionTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for observation signal model with deterministic source positions."""
  delay_vector = [
      dict(
          testcase_name="positive_delay",
          delay=2.3e-3,
          num_sources=1,
          n_mixture_components=1,
      ),
      dict(
          testcase_name="negative_delay",
          delay=-2.3e-3,
          num_sources=1,
          n_mixture_components=1,
      ),
  ]
  mic_distance = 1.
  microphone_positions = np.array(([-mic_distance / 2,
                                    0.], [mic_distance / 2, 0.]))
  source_length = 0.1
  sampling_freq = 1000
  sample_time = 1. / sampling_freq
  window_time = 30e-3
  time = np.arange(0, window_time, sample_time)
  num_samples = len(time)
  freq_array = np.arange(
      0,
      int(num_samples / 2 +
          1)) * sampling_freq / num_samples  # Continuous time frequency index
  log_noise_var = np.log(
      1e-3)  # small value added to diagonals to ensure PSD matrices

  @parameterized.named_parameters(*delay_vector)
  def test_construct_delayed_kernel_mat(self, delay, num_sources,
                                        n_mixture_components):
    """Checks the structure of source auto/cross covariance matrices."""
    params = get_random_params(0, num_sources, n_mixture_components,
                               self.source_length, self.log_noise_var)
    params["source_params"][
        "log_lengthscale"] = params["source_params"]["log_lengthscale"] - 3
    kernel_fun = gen_fixed_params_spectral_mixture_kernel_fn(
        base_kernel_name="rbf",
        source_params=params["source_params"])
    k00, k01 = gp_inference.construct_delayed_kernel_mat(
        self.time, kernel_fun, delay)
    self.assertAllClose(k00, k00.T)
    self.assertFalse(jnp.any(jnp.isnan(k00)))
    self.assertFalse(jnp.any(jnp.isnan(k01)))
    if delay != 0:
      self.assertNotAllClose(k01, k01.T)
    else:
      self.assertAllClose(k01, k01.T)
    # the delay is reflected in the cross covariance maximum value diagonal band
    diag_indices = jnp.arange(self.num_samples)
    if delay > 0:
      self.assertAllLess((k01.argmax(axis=1)-diag_indices).mean(), 0)
    elif delay < 0:
      self.assertAllGreater((k01.argmax(axis=1)-diag_indices).mean(), 0)

  @parameterized.named_parameters(*delay_vector)
  def test_build_single_source_mic_covariance(self, delay, num_sources,
                                              n_mixture_components):
    """Checks the structure of the microphone signals covariance matrix."""
    del delay  # delete to suppress unused arg warning
    params = get_random_params(0, num_sources, n_mixture_components,
                               self.source_length, self.log_noise_var)
    # ensure lengthscale is small relative to problem timescale
    params["source_params"][
        "log_lengthscale"] = params["source_params"]["log_lengthscale"] - 4
    # we vmap since entries of params["source_params"] are shaped (num_sources,
    # ...) but build_single_source_mic_covariance expects an array of parameters
    # for a single source, i.e. shape(...)
    kyy = jax.jit(
        jax.vmap(
            functools.partial(
                gp_inference.build_single_source_mic_covariance,
                microphone_positions=self.microphone_positions,
                observed_t=self.time,
                c=343.,
                source_length=self.source_length)))(
                    source_params=params["source_params"]).sum(axis=0)
    self.assertFalse(jnp.any(jnp.isnan(kyy)))
    self.assertAllEqual(kyy.shape,
                        (2*self.num_samples, 2*self.num_samples))
    self.assertEqual(
        jnp.linalg.matrix_rank(kyy + jnp.eye(2 * self.num_samples) *
                               jnp.exp(self.log_noise_var)),
        2 * self.num_samples)
    ky0y0 = kyy[:self.num_samples, :self.num_samples]
    ky1y1 = kyy[self.num_samples:, self.num_samples:]
    ky0y1 = kyy[:self.num_samples, self.num_samples:]
    ky1y0 = kyy[self.num_samples:, :self.num_samples]
    # The ratio between diagonal entries of block diagonal matrices should match
    # ratio between other block diagonal entries
    self.assertAllClose(jnp.diag(ky0y0 / ky1y1).mean(), (ky0y0 / ky1y1)[0, 3])
    # Off-block-diagonals should be equal to each other transposed
    self.assertAllClose(ky1y0, ky0y1.T)

  @parameterized.named_parameters(*delay_vector)
  def test_build_mic_covariance(self, delay, num_sources, n_mixture_components):
    """Checks the structure of the microphone signals covariance matrix."""
    del delay  # delete to suppress unused arg warning
    params = get_random_params(0, num_sources, n_mixture_components,
                               self.source_length, self.log_noise_var)
    # ensure lengthscale is small relative to problem timescale
    params["source_params"][
        "log_lengthscale"] = params["source_params"]["log_lengthscale"] - 4
    kyy = gp_inference.build_mic_covariance(
        params=params,
        microphone_positions=self.microphone_positions,
        observed_t=self.time,
        c=343.,
        source_length=self.source_length,
        noise_eps=1e-7)
    self.assertFalse(jnp.any(jnp.isnan(kyy)))
    self.assertAllEqual(kyy.shape,
                        (2*self.num_samples, 2*self.num_samples))
    self.assertEqual(
        jnp.linalg.matrix_rank(kyy + jnp.eye(2 * self.num_samples) *
                               jnp.exp(self.log_noise_var)),
        2 * self.num_samples)
    ky0y0 = kyy[:self.num_samples, :self.num_samples]
    ky1y1 = kyy[self.num_samples:, self.num_samples:]
    ky0y1 = kyy[:self.num_samples, self.num_samples:]
    ky1y0 = kyy[self.num_samples:, :self.num_samples]
    # The diagonal elements of block diagonal matrices should differ
    # from each other by a constant
    self.assertAllClose((ky0y0 / ky1y1)[0, 7], (ky0y0 / ky1y1)[12, 2])
    # Off-block-diagonals should be equal to each other transposed
    self.assertAllClose(ky1y0, ky0y1.T)

  def test_gp_evidence(self):
    """Checks the return type of the multivariate Normal log prob density."""
    ncols = 100
    a = jax.random.normal(jax.random.PRNGKey(0), shape=(ncols, ncols))
    a_sym = a @ a.T + jnp.eye(ncols) * jnp.exp(self.log_noise_var)
    v = jax.random.normal(jax.random.PRNGKey(0), shape=(ncols,))
    evidence = gp_inference.gp_evidence(v, a_sym)
    self.assertDTypeEqual(evidence, jnp.float32)
    self.assertFalse(jnp.any(jnp.isnan(evidence)))

  @parameterized.named_parameters(*delay_vector)
  def test_two_microphone_evidence(self, delay, num_sources,
                                   n_mixture_components):
    """Checks the model evidence is higher when data is generated by the model."""
    del delay  # delete to suppress unused arg warning
    params = get_random_params(0, num_sources, n_mixture_components,
                               self.source_length, self.log_noise_var)
    kernel_mat = jax.jit(
        jax.vmap(
            functools.partial(
                gp_inference.build_single_source_mic_covariance,
                microphone_positions=self.microphone_positions,
                observed_t=self.time,
                c=343.,
                source_length=self.source_length)))(
                    source_params=params["source_params"]).sum(axis=0)
    y = jnp.squeeze(
        sample_functions(
            kernel_mat +
            jnp.eye(2 * self.num_samples) * jnp.exp(self.log_noise_var), 1))
    y0 = y[:self.num_samples]
    y1 = y[self.num_samples:]
    v0 = jax.random.normal(jax.random.PRNGKey(0), shape=(self.num_samples,))
    v1 = jax.random.normal(jax.random.PRNGKey(1), shape=(self.num_samples,))
    evidence_in_distribution = gp_inference.two_microphone_evidence(
        params=params,
        microphone_positions=self.microphone_positions,
        y0=y0,
        y1=y1,
        observed_t=self.time,
        c=343,
        source_length=self.source_length,
        noise_eps=1e-7)
    evidence_out_of_distribution = gp_inference.two_microphone_evidence(
        params=params,
        microphone_positions=self.microphone_positions,
        y0=v0,
        y1=v1,
        observed_t=self.time,
        c=343,
        source_length=self.source_length,
        noise_eps=1e-7)
    self.assertFalse(jnp.any(jnp.isnan(evidence_in_distribution)))
    self.assertFalse(jnp.any(jnp.isnan(evidence_out_of_distribution)))
    self.assertAllGreater(evidence_in_distribution,
                          evidence_out_of_distribution)

  @parameterized.named_parameters(*delay_vector)
  def test_get_fully_learnable_evidence_fn(self, delay, num_sources,
                                           n_mixture_components):
    """Integration test for step of multiple source position learning."""
    del delay  # delete to suppress unused arg warning):
    params = get_random_params(0, num_sources, n_mixture_components,
                               self.source_length, self.log_noise_var)
    v0 = jax.random.normal(jax.random.PRNGKey(0), shape=(self.num_samples,))
    v1 = jax.random.normal(jax.random.PRNGKey(1), shape=(self.num_samples,))
    fully_learnable_evidence = gp_inference.get_fully_learnable_evidence_fn(
        microphone_positions=self.microphone_positions,
        y0=v0,
        y1=v1,
        observed_t=self.time,
        c=343,
        source_length=self.source_length,
        noise_eps=1e-7)
    evidence, grad = jax.value_and_grad(fully_learnable_evidence)(params)
    self.assertFalse(jnp.any(jnp.isnan(evidence)))
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(utils.check_finite, grad["source_params"])))
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda x: jnp.all(jnp.abs(x) > 0),
                                   grad["source_params"])))
    self.assertNotAllEqual(grad["log_noise_var"], 0)

  def test_get_single_source_learnable_evidence_fn(self):
    """Integration test for step of single source position learning."""
    key = jax.random.PRNGKey(0)
    variational_params = get_random_variational_params(
        key,
        num_sources=5,
        n_mixture_components=1,
        log_noise_var=self.log_noise_var)
    (params, _) = gp_inference.position_params_from_variational_params(
        variational_params, self.source_length, 100)
    v0 = jax.random.normal(jax.random.PRNGKey(0), shape=(self.num_samples,))
    v1 = jax.random.normal(jax.random.PRNGKey(1), shape=(self.num_samples,))
    fully_learnable_evidence = gp_inference.get_single_source_learnable_evidence_fn(
        base_params=params,
        learnable_source_idx=0,
        microphone_positions=self.microphone_positions,
        y0=v0,
        y1=v1,
        observed_t=self.time,
        c=343,
        source_length=self.source_length,
        noise_eps=1e-7)
    evidence, grad = jax.value_and_grad(fully_learnable_evidence)(params)
    self.assertFalse(jnp.any(jnp.isnan(evidence)))
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(utils.check_finite, grad)))
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda x: jnp.all(x[0] != 0),
                                   grad["source_params"])))
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda x: jnp.all(x[1] == 0),
                                   grad["source_params"])))


class StochasticPossitionTest(tf.test.TestCase, parameterized.TestCase):
  mic_distance = 1.
  microphone_positions = np.array(([-mic_distance / 2,
                                    0.], [mic_distance / 2, 0.]))
  source_length = 0.1
  sampling_freq = 1000
  sample_time = 1. / sampling_freq
  window_time = 30e-3
  time = np.arange(0, window_time, sample_time)
  num_samples = len(time)
  log_noise_var = np.log(
      1e-3)  # small value added to diagonals to ensure PSD matrices

  def test_gen_single_sample_elbo_fn(self):
    """Integration test for step of learning distribution over source positions."""
    key = jax.random.PRNGKey(0)
    variational_params = get_random_variational_params(
        key,
        num_sources=2,
        n_mixture_components=1,
        log_noise_var=self.log_noise_var)
    v0 = jax.random.normal(jax.random.PRNGKey(0), shape=(self.num_samples,))
    v1 = jax.random.normal(jax.random.PRNGKey(1), shape=(self.num_samples,))
    fully_learnable_evidence = gp_inference.get_fully_learnable_evidence_fn(
        microphone_positions=self.microphone_positions,
        y0=v0,
        y1=v1,
        observed_t=self.time,
        c=343,
        source_length=self.source_length,
        noise_eps=1e-7)
    single_sample_elbo = gp_inference.gen_single_sample_elbo_fn(
        fully_learnable_evidence, source_length=self.source_length)
    evidence, grad = jax.value_and_grad(
        single_sample_elbo, argnums=1)(
            key,
            variational_params,
        )
    self.assertFalse(jnp.any(jnp.isnan(evidence)))
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(utils.check_finite, grad)))
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda x: jnp.all(jnp.abs(x) > 0), grad)))
  def test_gen_multi_sample_estimator_fn(self):
    key = jax.random.PRNGKey(0)
    variational_params = get_random_variational_params(
        key,
        num_sources=2,
        n_mixture_components=1,
        log_noise_var=self.log_noise_var)
    v0 = jax.random.normal(jax.random.PRNGKey(0), shape=(self.num_samples,))
    v1 = jax.random.normal(jax.random.PRNGKey(1), shape=(self.num_samples,))
    fully_learnable_evidence = gp_inference.get_fully_learnable_evidence_fn(
        microphone_positions=self.microphone_positions,
        y0=v0,
        y1=v1,
        observed_t=self.time,
        c=343,
        source_length=self.source_length,
        noise_eps=1e-7)
    single_sample_elbo = gp_inference.gen_single_sample_elbo_fn(
        fully_learnable_evidence, source_length=self.source_length)
    evidence, grad = jax.value_and_grad(
        gp_inference.gen_multi_sample_estimator_fn(
            single_sample_estimator=single_sample_elbo, num_samples=3),
        argnums=1)(
            key,
            variational_params,
        )
    self.assertFalse(jnp.any(jnp.isnan(evidence)))
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(utils.check_finite, grad)))
    self.assertTrue(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda x: jnp.all(jnp.abs(x) > 0), grad)))


class SignalReconstructionTest(tf.test.TestCase, parameterized.TestCase):
  mic_distance = 1.
  microphone_positions = np.array(([-mic_distance / 2,
                                    0.], [mic_distance / 2, 0.]))
  source_length = 0.1
  sampling_freq = 1000
  sample_time = 1. / sampling_freq
  window_time = 30e-3
  observed_t = np.arange(0, window_time, sample_time)
  eval_t = np.arange(window_time-5e-3, window_time+5e-3, sample_time)
  num_samples_observed = len(observed_t)
  num_samples_eval = len(eval_t)
  log_noise_var = np.log(
      1e-3)  # small value added to diagonals to ensure PSD matrices

  def test_build_source_self_covariance(self):
    """Checks structure of source self covariance matrix."""
    params = get_random_params(
        key=0,
        num_sources=1,
        n_mixture_components=4,
        source_length=self.source_length,
        log_noise_var=self.log_noise_var)
    single_source_params = jax.tree_util.tree_map(lambda x: x[0],
                                                  params["source_params"])
    kxx = gp_inference.build_source_self_covariance(single_source_params,
                                                    self.eval_t)
    self.assertFalse(jnp.any(jnp.isnan(kxx)))
    self.assertAllEqual(kxx.shape,
                        (self.num_samples_eval, self.num_samples_eval))
    self.assertEqual(
        jnp.linalg.matrix_rank(kxx + jnp.eye(self.num_samples_eval) *
                               jnp.exp(self.log_noise_var)),
        self.num_samples_eval)
    # Checks matrix symmetry
    self.assertAllClose(kxx, kxx.T)
    # check the marginal variance matches model parameters
    self.assertAllClose(
        jnp.diag(kxx),
        jnp.ones(self.num_samples_eval) *
        jnp.exp(single_source_params["log_marg_var"]).sum())

  def test_build_source_mic_cross_covariance(self):
    """Checks structure of source-microphone cross covariance matrix."""
    params = get_random_params(
        key=0,
        num_sources=1,
        n_mixture_components=4,
        source_length=self.source_length,
        log_noise_var=self.log_noise_var)
    single_source_params = jax.tree_util.tree_map(lambda x: x[0],
                                                  params["source_params"])
    kxy = gp_inference.build_source_mic_cross_covariance(
        source_params=single_source_params,
        microphone_position=self.microphone_positions[0],
        eval_t=self.eval_t,
        observed_t=self.observed_t,
        c=343,
        source_length=self.source_length)
    r = utils.propagation_radius(
        utils.polar_params_to_cartesian(
            single_source_params["source_position"],
            source_length=self.source_length), self.microphone_positions[0])
    self.assertFalse(jnp.any(jnp.isnan(kxy)))
    self.assertAllEqual(kxy.shape,
                        (self.num_samples_eval, self.num_samples_observed))
    # check the maximum covariance does not exceed the marginal variance decayed
    # by attenuation
    self.assertLessEqual(
        jnp.max(kxy), jnp.exp(single_source_params["log_marg_var"]).sum() / r)

  def test_build_mic_cross_covariance(self):
    """Checks structure of microphone-microphone cross covariance matrix."""
    params = get_random_params(
        key=0,
        num_sources=1,
        n_mixture_components=4,
        source_length=self.source_length,
        log_noise_var=self.log_noise_var)
    single_source_params = jax.tree_util.tree_map(lambda x: x[0],
                                                  params["source_params"])
    kmy = gp_inference.build_mic_cross_covariance(
        source_params=single_source_params,
        microphone0_position=self.microphone_positions[0],
        microphone1_position=self.microphone_positions[1],
        t0=self.eval_t,
        t1=self.observed_t,
        c=343,
        source_length=self.source_length)
    r0 = utils.propagation_radius(
        utils.polar_params_to_cartesian(
            single_source_params["source_position"],
            source_length=self.source_length), self.microphone_positions[0])
    r1 = utils.propagation_radius(
        utils.polar_params_to_cartesian(
            single_source_params["source_position"],
            source_length=self.source_length), self.microphone_positions[1])
    self.assertFalse(jnp.any(jnp.isnan(kmy)))
    self.assertAllEqual(kmy.shape,
                        (self.num_samples_eval, self.num_samples_observed))
    # check the maximum covariance does not exceed the marginal variance decayed
    # by attenuation
    self.assertLessEqual(
        jnp.max(kmy),
        jnp.exp(single_source_params["log_marg_var"]).sum() / (r0 * r1))

  def test_build_mic_self_covariance(self):
    """Checks structure of microphone self covariance matrix."""
    params = get_random_params(
        key=0,
        num_sources=1,
        n_mixture_components=4,
        source_length=self.source_length,
        log_noise_var=self.log_noise_var)
    single_source_params = jax.tree_util.tree_map(lambda x: x[0],
                                                  params["source_params"])
    kmm = gp_inference.build_mic_cross_covariance(
        source_params=single_source_params,
        microphone0_position=self.microphone_positions[0],
        microphone1_position=self.microphone_positions[0],
        t0=self.eval_t,
        t1=self.eval_t,
        c=343,
        source_length=self.source_length)
    r0 = utils.propagation_radius(
        utils.polar_params_to_cartesian(
            single_source_params["source_position"],
            source_length=self.source_length), self.microphone_positions[0])
    self.assertFalse(jnp.any(jnp.isnan(kmm)))
    self.assertAllEqual(kmm.shape,
                        (self.num_samples_eval, self.num_samples_eval))
    # check the maximum covariance does not exceed the marginal variance decayed
    # by attenuation
    self.assertLessEqual(
        jnp.max(kmm), (jnp.exp(single_source_params["log_marg_var"]).sum() +
                       jnp.exp(self.log_noise_var)) / (r0 ** 2))

  def test_get_source_cov_mat_generator(self):
    """Checks structure of source self and source-mic cross covariance matrices."""
    num_sources = 3
    params = get_random_params(
        key=0,
        num_sources=num_sources,
        n_mixture_components=4,
        source_length=self.source_length,
        log_noise_var=self.log_noise_var)
    construct_source_cov_mats = gp_inference.get_source_cov_mat_generator(
        params=params,
        microphone_positions=self.microphone_positions,
        observed_t=self.observed_t,
        c=343,
        source_length=self.source_length)
    kxy, kxx = construct_source_cov_mats(self.eval_t)
    # nan checks
    self.assertFalse(jnp.any(jnp.isnan(kxy)))
    self.assertFalse(jnp.any(jnp.isnan(kxx)))
    # shape checks
    self.assertAllEqual(
        kxy.shape,
        (num_sources, self.num_samples_eval, 2 * self.num_samples_observed))
    self.assertAllEqual(
        kxx.shape, (num_sources, self.num_samples_eval, self.num_samples_eval))

  def test_get_upmix_cov_mat_generator(self):
    """Checks structure of mic self and mic-mic cross covariance matrices."""
    num_sources = 3
    params = get_random_params(
        key=0,
        num_sources=num_sources,
        n_mixture_components=4,
        source_length=self.source_length,
        log_noise_var=self.log_noise_var)
    num_upmix = 8
    up_mix_array_positions = testbench.equally_spaced_mic_positions(
        self.mic_distance, num_upmix)
    construct_upmix_cov_mats = gp_inference.get_upmix_cov_mat_generator(
        params=params,
        microphone_positions=self.microphone_positions,
        up_mix_array_positions=up_mix_array_positions,
        observed_t=self.observed_t,
        c=343,
        source_length=self.source_length)
    kmy, kmm = construct_upmix_cov_mats(self.eval_t)
    # nan check
    self.assertFalse(jnp.any(jnp.isnan(kmy)))
    self.assertFalse(jnp.any(jnp.isnan(kmm)))
    # shape check
    self.assertAllEqual(
        kmy.shape,
        (num_upmix, self.num_samples_eval, 2 * self.num_samples_observed))
    self.assertAllEqual(
        kmm.shape, (num_upmix, self.num_samples_eval, self.num_samples_eval))

  def test_gp_reconstruction(self):
    """Tests returned shape, type and finiteness for gp predictive mean."""
    kyy = jnp.eye(self.num_samples_observed)
    kxy = jax.random.normal(
        jax.random.PRNGKey(0),
        shape=(self.num_samples_eval, self.num_samples_observed))
    y = jax.random.normal(
        jax.random.PRNGKey(0), shape=(self.num_samples_observed,))
    mu = gp_inference.gp_reconstruction(y, kyy, kxy)
    self.assertAllEqual(
        mu.shape,
        (self.num_samples_eval,))
    self.assertDTypeEqual(mu, jnp.float32)
    self.assertFalse(jnp.any(jnp.isnan(mu)))

  def test_batched_gp_predictive_covariance(self):
    """Tests returned shape, type and finiteness for gp predictive cov."""
    batch_size = 4
    kyy = jnp.eye(self.num_samples_observed)
    kxy = jax.random.normal(
        jax.random.PRNGKey(0),
        shape=(batch_size, self.num_samples_eval, self.num_samples_observed))
    kxx = jax.random.normal(
        jax.random.PRNGKey(0),
        shape=(batch_size, self.num_samples_eval, self.num_samples_eval))
    cov = gp_inference.batched_gp_predictive_covariance(kyy, kxx, kxy)
    self.assertAllEqual(
        cov.shape,
        (batch_size, self.num_samples_eval, self.num_samples_eval))
    self.assertDTypeEqual(cov, jnp.float32)
    self.assertFalse(jnp.any(jnp.isnan(cov)))

  def test_batched_gp_reconstruction_and_predictive_covariance(self):
    """Tests returned shape, type and finiteness for gp predictive mean & cov."""
    batch_size = 4
    kyy = jnp.eye(self.num_samples_observed)
    kxy = jax.random.normal(
        jax.random.PRNGKey(0),
        shape=(batch_size, self.num_samples_eval, self.num_samples_observed))
    kxx = jax.random.normal(
        jax.random.PRNGKey(0),
        shape=(batch_size, self.num_samples_eval, self.num_samples_eval))
    y = jax.random.normal(
        jax.random.PRNGKey(0), shape=(self.num_samples_observed,))
    mu, cov = gp_inference.batched_gp_reconstruction_and_predictive_covariance(
        y, kyy, kxx, kxy)
    self.assertAllEqual(mu.shape, (batch_size, self.num_samples_eval))
    self.assertDTypeEqual(mu, jnp.float32)
    self.assertFalse(jnp.any(jnp.isnan(mu)))
    self.assertAllEqual(
        cov.shape,
        (batch_size, self.num_samples_eval, self.num_samples_eval))
    self.assertDTypeEqual(cov, jnp.float32)
    self.assertFalse(jnp.any(jnp.isnan(cov)))

  def test_gen_reconstruct_sources_fns(self):
    """Tests shape and value agreement between source reconstructions."""
    num_sources = 3
    params = get_random_params(
        key=0,
        num_sources=num_sources,
        n_mixture_components=4,
        source_length=self.source_length,
        log_noise_var=self.log_noise_var)
    y_stack = jax.random.normal(
        jax.random.PRNGKey(0), shape=(2 * self.num_samples_observed,))
    (reconstruct_source_signals,
     reconstruct_source_signals_with_block_uncertainty,
     reconstruct_source_signals_with_full_uncertainty
    ) = gp_inference.gen_reconstruct_sources_fns(
        params=params,
        microphone_positions=self.microphone_positions,
        y_stack=y_stack,
        observed_t=self.observed_t,
        c=343,
        source_length=self.source_length,
        noise_eps=1e-7)
    mu0 = reconstruct_source_signals(self.eval_t)
    mu1, cov1 = reconstruct_source_signals_with_block_uncertainty(self.eval_t)
    mu2, cov2 = reconstruct_source_signals_with_full_uncertainty(self.eval_t)
    # nan check
    self.assertFalse(jnp.any(jnp.isnan(mu0)))
    self.assertFalse(jnp.any(jnp.isnan(cov1)))
    self.assertFalse(jnp.any(jnp.isnan(cov2)))
    # Check that all functions return the same predictive mean
    self.assertAllClose(mu0, mu1)
    self.assertAllClose(mu0, mu2.reshape(num_sources, self.num_samples_eval))
    # check that block diagonal covariance estimates match
    for i in range(num_sources):
      self.assertAllClose(
          cov1[i],
          cov2[i * self.num_samples_eval:(i + 1) * self.num_samples_eval,
               i * self.num_samples_eval:(i + 1) * self.num_samples_eval],
          atol=1e-5)
    # check shapes
    self.assertAllEqual(
        mu0.shape,
        (num_sources, self.num_samples_eval))
    self.assertAllEqual(
        cov1.shape,
        (num_sources, self.num_samples_eval, self.num_samples_eval))
    self.assertAllEqual(
        cov2.shape,
        (num_sources*self.num_samples_eval, num_sources*self.num_samples_eval))

  def test_gen_upmixing_fns(self):
    """Tests shape and value agreement between up-mixing predictions."""
    num_upmix = 8
    up_mix_array_positions = testbench.equally_spaced_mic_positions(
        self.mic_distance, num_upmix)
    num_sources = 3
    params = get_random_params(
        key=0,
        num_sources=num_sources,
        n_mixture_components=4,
        source_length=self.source_length,
        log_noise_var=self.log_noise_var)
    y_stack = jax.random.normal(
        jax.random.PRNGKey(0), shape=(2*self.num_samples_observed,))
    (upmixed_signals, upmixed_signals_with_block_uncertainty,
     upmixed_signals_with_full_uncertainty) = gp_inference.gen_upmixing_fns(
         params=params,
         microphone_positions=self.microphone_positions,
         up_mix_array_positions=up_mix_array_positions,
         y_stack=y_stack,
         observed_t=self.observed_t,
         c=343,
         source_length=self.source_length,
         noise_eps=1e-7)
    mu0 = upmixed_signals(self.eval_t)
    mu1, cov1 = upmixed_signals_with_block_uncertainty(self.eval_t)
    mu2, cov2 = upmixed_signals_with_full_uncertainty(self.eval_t)
    # check for nans
    self.assertFalse(jnp.any(jnp.isnan(mu0)))
    self.assertFalse(jnp.any(jnp.isnan(cov1)))
    self.assertFalse(jnp.any(jnp.isnan(cov2)))
    # Check that all functions return the same predictive mean
    self.assertAllClose(mu0, mu1)
    self.assertAllClose(mu0, mu2.reshape(num_upmix, self.num_samples_eval))
    # check that block diagonal covariance estimates match
    for i in range(num_upmix):
      self.assertAllClose(
          cov1[i],
          cov2[i * self.num_samples_eval:(i + 1) * self.num_samples_eval,
               i * self.num_samples_eval:(i + 1) * self.num_samples_eval],
          atol=1e-5)
    # check shapes
    self.assertAllEqual(
        mu0.shape,
        (num_upmix, self.num_samples_eval))
    self.assertAllEqual(
        cov1.shape,
        (num_upmix, self.num_samples_eval, self.num_samples_eval))
    self.assertAllEqual(
        cov2.shape,
        (num_upmix*self.num_samples_eval, num_upmix*self.num_samples_eval))

if __name__ == "__main__":
  tf.test.main()
