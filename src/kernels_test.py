"""Tests for kernels."""


import functools

from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from google3.util.compression.korvapuusti.up_mixing.src import kernels


def gen_random_indices(n_test_poinsample_time=100, scale=3.):
  """Generates random poinsample_time at which to index kernel functions."""
  key = jax.random.PRNGKey(0)
  test_indices = jax.random.normal(key, shape=(n_test_poinsample_time,)) * scale
  return test_indices


def get_spectral_covariance(time_domain_covariance):
  """Returns spectral density function corresponding to inputed covariance kernel."""
  if time_domain_covariance == kernels.rbf_cov:
    return kernels.rbf_spectral_density

  if time_domain_covariance == kernels.matern12_cov:
    return kernels.gen_matern_spectral_density_fn(1)

  if time_domain_covariance == kernels.matern32_cov:
    return kernels.gen_matern_spectral_density_fn(3)

  if time_domain_covariance == kernels.matern52_cov:
    return kernels.gen_matern_spectral_density_fn(5)


def sample_functions(k, n_samples=100):
  """Samples function from Gaussian process with given Kernel."""
  key = jax.random.PRNGKey(0)
  test_indices = jax.random.normal(key, shape=(k.shape[0], n_samples))
  chol = jnp.linalg.cholesky(k)
  return (chol @ test_indices).T  # shape (n_samples, k.shape[0])


class TimeDomainKernelsTest(tf.test.TestCase, parameterized.TestCase):
  """Tests properties and stability of kernel functions in the time domain."""
  named_parameter_list = [
      dict(
          testcase_name='rbf',
          cov_fun=functools.partial(
              kernels.rbf_cov, marg_var=2., lengthscale=1.),
          marg_var=2.,
          lengthscale=1.),
      dict(
          testcase_name='matern12',
          cov_fun=functools.partial(
              kernels.matern12_cov, marg_var=2., lengthscale=1.),
          marg_var=2.,
          lengthscale=1.),
      dict(
          testcase_name='matern32',
          cov_fun=functools.partial(
              kernels.matern32_cov, marg_var=2., lengthscale=1.),
          marg_var=2.,
          lengthscale=1.,
      ),
      dict(
          testcase_name='matern52',
          cov_fun=functools.partial(
              kernels.matern52_cov, marg_var=2., lengthscale=1.),
          marg_var=2.,
          lengthscale=1.),
      ]

  jitter = 1e-3  # small value added to diagonals to ensure kernels are PSD

  named_parameter_list_with_mixtures = named_parameter_list + [
      dict(
          testcase_name='rbf_mixture',
          cov_fun=functools.partial(
              kernels.gen_spectral_mixture_kernel_fn(kernels.rbf_cov),
              marg_var_vec=np.array([2., 1.]),
              lengthscale_vec=np.array([1., 0.5]),
              carrier_vec=np.array([0., 500.])),
          marg_var=3.,
          lengthscale=jnp.nan),
      dict(
          testcase_name='matern12_mixture',
          cov_fun=functools.partial(
              kernels.gen_spectral_mixture_kernel_fn(kernels.matern12_cov),
              marg_var_vec=np.array([2., 1.]),
              lengthscale_vec=np.array([1., 0.5]),
              carrier_vec=np.array([0., 500.])),
          marg_var=3.,
          lengthscale=jnp.nan),
      dict(
          testcase_name='matern32_mixture',
          cov_fun=functools.partial(
              kernels.gen_spectral_mixture_kernel_fn(kernels.matern32_cov),
              marg_var_vec=np.array([2., 1.]),
              lengthscale_vec=np.array([1., 0.5]),
              carrier_vec=np.array([0., 500.])),
          marg_var=3.,
          lengthscale=jnp.nan),
      dict(
          testcase_name='matern52_mixture',
          cov_fun=functools.partial(
              kernels.gen_spectral_mixture_kernel_fn(kernels.matern52_cov),
              marg_var_vec=np.array([2., 1.]),
              lengthscale_vec=np.array([1., 0.5]),
              carrier_vec=np.array([0., 500.])),
          marg_var=3.,
          lengthscale=jnp.nan)
  ]

  @parameterized.named_parameters(*named_parameter_list_with_mixtures)
  def test_possitive_function(self, cov_fun, marg_var, lengthscale):
    """Kernels are possitive functions."""
    del marg_var, lengthscale  # delete to suppress unused arg warning
    n_test_poinsample_time = 100
    test_indices0 = gen_random_indices(n_test_poinsample_time)
    test_indices1 = gen_random_indices(n_test_poinsample_time)
    k = cov_fun(test_indices0, test_indices1)
    self.assertAllGreater(k, 0)
    self.assertFalse(jnp.any(jnp.isnan(k)))
    self.assertAllEqual(k.shape, (n_test_poinsample_time,))

  @parameterized.named_parameters(*named_parameter_list_with_mixtures)
  def test_stationary(self, cov_fun, marg_var, lengthscale):
    """The kernels under consideration are stationary."""
    del marg_var, lengthscale  # delete to suppress unused arg warning
    n_test_poinsample_time = 100
    shift = 3.2
    test_indices0 = gen_random_indices(n_test_poinsample_time)
    test_indices1 = gen_random_indices(n_test_poinsample_time)
    k0 = cov_fun(test_indices0 + shift, test_indices1 + shift)
    k1 = cov_fun(test_indices0, test_indices1)
    self.assertAllClose(k0, k1)
    self.assertFalse(jnp.any(jnp.isnan(k0)))
    self.assertFalse(jnp.any(jnp.isnan(k1)))

  @parameterized.named_parameters(*named_parameter_list)
  def test_decay_with_lengthscale_inverse(self, cov_fun, marg_var, lengthscale):
    """Covariances should decrease as the lengthscale does."""
    del marg_var  # delete to suppress unused arg warning
    n_test_poinsample_time = 100
    test_indices0 = gen_random_indices(n_test_poinsample_time)
    test_indices1 = gen_random_indices(n_test_poinsample_time)
    k0 = cov_fun(test_indices0, test_indices1)
    # partial allows argument overriding
    k1 = cov_fun(test_indices0, test_indices1, lengthscale=lengthscale / 2.)
    self.assertAllGreaterEqual(k0 - k1, 0)
    self.assertFalse(jnp.any(jnp.isnan(k0)))
    self.assertFalse(jnp.any(jnp.isnan(k1)))

  @parameterized.named_parameters(*named_parameter_list_with_mixtures)
  def test_marginal_var(self, cov_fun, marg_var, lengthscale):
    """Autocovariance should match the marginal variance."""
    del lengthscale  # delete to suppress unused arg warning
    n_test_poinsample_time = 100
    test_indices0 = gen_random_indices(n_test_poinsample_time)
    k0 = cov_fun(test_indices0, test_indices0)
    self.assertAllClose(k0, jnp.ones(n_test_poinsample_time) * marg_var)
    self.assertFalse(jnp.any(jnp.isnan(k0)))

  @parameterized.named_parameters(*named_parameter_list_with_mixtures)
  def test_sequence_covariance_shape_and_psd(self, cov_fun, marg_var,
                                             lengthscale):
    """The covariance of a sequence is of size (seq_len, seq_len) and PSD."""
    del lengthscale, marg_var  # delete to suppress unused arg warning
    n_test_poinsample_time = 100
    test_indices0 = gen_random_indices(n_test_poinsample_time)
    k0 = cov_fun(
        test_indices0[None, :],
        test_indices0[:, None]) + jnp.eye(n_test_poinsample_time) * self.jitter
    self.assertAllEqual(k0.shape,
                        (n_test_poinsample_time, n_test_poinsample_time))
    self.assertEqual(jnp.linalg.matrix_rank(k0), n_test_poinsample_time)
    self.assertFalse(jnp.any(jnp.isnan(k0)))


class FreqDomainKernelsTest(tf.test.TestCase, parameterized.TestCase):
  """Tessample_time properties of kernel functions in the freq domain."""

  single_kernel_named_parameter_list = [
      dict(
          testcase_name='rbf',
          cov_fun=kernels.rbf_cov,
          marg_var=0.5,
          lengthscale=3e-5),
      dict(
          testcase_name='matern12',
          cov_fun=kernels.matern12_cov,
          marg_var=0.5,
          lengthscale=3e-5),
      dict(
          testcase_name='matern32',
          cov_fun=kernels.matern32_cov,
          marg_var=0.5,
          lengthscale=3e-5),
      dict(
          testcase_name='matern52',
          cov_fun=kernels.matern52_cov,
          marg_var=0.5,
          lengthscale=3e-5),
      ]
  sampling_freq = 30e3
  sample_time = 1. / sampling_freq
  window_time = 75e-3
  time = np.arange(0, window_time, sample_time)
  num_samples = len(time)
  freq_array = np.arange(
      0,
      int(num_samples / 2 +
          1)) * sampling_freq / num_samples  # Continuous time frequency index
  jitter = 1e-3  # small value added to diagonals to ensure kernels are PSD

  @parameterized.named_parameters(*single_kernel_named_parameter_list)
  def test_possitive_function(self, cov_fun, marg_var, lengthscale):
    """Spectral density should be possitive."""
    spectrum_fun = get_spectral_covariance(cov_fun)
    s = spectrum_fun(self.freq_array, marg_var, lengthscale, carrier_f=0.)
    self.assertAllGreaterEqual(s, 0)
    self.assertFalse(jnp.any(jnp.isnan(s)))
    self.assertAllEqual(s.shape, (len(self.freq_array),))

  @parameterized.named_parameters(*single_kernel_named_parameter_list)
  def test_power(self, cov_fun, marg_var, lengthscale):
    """Signal power should agree in time and frequency domains."""
    spectrum_fun = get_spectral_covariance(cov_fun)
    time_domain_power = cov_fun(0, 0, marg_var, lengthscale)
    s = spectrum_fun(self.freq_array, marg_var, lengthscale, carrier_f=0.)
    # our signals are real and we only consider possitive frequencies so we
    #   multiply by 2 to account for negative frequencies
    spectral_domain_power = 2 * s.sum() / (self.sample_time * self.num_samples)
    self.assertAllClose(time_domain_power, spectral_domain_power, atol=1e-1)

  @parameterized.named_parameters(*single_kernel_named_parameter_list)
  def test_approximate_fft_matsolve(self, cov_fun, marg_var, lengthscale):
    """Test approximate linsolve for stationary kernels."""
    spectrum_fun = get_spectral_covariance(cov_fun)
    K = cov_fun(self.time[None, :], self.time[:, None], marg_var,  # single capital for matrix notation, pylint: disable=invalid-name
                lengthscale) + jnp.eye(self.num_samples) * self.jitter
    M = sample_functions(K, n_samples=30).T  # martix size (num_samples, 30), single capital for matrix notation, pylint: disable=invalid-name
    s = spectrum_fun(
        self.freq_array, marg_var, lengthscale,
        carrier_f=0.) / self.sample_time + self.jitter
    exact_solve = jax.scipy.linalg.solve(K, M, assume_a='pos')
    approx_solve = kernels.stationary_matsolve(s, M)
    err = jnp.median(jnp.abs(exact_solve - approx_solve) / jnp.abs(exact_solve))
    # we only require the approximation to explain half of the signal amplitude
    self.assertLess(err, 1.)

  def test_single_element_mixture(self,):
    """Test a single element mixture matches isample_time non-mixture counterpart."""
    spectrum_fun = get_spectral_covariance(kernels.matern12_cov)
    mixture_spectrum_fun = kernels.gen_spectral_mixture_density_fn(spectrum_fun)
    s = spectrum_fun(self.freq_array, 2., 1e-3, carrier_f=1e3)
    s_mixture = mixture_spectrum_fun(self.freq_array, jnp.array([2.]),
                                     jnp.array([1e-3]), jnp.array([1e3]))
    self.assertAllClose(s, s_mixture, atol=1e-2)

  def test_multi_element_mixture(self,):
    """Test that mixture spectral density is possitive and has power matching isample_time time domain counterpart."""
    marg_var_vec = jnp.array([2., 1.])
    lengthscale_vec = jnp.array([5e-4, 1e-4])
    carrier_vec = jnp.array([1e3, 10.])

    spectrum_fun = get_spectral_covariance(kernels.matern32_cov)
    mixture_spectrum_fun = kernels.gen_spectral_mixture_density_fn(spectrum_fun)
    cov_fun = kernels.gen_spectral_mixture_kernel_fn(kernels.matern32_cov)

    time_domain_power = cov_fun(0, 0, marg_var_vec, lengthscale_vec,
                                carrier_vec)
    s = mixture_spectrum_fun(self.freq_array, marg_var_vec, lengthscale_vec,
                             carrier_vec)
    spectral_domain_power = 2 * s.sum() / (self.sample_time * self.num_samples)

    self.assertAllGreaterEqual(s, 0)
    self.assertFalse(jnp.any(jnp.isnan(s)))
    self.assertAllEqual(s.shape, (len(self.freq_array),))
    self.assertAllClose(time_domain_power, spectral_domain_power, atol=1e-1)


if __name__ == '__main__':
  tf.test.main()
