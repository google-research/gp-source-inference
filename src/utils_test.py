"""Tests for utils."""

from functools import partial  # standard for jax, pylint: disable=g-importing-member

from absl.testing import parameterized
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random
import tensorflow as tf

from google3.util.compression.korvapuusti.up_mixing.src import utils


class SignalGeneratorsTest(tf.test.TestCase, parameterized.TestCase):

  signals_list = [
      dict(
          testcase_name='sine_f',
          signal_fun=partial(utils.sine_f, f=10.5),
          ),
      dict(
          testcase_name='baseband chirp',
          signal_fun=partial(utils.chirp, slope=2.3, carrier_freq=0),
          ),
      dict(
          testcase_name='modulated chirp',
          signal_fun=partial(utils.chirp, slope=2.3, carrier_freq=21.4),
          ),
      dict(
          testcase_name='Gaussian sine mixture',
          signal_fun=partial(
              utils.gaussian_sine_mixture,
              marg_var=1.2,
              freq_mean=10.1,
              freq_std=21.2,
              num_components=34),
      ),
  ]

  @parameterized.named_parameters(*signals_list)
  def test_shape_consistency(self, signal_fun):
    """Tests signal functions preserve the shape of index array."""
    signal_fun = signal_fun()  # jax methods must be called from within main
    t0 = jnp.linspace(-100, 100, 1003)
    s0 = signal_fun(t0)
    self.assertAllEqual(s0.shape, t0.shape)
    self.assertFalse(jnp.any(jnp.isnan(s0)))

    t1 = jax.random.normal(jax.random.PRNGKey(0), shape=(10, 20, 30))
    s1 = signal_fun(t1)
    self.assertAllEqual(s1.shape, t1.shape)
    self.assertFalse(jnp.any(jnp.isnan(s1)))


class PropagationUtilityTest(tf.test.TestCase, parameterized.TestCase):

  def test_propagation_radius(self):
    """Checks radius function returns the 2-norm."""
    source = jnp.array([-1., 2.])
    microphone = jnp.array([6., 5.])
    distance = jnp.sqrt(((source - microphone)**2).sum())
    self.assertEqual(utils.propagation_radius(source, microphone), distance)

  def test_delay(self):
    """Checks propagation delay matches the propagation distance over speed."""
    source = jnp.array([-1., 2.])
    microphone = jnp.array([6., 5.])
    c = 43453.3
    delay = jnp.sqrt(((source - microphone)**2).sum()) / c
    self.assertAlmostEqual(
        utils.propagation_delay(source, microphone, c),
        delay)

  def test_delay_and_attenuation(self):
    """Delay is distance / speed and max attenuation is capped by source_length."""
    source = jnp.array([-1., 2.])
    microphone = jnp.array([6., 5.])
    c = 43453.3
    distance = jnp.sqrt(((source - microphone)**2).sum())
    delay_check = distance / c
    source_length = 6.5
    attenuation_check = 1/jnp.clip(distance, a_min=source_length)
    delay, attenuation = utils.delay_and_attenuation(
        source, microphone, source_length, c)

    self.assertAlmostEqual(delay, delay_check)
    self.assertAlmostEqual(attenuation, attenuation_check)
    self.assertFalse(jnp.any(jnp.isnan(delay)))
    self.assertFalse(jnp.any(jnp.isnan(attenuation)))

    source_length = 1e-1
    attenuation_check = 1/jnp.clip(distance, a_min=source_length)
    delay, attenuation = utils.delay_and_attenuation(
        source, microphone, source_length, c)

    self.assertAlmostEqual(delay, delay_check)
    self.assertAlmostEqual(attenuation, attenuation_check)
    self.assertFalse(jnp.any(jnp.isnan(delay)))
    self.assertFalse(jnp.any(jnp.isnan(attenuation)))


class ParameterManipulationTest(tf.test.TestCase, parameterized.TestCase):

  def test_repeat_for_pytree(self):
    """Checks pytrees are replicated over new axis 0 and other shapes maintained."""
    my_pytree = {'var_a': jnp.arange(4), 'var_b': jnp.ones((3, 3))}
    n_repeat = 10
    repeated_pytree = jax.tree_util.tree_map(
        utils.repeat_for_pytree(n_repeat), my_pytree)
    self.assertAllEqual(repeated_pytree['var_a'].shape, (n_repeat, 4))
    self.assertAllEqual(repeated_pytree['var_b'].shape, (n_repeat, 3, 3))

  def test_polar_to_catesian(self):
    self.assertAllEqual(utils.polar_to_catesian(323., 32.).shape, (2,))
    self.assertAllClose(utils.polar_to_catesian(0., 2334.), jnp.zeros(2))
    self.assertAllClose(
        utils.polar_to_catesian(1., jnp.pi), jnp.array([-1., 0]))
    self.assertAllClose(
        utils.polar_to_catesian(2., jnp.pi / 2), jnp.array([0, 2.]))

  def test_polar_params_to_cartesian_saturation(self):
    """Checks limits of possible source position are not violated."""
    polar_params = jnp.array([-9, -9.])
    source_length = 1.
    cartesian_coords = utils.polar_params_to_cartesian(
        polar_params, source_length=source_length)

    self.assertFalse(jnp.any(jnp.isnan(cartesian_coords)))
    self.assertAllClose(cartesian_coords, jnp.array([1, 0.]), atol=1e-3)
    self.assertGreaterEqual(
        utils.propagation_radius(cartesian_coords, jnp.zeros(2)), source_length)

  def test_polar_params_to_cartesian_midpoint(self):
    polar_params = jnp.array([0., 0.])
    source_length = 1.4
    cartesian_coords = utils.polar_params_to_cartesian(
        polar_params, source_length=source_length)

    self.assertFalse(jnp.any(jnp.isnan(cartesian_coords)))
    self.assertAllClose(cartesian_coords, jnp.array([0., source_length+1]))
    self.assertGreaterEqual(
        utils.propagation_radius(cartesian_coords, jnp.zeros(2)),
        source_length + 1)

  def test_cartesian_to_polar_params_saturation(self):
    source_length = 1.4
    cartesian_coords = jnp.array([source_length, 0.])
    polar_params = utils.cartesian_to_polar_params(
        cartesian_coords, source_length=source_length)

    self.assertFalse(jnp.any(jnp.isnan(polar_params)))
    self.assertAllLessEqual(polar_params, -5.)

  def test_cartesian_to_polar_params_midpoint(self):

    source_length = 1.4
    cartesian_coords = jnp.array([0., source_length + 1])
    polar_params = utils.cartesian_to_polar_params(
        cartesian_coords, source_length=source_length)

    self.assertFalse(jnp.any(jnp.isnan(polar_params)))
    self.assertAllClose(polar_params, jnp.array([0., 0.]))

  def test_cartesian_to_polar_params_reversability(self):
    """Checks this function reverses 'polar_params_to_cartesian'."""
    source_length = 0.32
    polar_params = jax.random.normal(jax.random.PRNGKey(0), shape=(100, 2))

    cartesian_coords = vmap(
        utils.polar_params_to_cartesian, in_axes=(0, None))(polar_params,
                                                            source_length)
    polar_params_reverse = vmap(
        utils.cartesian_to_polar_params, in_axes=(0, None))(cartesian_coords,
                                                            source_length)

    self.assertFalse(jnp.any(jnp.isnan(polar_params)))
    self.assertFalse(jnp.any(jnp.isnan(cartesian_coords)))
    self.assertAllClose(polar_params_reverse, polar_params)

  def test_check_for_nans(self):
    all_nan = jnp.nan * jnp.ones((23, 4))
    one_nan = jnp.ones((23, 4))
    one_nan = one_nan.at[0, 0].set(jnp.nan)
    no_nan = jnp.ones((23, 4))

    self.assertFalse(utils.check_finite(all_nan))
    self.assertFalse(utils.check_finite(one_nan))
    self.assertTrue(utils.check_finite(no_nan))

  def test_optimise(self):
    """Verifies our optimiser finds the maxima of simple function -(x+1)^2."""
    objective = lambda key, x: (-(x+1)**2)[0]
    params = jnp.array([4.2,])
    num_steps = 50

    param_list, objective_values = utils.optimise(
        objective=objective,
        params=params,
        lr=5e-1,
        num_steps=num_steps,
        key=jax.random.PRNGKey(0),
        individual_abs_clip=1e9,
        adam_b1=0.5,
        adam_b2=0.99)

    self.assertAllClose(param_list[-1], jnp.array([-1.]))
    self.assertAllClose(objective_values[-1], 0.)
    self.assertAllEqual(objective_values.shape, (50,))
    self.assertLen(param_list, num_steps + 1)
    self.assertFalse(jnp.any(jnp.isnan(objective_values)))

  def test_windowing(self):
    """Verifies windowed signal's output shape is (num_windows, window_size)."""
    sequence_length = 4353542
    step_size = 3
    window_size = 10

    sequence = jnp.arange(sequence_length)
    windowed_sequence = utils.windowing(
        sequence, step_size=step_size, window_size=window_size)

    num_windows = jnp.floor((sequence_length - window_size) / step_size + 1)
    self.assertAllEqual(windowed_sequence.shape, (num_windows, window_size))
    self.assertFalse(jnp.any(jnp.isnan(windowed_sequence)))


if __name__ == '__main__':
  tf.test.main()
