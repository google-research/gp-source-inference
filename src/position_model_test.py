"""Tests for position_model."""

from jax.nn import sigmoid
import jax.numpy as jnp
import jax.random
import numpy as np
import tensorflow as tf

from google3.util.compression.korvapuusti.up_mixing.src import position_model


class PositionModelTest(tf.test.TestCase):

  def test_radius_sample_and_log_prob(self):
    key = jax.random.PRNGKey(0)
    num_samples = 100
    source_length = 1.
    radius_param, radius, log_prob = position_model.radius_sample_and_log_prob(
        key=key,
        log_shape=-3.,
        log_scale=3.,
        num_samples=num_samples,
        source_length=source_length)

    np.testing.assert_allclose(
        jnp.exp(radius_param) + source_length, radius, rtol=1e-05)
    self.assertAllEqual(radius_param.shape, (num_samples,))
    self.assertAllEqual(radius.shape, (num_samples,))
    self.assertAllEqual(log_prob.shape, (num_samples,))
    self.assertTrue(jnp.all(radius > 0))
    self.assertFalse(jnp.any(jnp.isnan(radius_param)))
    self.assertFalse(jnp.any(jnp.isnan(radius)))
    self.assertFalse(jnp.any(jnp.isnan(log_prob)))

  def test_angle_sample_and_log_prob(self):
    key = jax.random.PRNGKey(0)
    num_samples = 100
    angle_param, angle, log_prob = position_model.angle_sample_and_log_prob(
        key=key,
        mean=3.,
        log_var=1.5,
        num_samples=num_samples)

    np.testing.assert_allclose(sigmoid(angle_param) * jnp.pi, angle, rtol=1e-05)
    self.assertAllEqual(angle_param.shape, (num_samples,))
    self.assertAllEqual(angle.shape, (num_samples,))
    self.assertAllEqual(log_prob.shape, (num_samples,))
    self.assertTrue(jnp.all(jnp.logical_and(angle > 0, angle < jnp.pi)))
    self.assertFalse(jnp.any(jnp.isnan(angle_param)))
    self.assertFalse(jnp.any(jnp.isnan(angle)))
    self.assertFalse(jnp.any(jnp.isnan(log_prob)))


if __name__ == '__main__':
  tf.test.main()
