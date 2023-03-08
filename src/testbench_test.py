"""Tests for testbench."""

from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from google3.testing.pybase import googletest
from google3.util.compression.korvapuusti.up_mixing.src import testbench
from google3.util.compression.korvapuusti.up_mixing.src import utils


class SignalSuperpositionTest(tf.test.TestCase, googletest.TestCase):

  signals = [utils.sine_f(freq) for freq in [300, 500, 700]]

  def test_invalid_input(self):
    # Three signals and two source positions triggers an error.
    self.assertRaises(ValueError, testbench.get_signal_superposition_fn,
                      self.signals, jnp.array([[1, 2], [3, 4]]))

  def test_superposition(self):
    trivial_positions = testbench.get_signal_superposition_fn(
        self.signals, [jnp.array([0, 0])] * 3)
    # Should be close to zero for integers and half-integers.
    self.assertAllClose(
        trivial_positions(
            jnp.array([0, 0]), 1/100*jnp.array([0, 1, 1.5, 2, 10, 100.5])),
        jnp.zeros((6,)),
        atol=1e-4)

  def test_symmetry(self):
    source_positions = jnp.array([[-7, 2], [-3, 1.5], [1.7, 4.7]])
    # Rotated with matrix [[0.777777, -0.628888], [0.628888, 0.777777]].
    rotated_positions = jnp.array([[-6.702215, -2.846662],
                                   [-3.276663, -0.7199985],
                                   [-1.6335527, 4.7246615]])

    superposition = testbench.get_signal_superposition_fn(
        self.signals, source_positions)
    rotated_superposition = testbench.get_signal_superposition_fn(
        self.signals, rotated_positions)
    # The two superpostions come from points rotated around [0, 0],
    # so they should provide the same function when evaluated at zero.
    times_for_evaluation = jnp.linspace(2, 13.2, 7)
    self.assertAllClose(
        superposition(jnp.array([0, 0]), times_for_evaluation),
        rotated_superposition(jnp.array([0, 0]), times_for_evaluation),
        atol=1e-2)


class MicPositionTest(tf.test.TestCase, googletest.TestCase):

  mic_distance = 5.3
  n_mics = 14

  def test_n_mics_odd(self):
    self.assertRaises(ValueError, testbench.equally_spaced_mic_positions,
                      self.mic_distance, 7)

  def test_seq_mic_position_equal_distance(self):
    mic_positions = testbench.equally_spaced_mic_positions(
        self.mic_distance, self.n_mics)
    self.assertAllClose([
        jnp.linalg.norm(p - q) for p, q in zip(mic_positions, mic_positions[1:])
    ], [self.mic_distance] * (self.n_mics - 1))

  def test_centered_around_zero(self):
    mic_positions = testbench.equally_spaced_mic_positions(
        self.mic_distance, self.n_mics)
    self.assertAllClose(jnp.sum(mic_positions, axis=0), (0, 0), atol=1e-5)


class InterpolationTest(tf.test.TestCase, googletest.TestCase,
                        parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='sample middle',
          signal=np.array([-1, 1]),
          time=np.array([-1, 1]),
          sampling_freq=2,
          time_resample=np.array([-1, 0, 1]),
          expected_signal=np.array([-1, 0, 1])),
      dict(
          testcase_name='double sample',
          signal=np.array([0, 0, 1, 1, 0, 0]),
          time=np.arange(6),
          sampling_freq=1,
          time_resample=np.linspace(0, 5, 11),
          # for the expected signal note:
          # - the symmetry
          # - exact match at every other value
          # - slight overshoot in the middle (Gibbs phenomenon)
          expected_signal=np.array([
              0, -0.08488263, 0, 0.4244131, 1, 1.273239, 1, 0.4244131, 0,
              -0.08488263, 0
          ])))
  def test_sinc_interpolate(self, signal, time, sampling_freq, time_resample,
                            expected_signal):
    resampled_signal = testbench.sinc_interpolate(time_resample, signal, time,
                                                  sampling_freq)
    self.assertAllClose(expected_signal, resampled_signal, atol=1e-5)

if __name__ == '__main__':
  googletest.main()
