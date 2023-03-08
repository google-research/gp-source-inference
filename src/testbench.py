"""Testbench for up-mixing.

Simulates signal propagation and evaluates up-mixing and source separation
metrics.
"""
from typing import Callable

from jax import jit
from jax import numpy as jnp
from jax import typing

from google3.util.compression.korvapuusti.up_mixing.src import utils

RealArray = typing.ArrayLike


def get_signal_superposition_fn(
    signals: RealArray,
    source_positions: RealArray,
    source_length: float = 0.5,
    c: float = 343) -> Callable[[RealArray, RealArray], RealArray]:
  """Creates a function that calculates the superpostion of signals.

  Args:
    signals: An array-like of sound signals.
    source_positions: A [len(signals), 2]-array containing the
      (x,y)-coordinates of the microphones in meters.
    source_length: minimum distance between sources and microphones in meters.
      Prevents attenuation going to 0.
    c: Speed of sound in meters / second.

  Returns:
    A function that takes in a microphone position and point in time and returns
    the superposition of all source signals at that point.
  """
  # TODO(antoran): add option to include room filter response

  if len(signals) != len(source_positions):
    raise ValueError("'signals' must have the same length as 'sources'")

  def _signal_superposition(microphone, t):
    """Evaluates signal amplitude at a particular microphone and time."""
    superposition = 0
    for signal, source in zip(signals, source_positions):
      time_delay, attenuation = utils.delay_and_attenuation(
          source, microphone, source_length, c=c)
      superposition = superposition + signal(t - time_delay) * attenuation
    return superposition

  return _signal_superposition


@jit
def sinc_interpolate(time_resample: RealArray, signal: RealArray,
                     time: RealArray, sampling_freq: float) -> RealArray:
  """Interpolates a signal in a bandwidth limited fashion.

  This uses Whittaker-Shannon interpolation formula, also known as sinc
  interpolation, to provide a resampled signal.

  Args:
    time_resample: The new time points at which the signal should be resampled.
    signal: The original signal to be resampled.
    time: The time points at which the original signal was sampled.
    sampling_freq: The sampling frequency of the original signal.

  Returns:
    The resampled signal.
  """
  # TODO(antoran): Add sinc decimate for downsampling before
  # separation-localisation.
  to_sinc = jnp.tile(time_resample,
                     (len(time), 1)) - jnp.tile(time[:, jnp.newaxis],
                                                (1, len(time_resample)))
  return signal @ jnp.sinc(to_sinc * sampling_freq)


def equally_spaced_mic_positions(mic_distance: float,
                                 n_mics: int) -> RealArray:
  """Generates equally spaced positions for an array of microphones.

  The positions all lie on the x axis (y-coordinate being zero), and are
  centered around the origin.

  Args:
    mic_distance: distance between two adjacent microphones
    n_mics: an even integer; the number of microphones.

  Returns:
    A [n_microphones, 2]-array containing the ordered
    (x,y)-coordinates of the microphones.
  """
  if n_mics % 2:
    raise ValueError("n_mics should be even")
  half_width = (n_mics // 2 - 0.5) * mic_distance
  x_array_positions = jnp.linspace(-half_width, half_width, n_mics)
  return jnp.column_stack([x_array_positions, jnp.zeros(n_mics)])

