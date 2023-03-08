"""General utilities for up-mixing package.

Includes functions for simulating free-space audio propagation,for manipulating
model parameters and generating synthetic signals.
"""

from typing import Any, Callable, Optional, Sequence, Tuple, Union

from absl import logging
import jax
from jax import random
from jax import typing as jaxtyping
import jax.numpy as jnp
from jax.numpy.linalg import norm
import optax


RealArray = jaxtyping.ArrayLike
IntegerArray = jaxtyping.ArrayLike
KeyArray = Union[jax.Array, random.PRNGKeyArray]
Shape = Sequence[int]
PyTree = Union[dict[Any, Any], list[Any], jaxtyping.ArrayLike]

# Signal propagation utilities


@jax.jit
def propagation_radius(source_position: RealArray,
                       microphone_position: RealArray) -> float:
  """Computes 2-norm between possition vectors."""
  return norm(microphone_position - source_position, ord=2)


@jax.jit
def propagation_delay(source_position: RealArray,
                      microphone_position: RealArray,
                      c: float = 343) -> float:
  """Computes delay (s) radius (metres) over propagation speed c (metres / s)."""
  return propagation_radius(source_position, microphone_position) / c


@jax.jit
def delay_and_attenuation(source_position: RealArray,
                          microphone_position: RealArray,
                          source_length: float,
                          c: float = 343) -> Tuple[float, float]:
  """Computes delay and attenuation experienced by sound waves in free space.

  Args:
    source_position: [2]-jnp.ndarray, cartesian coordinates in meters.
    microphone_position: [2]-jnp.ndarray, cartesian coordinates in meters.
    source_length: minimum distance between sources and microphones. Prevents
      attenuation going to 0.
    c: speed of sound in meters / second.

  Returns:
    A tuple (time_delay, attenuation) where
    time_delay: delay in seconds between sound generation at source and
      recording at microphone.
    attenuation: proportion by which waveform at source is attenuated when it
      reaches microphone.
  """
  time_delay = propagation_delay(source_position, microphone_position, c=c)
  attenuation = 1 / jnp.clip(
      propagation_radius(source_position, microphone_position),
      a_min=source_length)
  return time_delay, attenuation


# Model parameter manipulation


def repeat_for_pytree(n: int) -> Callable[[RealArray], RealArray]:
  """Returns function that repeats and stacks array entries along axis 0."""
  # not using .tile() since number of indices is unknown.
  return lambda x: jnp.stack([x] * n, axis=0)


@jax.jit
def polar_to_catesian(radius: RealArray, angle: RealArray) -> RealArray:
  """Converts polar coordinates (angles in radians) to cartesian coordinates."""
  return jnp.array([jnp.cos(angle), jnp.sin(angle)]) * radius


@jax.jit
def polar_params_to_cartesian(polar_params: RealArray,
                              source_length: float) -> RealArray:
  r"""Converts polar position parameters to cartesian coordinates.

  Our polar parameters (rp, ap) relate to the standard radius and angle
  (in radians) parameters (r, a) as
  .. math::
    r = \exp(rp) + \mathrm{source_length} \quad \mathrm{and}
    \quad a = \mathrm{sigmoid}(ap) * \pi
  and these relate to cartesian coordinates as
  .. math::
    x, y = r \cos(a), r \sin(a).
  This parametrisation restricts a to (0, pi), only allowing for front-facing
  sources.

  Args:
    polar_params: Array of shape (2,) containing polar coordinate parameters,
      defined as described above.
    source_length: minimum distance between sources and microphones.

  Returns:
    Cartesian coordinate array of shape. (2,)
  """
  log_radius, angle_param = polar_params
  radius = jnp.exp(log_radius) + source_length
  angle = jax.nn.sigmoid(angle_param) * jnp.pi
  source = polar_to_catesian(radius, angle)
  return source


@jax.jit
def cartesian_to_polar_params(cartesian_params: RealArray,
                              source_length: float) -> RealArray:
  r"""Converts cartesian coordinates to polar position parameters.

  Our polar parameters (rp, ap) relate to the standard radius and angle
  (in radians) parameters (r, a) as
  .. math::
    r = \exp(rp) + \mathrm{source_length} \quad \mathrm{and}
    \quad a = \mathrm{sigmoid}(ap) * \pi
  and these relate to cartesian coordinates as
  .. math::
    x, y = r \cos(a), r \sin(a).
  This parametrisation restricts a to (0, pi), only allowing for front-facing
  sources.

  Args:
    cartesian_params: Array of shape (2,) containing x and y cartesian
      coordinates in meters.
    source_length: minimum distance between sources and microphones.

  Returns:
    Polar parameter array of shape. (2,)
  """
  origin = jnp.array([0, 0])
  radius = propagation_radius(cartesian_params, origin)
  angle = jnp.arctan2(cartesian_params[1], cartesian_params[0])
  radius_param = jnp.log(radius - source_length)
  angle_param = jax.scipy.special.logit(angle / jnp.pi)
  return jnp.array([radius_param, angle_param])


def check_finite(x: RealArray) -> bool:
  """Returns True if the passed tensor conatains any nan or infinite entries."""
  return jnp.all(jnp.isfinite(x))


def optimise(objective: Callable[[KeyArray, PyTree], jaxtyping.ArrayLike],
             params: PyTree,
             lr: float,
             num_steps: int,
             key: KeyArray,
             individual_abs_clip: float = 1e9,
             adam_b1: float = 0.5,
             adam_b2: float = 0.99,
             ) -> Tuple[list[PyTree], RealArray]:
  """Finds parameters that improve the value of a objective function.

  Runs the Adam optimiser with gradient clipping for a fixed number of steps to
  maximise the objective. At each step, a random key is passed to accommodate
  for stochastic objective functions.

  TODO(antoran): add documentation explaining why this particular choice of
  optimiser is suitable for the problem of interest.

  Args:
    objective: Objective function that takes in a random key and parameter
      pytree and returns a float.
    params: PyTree containing parameters to be passed to objective and
      optimised.
    lr: learning rate parameter
    num_steps: Number of steps of optimisation to perform.
    key: jax random PRNGKey which is split across iterations and passed to
      objective.
    individual_abs_clip: clips the gradient each leaf node of
    adam_b1: exponential moving average parameter for gradient parameter.
      (Default: 0.5)
    adam_b2: exponential moving average parameter for diagonal curvature
      parameter. (Default: 0.99)

  Returns:
    A tuple (param_list, objective_values) where
    param_list: List of parameters obtained after each step.
    objective_values: Array of size (num_steps,) containing the value of
      objective obtained at each step.
  """

  optimiser = optax.chain(
      optax.clip(individual_abs_clip), optax.adam(lr, b1=adam_b1, b2=adam_b2))
  opt_state = optimiser.init(params)

  def gen_update(objective):
    """Generates jitted inner loop update function."""
    def update(params, opt_state, key):
      """Inner loop update of parameters and optimiser state."""
      # differentiate with respect to argument 1 since argument 0
      # is the random key
      value, grads = jax.value_and_grad(objective, argnums=1)(key, params)
      grads = jax.tree_map(
          lambda x: -x, grads
      )  # by default optax minimises but we want to maximise the objective
      updates, opt_state = optimiser.update(grads, opt_state)
      params = optax.apply_updates(params, updates)
      return value, params, opt_state

    return jax.jit(update)

  update = gen_update(objective)

  objective_values = []
  param_list = [params]

  keys = jax.random.split(key, num_steps)

  for step, key in enumerate(keys):  # Training loop.
    objective_value, new_params, opt_state = update(params, opt_state,
                                                    key)
    finite_params = jax.tree_util.tree_all(
        jax.tree_util.tree_map(check_finite,
                               new_params))

    if finite_params:
      params = new_params
    else:
      logging.error("Detected NaN or inf in updated parameters after"
                    " step %d of optimisation. Reverting to params after step"
                    " %d and stopping optimisation.", step, step-1)
      logging.debug("Offending parameter pytree %s", new_params)
      break

    objective_values.append(objective_value)
    param_list.append(params)

  return param_list, jnp.stack(objective_values, axis=0)


def windowing(x: RealArray, step_size: int, window_size: int) -> RealArray:
  """Separates signal x into overlapping windows.

  Trailing samples that do not fit in the final window are dropped and thus the
  num_windows is given by floor( (len(x) - window_size) / step_size + 1 )

  Args:
    x: signal of size. (num_samples,)
    step_size: stepsize between windows in samples.
    window_size: windowsize in samples.

  Returns:
    y: windowed signal matrix of size. (num_windows, window_size)
  """
  num_samples = x.shape[0]
  start_idxs = jnp.arange(0, num_samples - window_size + 1, step_size)
  num_windows = start_idxs.shape[0]
  indices = jnp.expand_dims(
      jnp.arange(0, window_size), axis=1) * jnp.ones(
          (1, num_windows)) + jnp.ones(
              (window_size, 1)) * start_idxs
  return x[indices.astype(int)].T


# Signal generators


def sine_f(f: float) -> Callable[[RealArray], RealArray]:
  """Returns sine function of specified frequency."""
  signal = lambda t: jnp.sin(f * 2.0 * jnp.pi * t)
  return jax.jit(signal)


def chirp(slope: float,
          carrier_freq: float = 0) -> Callable[[RealArray], RealArray]:
  """Returns modulated chirp signal."""
  assert slope > 0
  def signal(t: RealArray) -> RealArray:
    return jnp.cos(2 * jnp.pi * carrier_freq * t) * jnp.sin(
        slope * 2.0 * jnp.pi * t**2)

  return jax.jit(signal)


def gaussian_sine_mixture(
    marg_var: float,
    freq_mean: float,
    freq_std: float,
    num_components: int,
    key: Optional[KeyArray] = None,
) -> Callable[[RealArray], RealArray]:
  """Returns superposition of tones with Gaussian distributed frequency."""
  if key is None:
    key = jax.random.PRNGKey(0)
  random_freqs = jax.random.normal(
      key, shape=(num_components,)) * freq_std + freq_mean
  random_freqs = jnp.clip(random_freqs, a_min=1.)
  random_phase = jax.random.uniform(
      key, shape=(num_components,), minval=0, maxval=2 * jnp.pi)

  def random_sine(t: RealArray, phase: float, f: float) -> RealArray:
    """Returns sine function with random frequency and phase."""
    return jnp.sin(f * 2.0 * jnp.pi * t + phase)

  def signal(t: RealArray) -> RealArray:
    """Evaluates random sine superposition at a timepoint."""
    return (2 * (marg_var / num_components)**0.5) * jax.vmap(
        random_sine, in_axes=(None, 0, 0))(t, random_phase,
                                           random_freqs).sum(axis=0)

  return jax.jit(signal)

