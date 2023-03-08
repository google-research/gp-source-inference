"""Probabilisitc models for radius and angle of source position."""

from typing import Sequence, Tuple, Union

from jax import Array
from jax import jit
from jax import nn
import jax.numpy as jnp
import jax.random
from jax.scipy import stats
from jax.typing import ArrayLike


RealArray = ArrayLike
IntegerArray = ArrayLike
KeyArray = Union[Array, jax.random.PRNGKeyArray]
Shape = Sequence[int]


def radius_sample_and_log_prob(
    key: KeyArray, log_shape: float, log_scale: float, num_samples: int,
    source_length: float) -> Tuple[RealArray, RealArray, RealArray]:
  r"""Samples radius parameters and get their log-probs.

  Samples radius parameter from a Gamma distribution (support R+) and returns
  their value both in primal and re-parametrised form, and their log
  probability density under the sampling distribution.

  We sample:
  .. math::
    \mathrm{radius} \sim \mathrm{Gamma}(\mathrm{shape}=e^\mathrm{log_shape},
    \mathrm{scale}=\mathrm{scale}, \mathrm{loc}=\mathrm{source_length})
  The re-parametrised radius is given by:
  .. math::
    \mathrm{radius} = e^{\mathrm{radius_param}} + \mathrm{source_length}

  Args:
    key: jax random key
    log_shape: log of gamma shape parameter, referred to as  "a" in
      https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.stats.gamma.pdf.html
    log_scale: log of gamma scale parameter
    num_samples: number of iid samples to draw
    source_length: location parameter---positive scalar added to sampled radius
      to avoiding divisions by 0

  Returns:
    A tuple (radius_param, radius, log_prob) where
    radius_param: reparametrised radius samples, i.e. log of radius without
      source_length added
    radius: radius samples
    log_prob: log-density of radius samples under sampling distribution
  """

  scale = jnp.exp(log_scale)
  shape_param = jnp.exp(
      log_shape)  # named shape param to disambiguate from array shape property

  radius = jax.random.gamma(
      key, a=shape_param, shape=(num_samples,)) * scale + source_length
  radius_param = jnp.log(radius - source_length)

  log_prob = stats.gamma.logpdf(
      radius, a=shape_param, loc=source_length, scale=scale)

  return radius_param, radius, log_prob


def angle_sample_and_log_prob(
    key: KeyArray, mean: float, log_var: float,
    num_samples: int) -> Tuple[RealArray, RealArray, RealArray]:
  r"""Samples angle parameters and gets their log-probs.

  Samples angle parameters from a sigmoid Normal distribution (support (0, pi) )
  and returns their value both in primal and re-parametrised form, and their
  log probability density under the sampling distribution.

  We sample:
  .. math::
  \mathrm{angle_param} \sim \mathrm{Normal}(\mu=\mathrm{mean},
    \sigma^2=e^\mathrm{logvar})
  The re-parametrised angle is given by:
  .. math::
  \mathrm{angle} = \pi \mathrm{sigmoid}(\mathrm{angle_param})
  The log-density is given by applying the change of variable formula:
  .. math::
    \log p(\mathrm{angle}) = \log \mathrm{Normal}(\mathrm{angle_param};
    \mu=\mathrm{mean}, \sigma^2=e^\mathrm{logvar}) - \log
    \mathrm{sigmoid}(angle_param) - \log \mathrm{sigmoid}(-angle_param) - \log
    \pi

  Args:
    key: jax random key
    mean: mean of Normal distribution
      https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.stats.gamma.pdf.html
    log_var: log of Normal distribution variance
    num_samples: number of iid samples to draw

  Returns:
    A tuple (angle_param, angle, log_prob) where
    angle_param: reparametrised angle samples, i.e. logit of angle over pi
    angle: radius parameter samples
    log_prob: log-density of angle samples under sampling distribution
  """
  # jax.random and jax.scipy.stats refer to the Gaussian standard dev as scale
  scale = 0.5 * jnp.exp(log_var)

  angle_param = jax.random.normal(key, shape=(num_samples,)) * scale + mean
  # clip angle param to avoid sigmoid saturation
  angle_param = jnp.clip(angle_param, a_min=-9., a_max=9.)
  log_angle = nn.log_sigmoid(angle_param)
  log_angle_comp = nn.log_sigmoid(
      -angle_param)  # note that 1 - sig(x) = sig(-x)
  angle = jnp.exp(log_angle) * jnp.pi

  base_logprob = stats.norm.logpdf(angle_param, loc=mean, scale=scale)
  log_prob = base_logprob - log_angle - log_angle_comp - jnp.log(jnp.pi)

  return angle_param, angle, log_prob


@jit
def radius_log_prior(r: RealArray) -> RealArray:
  r"""Returns log of an unnormalised prior over radius parameter.

  The prior density is:
  .. math::
    p(r) \propto \frac{1}{\pi r^2}
  And thus we return:
    .. math::
    - 2\log(r) - \log \pi

  Args:
    r: radius parameter
  """
  return - 2*jnp.log(r) - jnp.log(jnp.pi)
