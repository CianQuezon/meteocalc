"""
Mixing ratio of water vapour in air, w = ε · e / (p − e), in kg/kg.

Author: Cian Quezon
"""

from typing import Callable, Tuple, Union

import numpy as np
import numpy.typing as npt

from meteocalc.vapor._jit_equations import (
    _mixing_ratio_scalar,
    _mixing_ratio_vectorised,
)


class MixingRatioEquation:
    """
    Mixing ratio of water vapour in air, w = ε · e / (p − e).

    The mixing ratio is the mass of water vapour per unit mass of dry air
    (kg/kg). Inputs are broadcast together with NumPy rules, so a scalar
    pressure can be paired with a vapour pressure grid (e.g. one pressure
    level), or a ``(level, 1, 1)`` pressure array with a
    ``(level, lat, lon)`` vapour pressure field.

    ε = Rd/Rv is taken from :data:`meteocalc.shared.constants.eps`
    (≈ 0.62199). MetPy (0.62196) and ASHRAE (0.62195) use slightly different
    values; results agree with both to within a relative 1e-4.

    Methods
    -------
    calculate(vapor_pressure_hpa, pressure_hpa)
        Mixing ratio in kg/kg. Returns a Python ``float`` for scalar inputs
        and an ``ndarray`` of the broadcast shape otherwise.

    Notes
    -----
    - Returns ``nan`` where ``pressure_hpa <= vapor_pressure_hpa``
      (unphysical: the denominator, dry-air partial pressure, is ≤ 0).
    - Raises ``ValueError`` if the input shapes cannot be broadcast.
    - Not part of ``EQUATION_REGISTRY``: there is a single formula, and
      the choice of saturation vapour pressure equation is made upstream
      when computing ``e``.

    See Also
    --------
    Vapor.get_mixing_ratio : Public wrapper.
    Vapor.get_saturation_mixing_ratio : Mixing ratio at ``es(T)``.

    Examples
    --------
    >>> eq = MixingRatioEquation()
    >>> eq.calculate(vapor_pressure_hpa=23.37, pressure_hpa=1000.0)
    0.01488...
    >>> eq.calculate(vapor_pressure_hpa=[5.0, 10.0, 20.0], pressure_hpa=850.0)
    array([0.00368..., 0.00740..., 0.01498...])
    """

    def _broadcast_inputs(
        self,
        vapor_pressure_hpa: Union[npt.ArrayLike, float],
        pressure_hpa: Union[npt.ArrayLike, float],
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], bool]:
        """Broadcast vapor pressure and total pressure to compatible shapes.

        Parameters
        ----------
        vapor_pressure_hpa : float or array_like
            Vapor pressure in hPa.
        pressure_hpa : float or array_like
            Total atmospheric pressure in hPa.

        Returns
        -------
        Tuple of (vapor_pressure_broadcasted, pressure_broadcasted, is_scalar)
            - vapor_pressure_broadcasted: Broadcasted vapor pressure array.
            - pressure_broadcasted: Broadcasted total pressure array.
            - is_scalar: True if both inputs were scalars, False otherwise.
        """
        vapor_pressure_hpa = np.asarray(vapor_pressure_hpa)
        pressure_hpa = np.asarray(pressure_hpa)

        was_scalar = vapor_pressure_hpa.ndim == 0 and pressure_hpa.ndim == 0

        vapor_pressure_hpa, pressure_hpa = np.broadcast_arrays(
            vapor_pressure_hpa, pressure_hpa
        )

        return (
            np.ascontiguousarray(vapor_pressure_hpa),
            np.ascontiguousarray(pressure_hpa),
            was_scalar,
        )

    def _dispatch_scalar_or_vector(
        self,
        vapor_pressure_hpa: Union[npt.ArrayLike, float],
        pressure_hpa: Union[npt.ArrayLike, float],
        vector_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        scalar_func: Callable[[float, float], float],
    ) -> Union[npt.NDArray[np.float64], float]:
        """Dispatch to scalar or vectorized mixing ratio calculation.

        Parameters
        ----------
        vapor_pressure_hpa : float or array_like
            Vapor pressure in hPa.
        pressure_hpa : float or array_like
            Total atmospheric pressure in hPa.

        Returns
        -------
        float or ndarray
            Mixing ratio (dimensionless, kg/kg). NaN where
            ``pressure_hpa <= vapor_pressure_hpa`` (unphysical).
        """
        vapor_pressure_hpa = np.asarray(vapor_pressure_hpa)
        pressure_hpa = np.asarray(pressure_hpa)

        vapor_pressure_hpa, pressure_hpa, is_scalar = self._broadcast_inputs(
            vapor_pressure_hpa=vapor_pressure_hpa, pressure_hpa=pressure_hpa
        )

        if is_scalar:
            return scalar_func(vapor_pressure_hpa.item(), pressure_hpa.item())
        else:
            pressure_hpa_shape = pressure_hpa.shape
            vapor_pressure_hpa_flatten = vapor_pressure_hpa.flatten()
            pressure_hpa_flatten = pressure_hpa.flatten()
            return vector_func(
                vapor_pressure_hpa_flatten, pressure_hpa_flatten
            ).reshape(pressure_hpa_shape)

    def calculate(
        self,
        vapor_pressure_hpa: Union[npt.ArrayLike, float],
        pressure_hpa: Union[npt.ArrayLike, float],
    ) -> Union[npt.NDArray[np.float64], float]:
        """
        Calculate the mixing ratio of water vapor in air.

        Parameters
        ----------
        vapor_pressure_hpa : float or array_like
            Vapor pressure in hPa.
        pressure_hpa : float or array_like
            Total atmospheric pressure in hPa.

        Returns
        -------
        float or ndarray
            Mixing ratio (dimensionless, kg/kg). NaN where
            ``pressure_hpa <= vapor_pressure_hpa`` (unphysical).
        """
        return self._dispatch_scalar_or_vector(
            vapor_pressure_hpa=vapor_pressure_hpa,
            pressure_hpa=pressure_hpa,
            scalar_func=_mixing_ratio_scalar,
            vector_func=_mixing_ratio_vectorised,
        )
