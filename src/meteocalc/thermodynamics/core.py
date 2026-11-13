"""
Public interface for thermodynamic potential temperature calculations.
"""

from typing import Optional, Union

import numpy.typing as npt

from meteocalc.thermodynamics._enums import PotentialTemperatureMode
from meteocalc.thermodynamics._thermodynamic_equation import BoltonPotentialTempEquation, PoissonEquation
from meteocalc.thermodynamics._types import POTENTIAL_TEMP_REGISTRY
from meteocalc.shared._enum_tools import parse_enum


class Thermodynamics:
    """
    Static-method namespace for thermodynamic potential temperature calculations.

    Three potential temperature variants are provided, all temperatures in
    **Kelvin** and pressure in **hPa**:

    - **Dry** (``mode='dry'``): Poisson's equation θ = T(p₀/p)^κ.
      No moisture inputs required. Throughput ~100 M calculations/second.
    - **LCL** (``mode='lcl'``): Bolton (1980) eq. 24, θL at the lifting
      condensation level. Requires LCL temperature, vapour pressure,
      and mixing ratio.
    - **Equivalent** (``mode='equivalent'``): Bolton (1980) eq. 39, θE.
      Same inputs as LCL. Returns nan for unphysical exponential terms.

    See Also
    --------
    get_potential_temperature : Unified dispatcher for all modes.
    get_dry_potential_temperature : Poisson's equation path.
    get_lcl_potential_temperature : Bolton (1980) eq. 24 path.
    get_equivalent_potential_temperature : Bolton (1980) eq. 39 path.
    """

    @staticmethod
    def get_modes_available() -> list[str]:
        """
        Return names of all available potential temperature modes.

        Returns
        -------
        list[str]
            Mode name strings, one per ``PotentialTemperatureMode`` member.
            Current values: ``['dry', 'lcl', 'equivalent']``.

        Examples
        --------
        >>> Thermodynamics.get_modes_available()
        ['dry', 'lcl', 'equivalent']
        """
        return [mode.value for mode in PotentialTemperatureMode]

    @staticmethod
    def get_equation(mode: Union[str, PotentialTemperatureMode]):
        """
        Return the equation class corresponding to the given mode.

        Parameters
        ----------
        mode : str or PotentialTemperatureMode
            Potential temperature mode. Accepted strings: ``'dry'``,
            ``'lcl'``, ``'equivalent'``.

        Returns
        -------
        type
            Equation class registered under ``mode`` in
            :data:`POTENTIAL_TEMP_REGISTRY`. Instantiate and call
            ``.calculate()`` to run the equation.

        Examples
        --------
        >>> cls = Thermodynamics.get_equation('dry')
        >>> eq = cls()
        >>> eq.calculate(temp_k=288.15, pressure_hpa=850.0)
        301.85...
        """
        mode_enum = parse_enum(mode, PotentialTemperatureMode)
        return POTENTIAL_TEMP_REGISTRY[mode_enum]

    @staticmethod
    def get_potential_temperature(
        temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
        mode: Union[str, PotentialTemperatureMode] = "dry",
        lcl_temp_k: Optional[Union[float, npt.ArrayLike]] = None,
        vapor_pressure_hpa: Optional[Union[float, npt.ArrayLike]] = None,
        mixing_ratio: Optional[Union[float, npt.ArrayLike]] = None,
        p0: float = 1000.0,
    ) -> Union[float, npt.NDArray]:
        """
        Compute potential temperature, dispatching to the selected mode.

        Parameters
        ----------
        temp_k : float or array_like of float
            Air temperature in Kelvin.
        pressure_hpa : float or array_like of float
            Atmospheric pressure in hPa.
        mode : str or PotentialTemperatureMode, default ``'dry'``
            Potential temperature variant.
            ``'dry'`` → Poisson's equation (θ).
            ``'lcl'`` → Bolton (1980) eq. 24 (θL).
            ``'equivalent'`` → Bolton (1980) eq. 39 (θE).
        lcl_temp_k : float or array_like of float, optional
            LCL temperature in Kelvin. Required for ``'lcl'`` and
            ``'equivalent'`` modes.
        vapor_pressure_hpa : float or array_like of float, optional
            Water vapour pressure in hPa. Required for ``'lcl'`` and
            ``'equivalent'`` modes.
        mixing_ratio : float or array_like of float, optional
            Water vapour mixing ratio in kg/kg. Required for ``'lcl'``
            and ``'equivalent'`` modes.
        p0 : float, optional
            Reference pressure in hPa, default 1000.0.

        Returns
        -------
        float or ndarray of float64
            Potential temperature in Kelvin.

        Raises
        ------
        ValueError
            If ``mode`` is ``'lcl'`` or ``'equivalent'`` and any of
            ``lcl_temp_k``, ``vapor_pressure_hpa``, or ``mixing_ratio``
            is None.

        Examples
        --------
        >>> Thermodynamics.get_potential_temperature(
        ...     temp_k=288.15, pressure_hpa=850.0
        ... )
        301.85...

        >>> Thermodynamics.get_potential_temperature(
        ...     temp_k=293.15, pressure_hpa=1013.25,
        ...     mode='equivalent',
        ...     lcl_temp_k=283.15,
        ...     vapor_pressure_hpa=12.0,
        ...     mixing_ratio=0.008,
        ... )
        322.59...
        """
        mode_enum = parse_enum(mode, PotentialTemperatureMode)

        if mode_enum == PotentialTemperatureMode.DRY:
            return Thermodynamics.get_dry_potential_temperature(
                temp_k=temp_k, pressure_hpa=pressure_hpa, p0=p0
            )

        missing = [
            name
            for name, val in [
                ("lcl_temp_k", lcl_temp_k),
                ("vapor_pressure_hpa", vapor_pressure_hpa),
                ("mixing_ratio", mixing_ratio),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(
                f"mode='{mode_enum.value}' requires: {', '.join(missing)}"
            )

        if mode_enum == PotentialTemperatureMode.LCL:
            return Thermodynamics.get_lcl_potential_temperature(
                temp_k=temp_k,
                lcl_temp_k=lcl_temp_k,
                pressure_hpa=pressure_hpa,
                vapor_pressure_hpa=vapor_pressure_hpa,
                mixing_ratio=mixing_ratio,
                p0=p0,
            )

        return Thermodynamics.get_equivalent_potential_temperature(
            temp_k=temp_k,
            lcl_temp_k=lcl_temp_k,
            pressure_hpa=pressure_hpa,
            vapor_pressure_hpa=vapor_pressure_hpa,
            mixing_ratio=mixing_ratio,
            p0=p0,
        )

    @staticmethod
    def get_dry_potential_temperature(
        temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
        p0: float = 1000.0,
    ) -> Union[float, npt.NDArray]:
        """
        Calculate dry potential temperature using Poisson's equation.

        Implements θ = T(p₀/p)^κ where κ = Rd/cp = 287.04/1004.0.

        Parameters
        ----------
        temp_k : float or array_like of float
            Air temperature in Kelvin.
        pressure_hpa : float or array_like of float
            Atmospheric pressure in hPa.
        p0 : float, optional
            Reference pressure in hPa, default 1000.0.

        Returns
        -------
        float or ndarray of float64
            Dry potential temperature θ in Kelvin.

        Examples
        --------
        >>> Thermodynamics.get_dry_potential_temperature(
        ...     temp_k=288.15, pressure_hpa=850.0
        ... )
        301.85...
        """
        eq = PoissonEquation()
        return eq.calculate(temp_k=temp_k, pressure_hpa=pressure_hpa, p0=p0)

    @staticmethod
    def get_lcl_potential_temperature(
        temp_k: Union[float, npt.ArrayLike],
        lcl_temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
        vapor_pressure_hpa: Union[float, npt.ArrayLike],
        mixing_ratio: Union[float, npt.ArrayLike],
        p0: float = 1000.0,
    ) -> Union[float, npt.NDArray]:
        """
        Calculate potential temperature at the LCL (θL) using Bolton (1980) eq. 24.

        Implements θL = T(p₀/(p−e))^κ · (T/TL)^(0.28w).

        Parameters
        ----------
        temp_k : float or array_like of float
            Surface air temperature in Kelvin.
        lcl_temp_k : float or array_like of float
            LCL temperature in Kelvin. Compute via
            :class:`~meteocalc.lcl.core.Lcl`.
        pressure_hpa : float or array_like of float
            Total atmospheric pressure in hPa.
        vapor_pressure_hpa : float or array_like of float
            Water vapour pressure in hPa.
        mixing_ratio : float or array_like of float
            Water vapour mixing ratio in kg/kg.
        p0 : float, optional
            Reference pressure in hPa, default 1000.0.

        Returns
        -------
        float or ndarray of float64
            Potential temperature at LCL (θL) in Kelvin.
        """
        eq = BoltonPotentialTempEquation(mode=PotentialTemperatureMode.LCL)
        return eq.calculate(
            temp_k=temp_k,
            lcl_temp_k=lcl_temp_k,
            pressure_hpa=pressure_hpa,
            vapor_pressure_hpa=vapor_pressure_hpa,
            mixing_ratio=mixing_ratio,
            p0=p0,
        )

    @staticmethod
    def get_equivalent_potential_temperature(
        temp_k: Union[float, npt.ArrayLike],
        lcl_temp_k: Union[float, npt.ArrayLike],
        pressure_hpa: Union[float, npt.ArrayLike],
        vapor_pressure_hpa: Union[float, npt.ArrayLike],
        mixing_ratio: Union[float, npt.ArrayLike],
        p0: float = 1000.0,
    ) -> Union[float, npt.NDArray]:
        """
        Calculate equivalent potential temperature (θE) using Bolton (1980) eq. 39.

        Implements θE = θL · exp[(3376/TL − 2.54) · w · (1 + 0.81w)]
        where θL is Bolton (1980) eq. 24. Returns nan for unphysical
        inputs where the exponential term exceeds 50.

        Parameters
        ----------
        temp_k : float or array_like of float
            Surface air temperature in Kelvin.
        lcl_temp_k : float or array_like of float
            LCL temperature in Kelvin. Compute via
            :class:`~meteocalc.lcl.core.Lcl`.
        pressure_hpa : float or array_like of float
            Total atmospheric pressure in hPa.
        vapor_pressure_hpa : float or array_like of float
            Water vapour pressure in hPa.
        mixing_ratio : float or array_like of float
            Water vapour mixing ratio in kg/kg.
        p0 : float, optional
            Reference pressure in hPa, default 1000.0.

        Returns
        -------
        float or ndarray of float64
            Equivalent potential temperature θE in Kelvin.
            nan where exponential term exceeds physical bounds.

        Examples
        --------
        >>> Thermodynamics.get_equivalent_potential_temperature(
        ...     temp_k=293.15, lcl_temp_k=283.15, pressure_hpa=1013.25,
        ...     vapor_pressure_hpa=12.0, mixing_ratio=0.008,
        ... )
        322.59...
        """
        eq = BoltonPotentialTempEquation(mode=PotentialTemperatureMode.EQUIVALENT)
        return eq.calculate(
            temp_k=temp_k,
            lcl_temp_k=lcl_temp_k,
            pressure_hpa=pressure_hpa,
            vapor_pressure_hpa=vapor_pressure_hpa,
            mixing_ratio=mixing_ratio,
            p0=p0,
        )


if __name__ == "__main__":
    # Smoke test for the public Thermodynamics API.
    # Run with: python -m meteocalc.thermodynamics.core
    import math

    import numpy as np

    results: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        results.append((name, bool(condition), detail))

    def raises(exc_type: type, func, *args, **kwargs) -> tuple[bool, str]:
        try:
            func(*args, **kwargs)
        except exc_type as exc:
            return True, f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # wrong exception type
            return False, f"got {type(exc).__name__}: {exc}"
        return False, "no exception raised"

    # Reference parcel (values verified against the JIT kernels / MetPy)
    moist = {
        "temp_k": 293.15,
        "lcl_temp_k": 283.15,
        "pressure_hpa": 1013.25,
        "vapor_pressure_hpa": 12.0,
        "mixing_ratio": 0.008,
    }
    EXPECTED_DRY = 301.8543925  # T=288.15 K, p=850 hPa
    EXPECTED_THETA_L = 293.0681047
    EXPECTED_THETA_E = 314.8824105
    TOL = 1e-4

    # 1. Mode listing
    modes = Thermodynamics.get_modes_available()
    check(
        "get_modes_available",
        sorted(modes) == ["dry", "equivalent", "lcl"],
        str(modes),
    )

    # 2. Per-mode methods return expected values
    dry = Thermodynamics.get_dry_potential_temperature(288.15, 850.0)
    check("dry value", math.isclose(dry, EXPECTED_DRY, abs_tol=TOL), f"{dry:.6f}")

    theta_l = Thermodynamics.get_lcl_potential_temperature(**moist)
    check(
        "theta_l value",
        math.isclose(theta_l, EXPECTED_THETA_L, abs_tol=TOL),
        f"{theta_l:.6f}",
    )

    theta_e = Thermodynamics.get_equivalent_potential_temperature(**moist)
    check(
        "theta_e value",
        math.isclose(theta_e, EXPECTED_THETA_E, abs_tol=TOL),
        f"{theta_e:.6f}",
    )

    # 3. Physical sanity
    at_p0 = Thermodynamics.get_dry_potential_temperature(288.15, 1000.0)
    check("dry: theta == T at p0", math.isclose(at_p0, 288.15), f"{at_p0:.6f}")
    check(
        "ordering theta_l < theta_e",
        theta_l < theta_e,
        f"{theta_l:.3f} < {theta_e:.3f}",
    )

    # 4. Dispatcher matches per-mode methods (string and enum input)
    for mode_str, expected in [
        ("dry", None),
        ("lcl", theta_l),
        ("equivalent", theta_e),
    ]:
        for mode_arg in (mode_str, PotentialTemperatureMode(mode_str)):
            if mode_str == "dry":
                got = Thermodynamics.get_potential_temperature(
                    288.15, 850.0, mode=mode_arg
                )
                expected_val = dry
            else:
                got = Thermodynamics.get_potential_temperature(
                    temp_k=moist["temp_k"],
                    pressure_hpa=moist["pressure_hpa"],
                    mode=mode_arg,
                    lcl_temp_k=moist["lcl_temp_k"],
                    vapor_pressure_hpa=moist["vapor_pressure_hpa"],
                    mixing_ratio=moist["mixing_ratio"],
                )
                expected_val = expected
            check(
                f"dispatcher mode={mode_arg!r}",
                math.isclose(got, expected_val),
                f"{got:.6f}",
            )

    # 5. get_equation returns an equation computing the right mode
    for mode_str, expected in [("lcl", theta_l), ("equivalent", theta_e)]:
        cls = Thermodynamics.get_equation(mode_str)
        got = cls().calculate(**moist)
        check(
            f"get_equation('{mode_str}')().calculate",
            math.isclose(got, expected),
            f"got {got:.6f}, expected {expected:.6f}",
        )
    dry_eq = Thermodynamics.get_equation("dry")()
    got = dry_eq.calculate(temp_k=288.15, pressure_hpa=850.0)
    check("get_equation('dry')().calculate", math.isclose(got, dry), f"{got:.6f}")

    # 6. Error handling
    ok, detail = raises(
        ValueError,
        Thermodynamics.get_potential_temperature,
        293.15,
        1000.0,
        mode="equivalent",
    )
    check("missing moist inputs -> ValueError", ok, detail)

    ok, detail = raises(
        Exception, Thermodynamics.get_potential_temperature, 293.15, 1000.0, mode="bad"
    )
    check("invalid mode raises", ok, detail)

    # 7. Scalar / array handling
    check("scalar in -> float out", isinstance(dry, float), type(dry).__name__)

    temps = np.array([[280.0, 290.0], [300.0, 310.0]])
    arr = Thermodynamics.get_dry_potential_temperature(temps, 850.0)
    check(
        "2D array in -> same-shape ndarray",
        isinstance(arr, np.ndarray) and arr.shape == temps.shape,
        f"shape={getattr(arr, 'shape', None)}",
    )
    scalar_loop = np.array(
        [Thermodynamics.get_dry_potential_temperature(t, 850.0) for t in temps.ravel()]
    ).reshape(temps.shape)
    check(
        "array matches scalar loop",
        np.allclose(arr, scalar_loop),
        f"max diff={np.max(np.abs(arr - scalar_loop)):.2e}",
    )

    n = 10_000
    rng = np.random.default_rng(0)
    big_t = rng.uniform(250.0, 310.0, n)
    big_tl = big_t - rng.uniform(0.0, 15.0, n)
    big_p = rng.uniform(500.0, 1050.0, n)
    big_e = rng.uniform(1.0, 30.0, n)
    big_w = 0.622 * big_e / (big_p - big_e)
    big_theta_e = Thermodynamics.get_equivalent_potential_temperature(
        big_t, big_tl, big_p, big_e, big_w
    )
    check(
        "large array theta_e finite",
        big_theta_e.shape == (n,) and np.all(np.isfinite(big_theta_e)),
        f"n={n}",
    )

    # Report
    width = max(len(name) for name, _, _ in results)
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name:<{width}}  {detail}")
    n_fail = sum(not passed for _, passed, _ in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    raise SystemExit(1 if n_fail else 0)
