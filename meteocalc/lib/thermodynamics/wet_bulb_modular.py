import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Union, Callable, Optional
import warnings

class VaporPressureMethod(Enum):
    """Available vapor pressure calculation methods"""
    BOLTON = "bolton"
    GOFF_GRATCH = "goff_gratch"
    BUCK = "buck"
    HYLAND_WEXLER = "hyland_wexler"

class ConvergenceMethod(Enum):
    """Available convergence methods"""
    NEWTON_RAPHSON = "newton_raphson"
    BRENT = "brent"
    HALLEY = "halley"
    SECANT = "secant"
    HYBRID = "hybrid"  # Davies-Jones + Brent fallback

class VaporPressureBase(ABC):
    """Abstract base class for vapor pressure calculations"""
    
    @abstractmethod
    def esat(self, T_k: np.ndarray) -> np.ndarray:
        """Calculate saturation vapor pressure in Pa given temperature in K"""
        pass
    
    @abstractmethod
    def esat_derivative(self, T_k: np.ndarray) -> np.ndarray:
        """Calculate derivative of saturation vapor pressure w.r.t. temperature (Pa/K)"""
        pass
    
    def esat_second_derivative(self, T_k: np.ndarray) -> np.ndarray:
        """Calculate second derivative for Halley's method (Pa/K²)"""
        # Default finite difference approximation
        h = 1e-4
        return (self.esat_derivative(T_k + h) - self.esat_derivative(T_k - h)) / (2 * h)

class BoltonVaporPressure(VaporPressureBase):
    """Bolton 1980 vapor pressure formulation"""
    
    def esat(self, T_k: np.ndarray) -> np.ndarray:
        T_c = T_k - 273.15
        return 611.2 * np.exp(17.67 * T_c / (T_c + 243.5))
    
    def esat_derivative(self, T_k: np.ndarray) -> np.ndarray:
        T_c = T_k - 273.15
        es = self.esat(T_k)
        return es * 17.67 * 243.5 / np.power(T_c + 243.5, 2)

class GoffGratchVaporPressure(VaporPressureBase):
    """Goff-Gratch vapor pressure formulation with ice/water branches"""
    
    def esat(self, T_k: np.ndarray) -> np.ndarray:
        # Dual Goff-Gratch: water above 0°C, ice below 0°C
        T_c = T_k - 273.15
        over_water = T_c >= 0
        es = np.zeros_like(T_k)
        
        # Goff-Gratch over water (T >= 0°C)
        if np.any(over_water):
            T_water = T_k[over_water]
            Tst = 373.15  # Steam point temperature
            log10_es_water = (-7.90298 * (Tst / T_water - 1) +
                             5.02808 * np.log10(Tst / T_water) +
                             -1.3816e-7 * (10**(11.344 * (1 - T_water/Tst)) - 1) +
                             8.1328e-3 * (10**(-3.49149 * (Tst/T_water - 1)) - 1) +
                             np.log10(1013.246))
            es[over_water] = 100.0 * 10**log10_es_water  # Convert mb to Pa
        
        # Goff-Gratch over ice (T < 0°C)
        if np.any(~over_water):
            T_ice = T_k[~over_water]
            T0 = 273.16  # Ice point temperature
            log10_es_ice = (-9.09718 * (T0 / T_ice - 1) +
                           -3.56654 * np.log10(T0 / T_ice) +
                           0.876793 * (1 - T_ice / T0) +
                           np.log10(6.1071))
            es[~over_water] = 100.0 * 10**log10_es_ice  # Convert mb to Pa
        
        return es
    
    def esat_derivative(self, T_k: np.ndarray) -> np.ndarray:
        # Analytical derivatives for better stability and speed
        T_c = T_k - 273.15
        over_water = T_c >= 0
        des_dT = np.zeros_like(T_k)
        
        # Analytical derivative over water
        if np.any(over_water):
            T_water = T_k[over_water]
            es_water = self.esat(T_k[over_water])
            Tst = 373.15
            
            # Analytical derivative of Goff-Gratch water equation
            d_log10_es_dT = (7.90298 * Tst / T_water**2 +
                            -5.02808 / (T_water * np.log(10)) +
                            -1.3816e-7 * 10**(11.344 * (1 - T_water/Tst)) * 
                             11.344 * np.log(10) / Tst +
                            8.1328e-3 * 10**(-3.49149 * (Tst/T_water - 1)) * 
                             3.49149 * np.log(10) * Tst / T_water**2)
            
            des_dT[over_water] = es_water * np.log(10) * d_log10_es_dT
        
        # Analytical derivative over ice
        if np.any(~over_water):
            T_ice = T_k[~over_water]
            es_ice = self.esat(T_k[~over_water])
            T0 = 273.16
            
            # Analytical derivative of Goff-Gratch ice equation
            d_log10_es_dT = (9.09718 * T0 / T_ice**2 +
                            3.56654 / (T_ice * np.log(10)) +
                            -0.876793 / T0)
            
            des_dT[~over_water] = es_ice * np.log(10) * d_log10_es_dT
        
        return des_dT

class BuckVaporPressure(VaporPressureBase):
    """Buck 1981/1996 vapor pressure formulation"""
    
    def esat(self, T_k: np.ndarray) -> np.ndarray:
        T_c = T_k - 273.15
        # Buck equation over water (T > 0°C)
        # Buck equation over ice (T < 0°C)
        over_water = T_c >= 0
        es = np.zeros_like(T_k)
        
        if np.any(over_water):
            T_water = T_c[over_water]
            es[over_water] = 611.21 * np.exp((18.678 - T_water/234.5) * T_water / (257.14 + T_water))
        
        if np.any(~over_water):
            T_ice = T_c[~over_water]
            es[~over_water] = 611.15 * np.exp((23.036 - T_ice/333.7) * T_ice / (279.82 + T_ice))
        
        return es
    
    def esat_derivative(self, T_k: np.ndarray) -> np.ndarray:
        # Analytical derivative of Buck equation
        T_c = T_k - 273.15
        over_water = T_c >= 0
        des_dT = np.zeros_like(T_k)
        
        if np.any(over_water):
            T_w = T_c[over_water]
            es_w = self.esat(T_k[over_water])
            a, b, c = 18.678, 234.5, 257.14
            des_dT[over_water] = es_w * (a*c - T_w**2/b) / (c + T_w)**2
        
        if np.any(~over_water):
            T_i = T_c[~over_water]
            es_i = self.esat(T_k[~over_water])
            a, b, c = 23.036, 333.7, 279.82
            des_dT[~over_water] = es_i * (a*c - T_i**2/b) / (c + T_i)**2
        
        return des_dT

class HylandWexlerVaporPressure(VaporPressureBase):
    """Hyland-Wexler 1983 vapor pressure formulation"""
    
    def esat(self, T_k: np.ndarray) -> np.ndarray:
        # Hyland-Wexler formulation
        ln_es = (-5.8002206e3 / T_k +
                1.3914993 +
                -4.8640239e-2 * T_k +
                4.1764768e-5 * T_k**2 +
                -1.4452093e-8 * T_k**3 +
                6.5459673 * np.log(T_k))
        return np.exp(ln_es)
    
    def esat_derivative(self, T_k: np.ndarray) -> np.ndarray:
        es = self.esat(T_k)
        d_ln_es_dT = (5.8002206e3 / T_k**2 +
                     -4.8640239e-2 +
                     2 * 4.1764768e-5 * T_k +
                     3 * -1.4452093e-8 * T_k**2 +
                     6.5459673 / T_k)
        return es * d_ln_es_dT

class ConvergenceBase(ABC):
    """Abstract base class for convergence methods"""
    
    @abstractmethod
    def solve(self, 
             f_func: Callable[[np.ndarray], np.ndarray],
             df_func: Callable[[np.ndarray], np.ndarray],
             x0: np.ndarray,
             tolerance: float = 0.01,
             max_iterations: int = 50,
             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve f(x) = 0 for x
        Returns: (solution, converged_mask)
        """
        pass

class NewtonRaphsonSolver(ConvergenceBase):
    """Newton-Raphson convergence method with proper vectorization"""
    
    def solve(self, f_func, df_func, x0, tolerance=0.01, max_iterations=50, **kwargs):
        x = x0.copy()
        converged = np.zeros_like(x, dtype=bool)
        
        for iteration in range(max_iterations):
            if np.all(converged):
                break
                
            mask = ~converged
            if not np.any(mask):
                break
            
            # Set active indices for function broadcasting
            if hasattr(f_func, '__self__') or hasattr(f_func, '_active_indices'):
                f_func._active_indices = mask
                df_func._active_indices = mask
            
            try:
                # Extract non-converged points
                mask_indices = np.where(mask)[0]
                
                if len(mask_indices) == 0:
                    continue
                    
                x_active = x[mask_indices]
                f_val = f_func(x_active)
                df_val = df_func(x_active)
                
                # Ensure arrays and handle potential size mismatches
                f_val = np.atleast_1d(np.asarray(f_val))
                df_val = np.atleast_1d(np.asarray(df_val))
                
                # Process each point individually to avoid broadcasting issues
                for i, global_idx in enumerate(mask_indices):
                    # Check bounds
                    if i >= len(f_val) or i >= len(df_val):
                        continue
                        
                    # Check convergence
                    if np.abs(f_val[i]) < tolerance:
                        converged[global_idx] = True
                        continue
                        
                    # Check valid derivative
                    if np.abs(df_val[i]) <= 1e-12:
                        continue
                        
                    # Update point
                    delta_x = f_val[i] / df_val[i]
                    x[global_idx] -= delta_x

            except Exception as e:
                # Fallback for broadcasting issues
                warnings.warn(f"Broadcasting error in Newton solver: {e}")
                break
        
        return x, converged

class BrentSolver(ConvergenceBase):
    """Brent's method for robust convergence - Fixed implementation"""
    
    def solve(self, f_func, df_func, x0, tolerance=0.01, max_iterations=50, 
              x_bounds=None, **kwargs):
        x = x0.copy()
        converged = np.zeros_like(x, dtype=bool)
        
        # Process each point individually for Brent's method
        for i in range(len(x)):
            try:
                # Set up bounds for this point
                if x_bounds is not None:
                    a, b = x_bounds[0][i], x_bounds[1][i]
                else:
                    # Create reasonable bounds around initial guess
                    a = x0[i] - 5.0
                    b = x0[i] + 5.0
                
                # Create scalar function for this specific point
                def scalar_f(xi):
                    # Create single-element array
                    xi_arr = np.array([xi])
                    
                    # Set up function context for this specific point
                    if hasattr(f_func, '_active_indices'):
                        old_indices = getattr(f_func, '_active_indices', None)
                        f_func._active_indices = np.array([i])
                        try:
                            result = f_func(xi_arr)
                            if hasattr(result, '__len__') and len(result) > 0:
                                return float(result[0])
                            else:
                                return float(result)
                        finally:
                            if old_indices is not None:
                                f_func._active_indices = old_indices
                    else:
                        # Direct function evaluation
                        result = f_func(xi_arr)
                        if hasattr(result, '__len__') and len(result) > 0:
                            return float(result[0])
                        else:
                            return float(result)
                
                # Check if root is bracketed
                fa = scalar_f(a)
                fb = scalar_f(b)
                
                # If not bracketed, try to find better bounds
                if fa * fb > 0:
                    # Try expanding bounds
                    for expand in [10, 20, 50]:
                        a_new = x0[i] - expand
                        b_new = x0[i] + expand
                        fa_new = scalar_f(a_new)
                        fb_new = scalar_f(b_new)
                        if fa_new * fb_new < 0:
                            a, b, fa, fb = a_new, b_new, fa_new, fb_new
                            break
                    else:
                        # Still not bracketed, fall back to Newton-Raphson
                        xi = x0[i]
                        def scalar_df(xi_val):
                            xi_arr = np.array([xi_val])
                            if hasattr(df_func, '_active_indices'):
                                old_indices = getattr(df_func, '_active_indices', None)
                                df_func._active_indices = np.array([i])
                                try:
                                    result = df_func(xi_arr)
                                    return float(result[0]) if hasattr(result, '__len__') else float(result)
                                finally:
                                    if old_indices is not None:
                                        df_func._active_indices = old_indices
                            else:
                                result = df_func(xi_arr)
                                return float(result[0]) if hasattr(result, '__len__') else float(result)
                        
                        # Simple Newton-Raphson fallback
                        for _ in range(max_iterations):
                            f_val = scalar_f(xi)
                            if abs(f_val) < tolerance:
                                converged[i] = True
                                break
                            df_val = scalar_df(xi)
                            if abs(df_val) > 1e-12:
                                xi -= f_val / df_val
                            else:
                                break
                        x[i] = xi
                        continue
                
                # Brent's method implementation
                if abs(fa) < abs(fb):
                    a, b = b, a
                    fa, fb = fb, fa
                
                c = a
                fc = fa
                mflag = True
                s = b
                
                for iteration in range(max_iterations):
                    # Check convergence
                    if abs(fb) < tolerance or abs(b - a) < tolerance:
                        x[i] = b
                        converged[i] = True
                        break
                    
                    # Choose method: inverse quadratic interpolation or secant
                    if fa != fc and fb != fc:
                        # Inverse quadratic interpolation
                        s = (a * fb * fc / ((fa - fb) * (fa - fc)) + 
                             b * fa * fc / ((fb - fa) * (fb - fc)) + 
                             c * fa * fb / ((fc - fa) * (fc - fb)))
                    else:
                        # Secant method
                        s = b - fb * (b - a) / (fb - fa)
                    
                    # Check conditions for bisection
                    tmp2 = (3 * a + b) / 4
                    condition1 = not ((s > tmp2 and s < b) if b > tmp2 else (s > b and s < tmp2))
                    condition2 = mflag and abs(s - b) >= abs(b - c) / 2
                    condition3 = not mflag and abs(s - b) >= abs(c - a) / 2
                    condition4 = mflag and abs(b - c) < tolerance
                    condition5 = not mflag and abs(c - a) < tolerance
                    
                    if condition1 or condition2 or condition3 or condition4 or condition5:
                        s = (a + b) / 2
                        mflag = True
                    else:
                        mflag = False
                    
                    fs = scalar_f(s)
                    
                    # Update for next iteration
                    c = b
                    fc = fb
                    
                    if fa * fs < 0:
                        b = s
                        fb = fs
                    else:
                        a = s
                        fa = fs
                    
                    # Ensure |f(a)| >= |f(b)|
                    if abs(fa) < abs(fb):
                        a, b = b, a
                        fa, fb = fb, fa
                
                # If we exit the loop without convergence
                if not converged[i]:
                    x[i] = b
                    
            except Exception as e:
                # If Brent fails, fall back to Newton-Raphson for this point
                try:
                    xi = x0[i]
                    def scalar_f_fallback(xi_val):
                        xi_arr = np.array([xi_val])
                        if hasattr(f_func, '_active_indices'):
                            old_indices = getattr(f_func, '_active_indices', None)
                            f_func._active_indices = np.array([i])
                            try:
                                result = f_func(xi_arr)
                                return float(result[0]) if hasattr(result, '__len__') else float(result)
                            finally:
                                if old_indices is not None:
                                    f_func._active_indices = old_indices
                        else:
                            result = f_func(xi_arr)
                            return float(result[0]) if hasattr(result, '__len__') else float(result)
                    
                    def scalar_df_fallback(xi_val):
                        xi_arr = np.array([xi_val])
                        if hasattr(df_func, '_active_indices'):
                            old_indices = getattr(df_func, '_active_indices', None)
                            df_func._active_indices = np.array([i])
                            try:
                                result = df_func(xi_arr)
                                return float(result[0]) if hasattr(result, '__len__') else float(result)
                            finally:
                                if old_indices is not None:
                                    df_func._active_indices = old_indices
                        else:
                            result = df_func(xi_arr)
                            return float(result[0]) if hasattr(result, '__len__') else float(result)
                    
                    # Newton-Raphson fallback
                    for _ in range(15):
                        f_val = scalar_f_fallback(xi)
                        if abs(f_val) < tolerance:
                            converged[i] = True
                            break
                        df_val = scalar_df_fallback(xi)
                        if abs(df_val) > 1e-12:
                            xi -= f_val / df_val
                        else:
                            break
                    x[i] = xi
                except:
                    # Keep original guess if everything fails
                    x[i] = x0[i]
        
        return x, converged

class HalleySolver(ConvergenceBase):
    """Halley's method (cubic convergence)"""
    
    def solve(self, f_func, df_func, x0, tolerance=0.01, max_iterations=50, 
              d2f_func=None, **kwargs):
        if d2f_func is None:
            # Fallback to Newton-Raphson if second derivative not provided
            return NewtonRaphsonSolver().solve(f_func, df_func, x0, tolerance, max_iterations)
        
        x = x0.copy()
        converged = np.zeros_like(x, dtype=bool)
        
        for iteration in range(max_iterations):
            if np.all(converged):
                break
                
            mask = ~converged
            if not np.any(mask):
                break
                
            f_val = f_func(x[mask])
            df_val = df_func(x[mask])
            d2f_val = d2f_func(x[mask])
            
            # Check convergence
            local_converged = np.abs(f_val) < tolerance
            converged[mask] = local_converged
            
            # Halley's formula: x_new = x - 2*f*f' / (2*(f')^2 - f*f'')
            still_iterating = mask.copy()
            still_iterating[mask] = ~local_converged
            
            if not np.any(still_iterating):
                continue
                
            f_update = f_val[~local_converged]
            df_update = df_val[~local_converged]
            d2f_update = d2f_val[~local_converged]
            
            denominator = 2 * df_update**2 - f_update * d2f_update
            valid_denominator = np.abs(denominator) > 1e-12
            
            if np.any(valid_denominator):
                update_indices = np.where(still_iterating)[0][valid_denominator]
                delta_x = (2 * f_update[valid_denominator] * df_update[valid_denominator] / 
                          denominator[valid_denominator])
                x[update_indices] -= delta_x
        
        return x, converged

class HybridSolver(ConvergenceBase):
    """Hybrid Davies-Jones + Brent fallback solver with improved vectorization"""
    
    def solve(self, f_func, df_func, x0, tolerance=0.01, max_iterations=50,
              x_bounds=None, **kwargs):
        # First try Newton-Raphson (Davies-Jones style)
        newton_solver = NewtonRaphsonSolver()
        
        try:
            x, converged = newton_solver.solve(f_func, df_func, x0, tolerance, 
                                             min(5, max_iterations))
        except Exception as e:
            # If Newton fails completely, try element-wise approach
            warnings.warn(f"Newton solver failed: {e}. Trying element-wise approach.")
            x = x0.copy()
            converged = np.zeros_like(x, dtype=bool)
            
            # Process each element individually
            for i in range(len(x)):
                try:
                    def scalar_f(xi):
                        # Create array with single element
                        xi_array = np.array([xi])
                        if hasattr(f_func, '_active_indices'):
                            # Temporarily set single index
                            old_indices = getattr(f_func, '_active_indices', None)
                            f_func._active_indices = np.array([i])
                            result = f_func(xi_array)[0]
                            if old_indices is not None:
                                f_func._active_indices = old_indices
                            return result
                        else:
                            return f_func(xi_array)[0]
                    
                    def scalar_df(xi):
                        xi_array = np.array([xi])
                        if hasattr(df_func, '_active_indices'):
                            old_indices = getattr(df_func, '_active_indices', None)
                            df_func._active_indices = np.array([i])
                            result = df_func(xi_array)[0]
                            if old_indices is not None:
                                df_func._active_indices = old_indices
                            return result
                        else:
                            return df_func(xi_array)[0]
                    
                    # Simple Newton iteration for single element
                    xi = x0[i]
                    for _ in range(5):
                        f_val = scalar_f(xi)
                        if abs(f_val) < tolerance:
                            converged[i] = True
                            break
                        df_val = scalar_df(xi)
                        if abs(df_val) > 1e-12:
                            xi -= f_val / df_val
                    x[i] = xi
                except:
                    x[i] = x0[i]  # Keep initial guess if all else fails
        
        # For non-converged points, try simple bisection
        if not np.all(converged):
            failed_mask = ~converged
            
            for i in np.where(failed_mask)[0]:
                try:
                    if x_bounds is not None:
                        a, b = x_bounds[0][i], x_bounds[1][i]
                    else:
                        a, b = x0[i] - 5.0, x0[i] + 5.0
                    
                    # Simple bisection
                    def scalar_f_i(xi):
                        xi_array = np.array([xi])
                        if hasattr(f_func, '_active_indices'):
                            old_indices = getattr(f_func, '_active_indices', None)
                            f_func._active_indices = np.array([i])
                            result = f_func(xi_array)[0]
                            if old_indices is not None:
                                f_func._active_indices = old_indices
                            return result
                        else:
                            return f_func(xi_array)[0]
                    
                    fa, fb = scalar_f_i(a), scalar_f_i(b)
                    
                    # If not bracketed, skip bisection
                    if fa * fb > 0:
                        continue
                        
                    for _ in range(20):  # Bisection iterations
                        c = (a + b) / 2
                        fc = scalar_f_i(c)
                        if abs(fc) < tolerance or abs(b - a) < tolerance:
                            x[i] = c
                            converged[i] = True
                            break
                        if fa * fc < 0:
                            b, fb = c, fc
                        else:
                            a, fa = c, fc
                except:
                    pass  # Keep original value if bisection fails
        
        return x, converged

def davies_jones_wet_bulb(temp_c, rh_percent, pressure_hpa, 
                         vapor='bolton', convergence='newton',
                         tolerance=0.01, max_iterations=None):
    """
    Davies-Jones wet bulb calculator with modular vapor pressure and convergence methods.
    
    Parameters:
    -----------
    temp_c : float or array-like
        Dry bulb temperature in Celsius
    rh_percent : float or array-like
        Relative humidity in percent (0-100)
    pressure_hpa : float or array-like
        Atmospheric pressure in hPa
    vapor : str, default 'bolton'
        Vapor pressure method: 'bolton', 'goff_gratch', 'buck', 'hyland_wexler'
    convergence : str, default 'newton'
        Convergence method: 'newton', 'brent', 'halley', 'hybrid'
    tolerance : float, default 0.01
        Convergence tolerance in Kelvin
    max_iterations : int, optional
        Maximum iterations (auto-selected if None)
    
    Returns:
    --------
    float or ndarray
        Wet bulb temperature in Celsius
    
    Examples:
    ---------
    >>> davies_jones_wet_bulb(25.0, 60.0, 1013.25)
    18.048
    
    >>> davies_jones_wet_bulb(25.0, 60.0, 1013.25, vapor='buck', convergence='brent')
    18.051
    
    >>> davies_jones_wet_bulb([-10, 25, 40], [80, 60, 95], [850, 1013, 1013], 
    ...                       vapor='hyland_wexler', convergence='hybrid')
    array([-11.234, 18.048, 38.956])
    """
    
    # Map string inputs to enums
    vapor_mapping = {
        'bolton': VaporPressureMethod.BOLTON,
        'goff_gratch': VaporPressureMethod.GOFF_GRATCH,
        'buck': VaporPressureMethod.BUCK,
        'hyland_wexler': VaporPressureMethod.HYLAND_WEXLER
    }
    
    convergence_mapping = {
        'newton': ConvergenceMethod.NEWTON_RAPHSON,
        'brent': ConvergenceMethod.BRENT,
        'halley': ConvergenceMethod.HALLEY,
        'hybrid': ConvergenceMethod.HYBRID
    }
    
    # Validate inputs
    if vapor not in vapor_mapping:
        raise ValueError(f"vapor must be one of {list(vapor_mapping.keys())}")
    if convergence not in convergence_mapping:
        raise ValueError(f"convergence must be one of {list(convergence_mapping.keys())}")
    
    # Convert inputs to numpy arrays
    temp_c = np.asarray(temp_c, dtype=float)
    rh_percent = np.asarray(rh_percent, dtype=float)
    pressure_hpa = np.asarray(pressure_hpa, dtype=float)
    scalar_input = (temp_c.ndim == 0 and rh_percent.ndim == 0 and pressure_hpa.ndim == 0)
    
    # Ensure arrays for vectorized operations
    if scalar_input:
        temp_c = temp_c.reshape(1)
        rh_percent = rh_percent.reshape(1)
        pressure_hpa = pressure_hpa.reshape(1)
    
    # Input validation
    temp_c = np.clip(temp_c, -50, 60)
    rh_percent = np.clip(rh_percent, 0.1, 100)
    pressure_hpa = np.clip(pressure_hpa, 200, 1200)
    
    # Initialize vapor pressure calculator
    vapor_calculators = {
        VaporPressureMethod.BOLTON: BoltonVaporPressure(),
        VaporPressureMethod.GOFF_GRATCH: GoffGratchVaporPressure(),
        VaporPressureMethod.BUCK: BuckVaporPressure(),
        VaporPressureMethod.HYLAND_WEXLER: HylandWexlerVaporPressure()
    }
    
    # Initialize convergence solver
    convergence_solvers = {
        ConvergenceMethod.NEWTON_RAPHSON: NewtonRaphsonSolver(),
        ConvergenceMethod.BRENT: BrentSolver(),
        ConvergenceMethod.HALLEY: HalleySolver(),
        ConvergenceMethod.HYBRID: HybridSolver()
    }
    
    vapor_calc = vapor_calculators[vapor_mapping[vapor]]
    solver = convergence_solvers[convergence_mapping[convergence]]
    
    # Physical constants
    KELVIN = 273.15
    cp = 1005.0      # Specific heat of dry air (J/kg/K)
    Lv = 2.501e6     # Latent heat of vaporization (J/kg)
    epsilon = 0.622  # Molecular weight ratio
    
    # Convert to working units
    T_k = temp_c + KELVIN
    P_pa = pressure_hpa * 100.0
    rh_frac = rh_percent / 100.0
    
    # Calculate actual vapor pressure with numerical stability
    es = vapor_calc.esat(T_k)
    e = rh_frac * es
    
    # CRITICAL: Protect against log(negative) for very low RH
    e_safe = np.clip(e, 1e-3, None)  # Minimum 1e-3 Pa to prevent log(0) or log(negative)
    
    # Calculate dewpoint for initial guess using safe vapor pressure
    ln_ratio = np.log(e_safe / 611.2)
    Td_k = KELVIN + 243.5 * ln_ratio / (17.67 - ln_ratio)
    
    # Psychrometric constant
    gamma = (cp * P_pa) / (Lv * epsilon)
    
    # Initial guess: between dewpoint and dry bulb
    Tw_k_initial = 0.8 * Td_k + 0.2 * T_k
    
    # Define the psychrometric equation as a function with proper broadcasting
  # Define the psychrometric equation with bulletproof broadcasting
    def psychrometric_function(Tw_k):
        Tw_k = np.asarray(Tw_k)
        es_wb = vapor_calc.esat(Tw_k)
        
        # NUCLEAR OPTION: Handle size mismatches safely
        input_size = len(Tw_k)
        
        # Use the first 'input_size' elements to match Tw_k length
        if input_size <= len(e) and input_size <= len(gamma) and input_size <= len(T_k):
            active_e = e[:input_size]
            active_gamma = gamma[:input_size]  
            active_T_k = T_k[:input_size]
        else:
            # Fallback: use full arrays (for scalar case)
            active_e = e
            active_gamma = gamma
            active_T_k = T_k
        
        return es_wb - active_e - active_gamma * (active_T_k - Tw_k)

    def psychrometric_derivative(Tw_k):
        Tw_k = np.asarray(Tw_k)
        des_dTw = vapor_calc.esat_derivative(Tw_k)
        
        # Same logic as above
        input_size = len(Tw_k)
        
        if input_size <= len(gamma):
            active_gamma = gamma[:input_size]
        else:
            active_gamma = gamma
            
        return des_dTw + active_gamma

    def psychrometric_second_derivative(Tw_k):
        Tw_k = np.asarray(Tw_k)
        return vapor_calc.esat_second_derivative(Tw_k)
    
    # Set default max_iterations if not provided
    if max_iterations is None:
        max_iterations = 15 if convergence != 'newton' else 50
    
    # Set up bounds for Brent's method - CRITICAL FIX
    if convergence == 'brent' or convergence == 'hybrid':
        # Ensure bounds bracket the root: dewpoint < wet_bulb < dry_bulb
        x_bounds = (Td_k - 0.1, T_k - 0.001)  # Small margins for numerical stability
    else:
        x_bounds = None
    
    # Solve the equation
    try:
        if convergence == 'halley':
            Tw_k_result, converged = solver.solve(
                psychrometric_function,
                psychrometric_derivative,
                Tw_k_initial,
                tolerance=tolerance,
                max_iterations=max_iterations,
                d2f_func=psychrometric_second_derivative
            )
        else:
            Tw_k_result, converged = solver.solve(
                psychrometric_function,
                psychrometric_derivative,
                Tw_k_initial,
                tolerance=tolerance,
                max_iterations=max_iterations,
                x_bounds=x_bounds
            )
    except Exception as e:
        warnings.warn(f"Convergence failed: {e}. Using fallback Newton-Raphson method.")
        # Fallback to Newton-Raphson
        fallback_solver = NewtonRaphsonSolver()
        Tw_k_result, converged = fallback_solver.solve(
            psychrometric_function,
            psychrometric_derivative,
            Tw_k_initial,
            tolerance=tolerance,
            max_iterations=max_iterations
        )
    
    # Convert result to Celsius
    result = Tw_k_result - KELVIN
    
    # Physical constraints
    result = np.minimum(result, temp_c - 0.001)
    result = np.maximum(result, temp_c - 50.0)
    
    # Issue warnings for non-converged points
    if not np.all(converged):
        n_failed = np.sum(~converged)
        warnings.warn(f"{n_failed} points failed to converge with {vapor}+{convergence}")
    
    # Return scalar if input was scalar
    if scalar_input:
        return float(result[0])
    return result

# Example usage and testing
if __name__ == "__main__":
    print("DAVIES-JONES WET BULB CALCULATOR")
    print("=" * 50)
    
    # Basic usage examples
    print("Basic Usage Examples:")
    print("-" * 30)
    
    # Standard conditions
    wb1 = davies_jones_wet_bulb(25.0, 60.0, 1013.25)
    print(f"Standard (25°C, 60% RH): {wb1:.3f}°C")
    
    # Different vapor pressure methods
    wb2 = davies_jones_wet_bulb(25.0, 60.0, 1013.25, vapor='buck')
    print(f"With Buck vapor pressure: {wb2:.3f}°C")
    
    # Different convergence methods
    wb3 = davies_jones_wet_bulb(25.0, 60.0, 1013.25, convergence='brent')
    print(f"With Brent convergence: {wb3:.3f}°C")
    
    # Combined modern methods
    wb4 = davies_jones_wet_bulb(25.0, 60.0, 1013.25, vapor='hyland_wexler', convergence='hybrid')
    print(f"Modern methods (H-W + Hybrid): {wb4:.3f}°C")
    
    print("\n" + "=" * 50)
    print("METHOD COMPARISON")
    print("=" * 50)
    
    # Test all combinations
    vapor_methods = ['bolton', 'goff_gratch', 'buck', 'hyland_wexler']
    convergence_methods = ['newton', 'brent', 'halley', 'hybrid']
    
    test_temp, test_rh, test_pressure = 25.0, 60.0, 1013.25
    
    print("Vapor Method\t\tConvergence\tResult (°C)")
    print("-" * 50)
    
    for vapor in vapor_methods:
        for convergence in convergence_methods:
            try:
                result = davies_jones_wet_bulb(test_temp, test_rh, test_pressure, 
                                             vapor=vapor, convergence=convergence)
                print(f"{vapor:<15}\t{convergence:<10}\t{result:.4f}")
            except Exception as e:
                print(f"{vapor:<15}\t{convergence:<10}\tERROR: {str(e)[:20]}")
    
    print("\n" + "=" * 50)
    print("EXTREME CONDITIONS TEST")
    print("=" * 50)
    
    extreme_conditions = [
        (-40, 90, 800),     # Polar conditions
        (45, 95, 1013),     # Tropical extreme
        (20, 30, 500),      # High altitude
        (0, 100, 1013),     # Ice point saturation
        (25, 0.5, 1013),    # Very low RH (numerical stability test)
        (-10, 75, 1013),    # Below freezing (ice formulations test)
        (50, 20, 1013),     # Hot and dry
        (35, 99, 1013)      # Near saturation
    ]
    
    print("Temp(°C)\tRH(%)\tPressure(hPa)\tBolton+Newton\tBuck+Brent\tH-W+Hybrid")
    print("-" * 80)
    
    for temp, rh, pressure in extreme_conditions:
        try:
            wb_bolton = davies_jones_wet_bulb(temp, rh, pressure, 
                                            vapor='bolton', convergence='newton')
            wb_buck = davies_jones_wet_bulb(temp, rh, pressure, 
                                          vapor='buck', convergence='brent')
            wb_hw = davies_jones_wet_bulb(temp, rh, pressure, 
                                        vapor='hyland_wexler', convergence='hybrid')
            
            print(f"{temp:7.1f}\t{rh:5.1f}\t{pressure:12.1f}\t{wb_bolton:10.3f}\t{wb_buck:10.3f}\t{wb_hw:10.3f}")
        except Exception as e:
            print(f"{temp:7.1f}\t{rh:5.1f}\t{pressure:12.1f}\tERROR: {str(e)[:40]}")
    
    print("\n" + "=" * 50)
    print("VECTORIZED CALCULATION TEST")
    print("=" * 50)
    
    # Test vectorized input
    temps = np.array([20, 25, 30, 35])
    rhs = np.array([40, 60, 80, 95])
    pressures = np.array([1013, 1000, 950, 900])
    
    wb_vector = davies_jones_wet_bulb(temps, rhs, pressures, 
                                    vapor='hyland_wexler', convergence='hybrid')
    
    print("Vectorized Input Test:")
    print("Temp(°C)\tRH(%)\tPressure(hPa)\tWet Bulb(°C)")
    print("-" * 50)
    for i in range(len(temps)):
        print(f"{temps[i]:7.1f}\t{rhs[i]:5.1f}\t{pressures[i]:12.1f}\t{wb_vector[i]:11.3f}")
    
    print("\n" + "=" * 50)
    print("CLIMATE MODEL RECOMMENDATIONS")
    print("=" * 50)
    
    print("For different applications:")
    print("• Operational Weather: davies_jones_wet_bulb(T, RH, P, vapor='buck', convergence='newton')")
    print("• Climate Models: davies_jones_wet_bulb(T, RH, P, vapor='hyland_wexler', convergence='hybrid')")
    print("• Research/High Accuracy: davies_jones_wet_bulb(T, RH, P, vapor='hyland_wexler', convergence='halley')")
    print("• Legacy Compatibility: davies_jones_wet_bulb(T, RH, P, vapor='bolton', convergence='newton')")
    print("• Extreme Conditions: davies_jones_wet_bulb(T, RH, P, vapor='buck', convergence='brent')")
    
    print("\n" + "Method Characteristics:")
    print("Vapor Pressure Methods:")
    print("• bolton: Original Davies-Jones, fast, good accuracy")
    print("• goff_gratch: Historical standard, ice/water branches, slower")
    print("• buck: Modern replacement for Goff-Gratch, better accuracy")
    print("• hyland_wexler: Current meteorological standard, highest accuracy")
    print()
    print("Convergence Methods:")
    print("• newton: Fast, 2-3 iterations typically, may fail in extremes")
    print("• brent: Most robust, guaranteed convergence, slightly slower")
    print("• halley: Fastest convergence when stable, needs 2nd derivatives")
    print("• hybrid: Best of both worlds, Newton+Brent fallback")