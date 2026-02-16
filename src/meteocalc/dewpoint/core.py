"""
Docstring for meteocalc.dewpoint.core
"""

from meteocalc.shared._enum_tools import parse_enum
from meteocalc.dewpoint._types import APPROXIMATION_DEWPOINT_REGISTRY
from meteocalc.dewpoint._enums import DewPointEquationName, CalculationMethod
from meteocalc.vapor._enums import EquationName
from meteocalc.dewpoint._dewpoint_equations import DewPointEquation, VaporInversionDewpoint
from typing import Union, Optional
import numpy.typing as npt

class Dewpoint:
    """
    Docstring for Dewpoint
    """

    @staticmethod
    def get_equations_available() -> list[str]:
        """
        Docstring for get_dewpoint_equations_available
        
        :return: Description
        :rtype: list[str]
        """
        equation_list = []
        
        for equation in DewPointEquationName:
            equation_name = equation.value
            equation_list.append(equation_name)
        return equation_list
    
    @staticmethod
    def get_dewpoint(temp_k: Union[float, npt.ArrayLike], rh: Union[float, npt.ArrayLike],
                     calculation_method: Union[str, CalculationMethod] = 'approximation', approximation_equation: Union[str, DewPointEquationName] = 'magnus',
                     solver_vapor_equation: Union[str, EquationName] = 'goff_gratch') -> Union[float, npt.NDArray]:
        """
        Docstring for get_dewpoint
        
        :param temp_k: Description
        :type temp_k: Union[float, npt.ArrayLike]
        :param rh: Description
        :type rh: Union[float, npt.ArrayLike]
        :param calculation_method: Description
        :type calculation_method: Union[str, CalculationMethod]
        :param approximation_equation: Description
        :type approximation_equation: Union[str, DewPointEquationName]
        :param solver_vapor_equation: Description
        :type solver_vapor_equation: Union[str, EquationName]
        :return: Description
        :rtype: float | NDArray
        """
        calculation_method = parse_enum(value=calculation_method, enum_class=CalculationMethod)
        
        if calculation_method == CalculationMethod.SOLVER:
            
            return Dewpoint.get_dewpoint_solver(temp_k=temp_k, rh=rh, 
                                                vapor_equation_name=solver_vapor_equation)
        
        elif calculation_method == CalculationMethod.APPROXIMATION:
            
            return Dewpoint.get_dewpoint_approximation(temp_k=temp_k, rh=rh, dewpoint_equation_name=approximation_equation)

    @staticmethod
    def get_frostpoint(temp_k: Union[float, npt.ArrayLike], rh: Union[float, npt.ArrayLike],
                     calculation_method: Union[str, CalculationMethod] = 'approximation', approximation_equation: Union[str, DewPointEquationName] = 'magnus',
                     solver_vapor_equation: Union[str, EquationName] = 'goff_gratch') -> Union[float, npt.NDArray]:
        """
        Docstring for get_frostpoint
        
        :param temp_k: Description
        :type temp_k: Union[float, npt.ArrayLike]
        :param rh: Description
        :type rh: Union[float, npt.ArrayLike]
        :param calculation_method: Description
        :type calculation_method: Union[str, CalculationMethod]
        :param approximation_equation: Description
        :type approximation_equation: Union[str, DewPointEquationName]
        :param solver_vapor_equation: Description
        :type solver_vapor_equation: Union[str, EquationName]
        :return: Description
        :rtype: float | NDArray
        """
        calculation_method = parse_enum(value=calculation_method, enum_class=CalculationMethod)
        
        if calculation_method == CalculationMethod.SOLVER:
            
            return Dewpoint.get_frostpoint_solver(temp_k=temp_k, rh=rh, 
                                                vapor_equation_name=solver_vapor_equation)
        
        elif calculation_method == CalculationMethod.APPROXIMATION:
            
            return Dewpoint.get_frostpoint_approximation(temp_k=temp_k, rh=rh, dewpoint_equation_name=approximation_equation)
    


    @staticmethod
    def get_dewpoint_solver(temp_k: Union[float, npt.ArrayLike], rh: Union[float, npt.ArrayLike],
                            vapor_equation_name: Union[str, EquationName] = "goff_gratch") -> Union[float, npt.NDArray]:
        """
        Docstring for det_dewpoint_solver
        
        :param temp_k: Description
        :type temp_k: Union[str, npt.ArrayLike]
        :param rh: Description
        :type rh: Union[float, npt.ArrayLike]
        :param vapor_equation: Description
        :type vapor_equation: Union[str, EquationName]
        :return: Description
        :rtype: float | NDArray
        """
        vapor_equation = parse_enum(value=vapor_equation, enum_class=EquationName)
   
        dewpoint_solver = VaporInversionDewpoint(surface_type='water', vapor_equation_name=vapor_equation_name)        
        dewpoints = dewpoint_solver.calculate(temp_k=temp_k, rh=rh)

        return dewpoints
        

    @staticmethod
    def get_frostpoint_solver(temp_k: Union[float, npt.ArrayLike], rh: Union[float, npt.ArrayLike],
                              vapor_equation_name: Union[str, EquationName] = 'goff_gratch') -> Union[float, npt.NDArray]:
        """
        Docstring for get_frostpoint_solver
        
        :param temp_k: Description
        :type temp_k: Union[float, npt.ArrayLike]
        :param rh: Description
        :type rh: Union[float, npt.ArrayLike]
        :param vapor_equation: Description
        :return: Description
        :rtype: float | NDArray
        """
        vapor_equation_name = parse_enum(value=vapor_equation_name, enum_class=EquationName)
   
        dewpoint_solver = VaporInversionDewpoint(surface_type='ice', vapor_equation_name=vapor_equation_name)        
        return dewpoint_solver.calculate(temp_k=temp_k, rh=rh)

    @staticmethod
    def get_dewpoint_approximation(temp_k: Union[float, npt.ArrayLike], rh: Union[float, npt.ArrayLike],
                                   dewpoint_equation_name: Union[str, DewPointEquationName] = 'magnus') -> Union[float, npt.NDArray]:
        """
        Docstring for get_dewpoint_approximation
        
        :param temp_k: Description
        :type temp_k: Union[float, npt.ArrayLike]
        :param rh: Description
        :type rh: Union[float, npt.ArrayLike]
        :param dewpoint_equation: Description
        :type dewpoint_equation: Union[str, DewPointEquationName]
        :return: Description
        :rtype: float | NDArray
        """ 
        dewpoint_equation_name = parse_enum(value=dewpoint_equation_name, enum_class=DewPointEquationName)

        if dewpoint_equation_name not in APPROXIMATION_DEWPOINT_REGISTRY:
            raise ValueError(
                f"{dewpoint_equation_name.value} is not an approximation method"
            )
        
        dewpoint_approximation_class = APPROXIMATION_DEWPOINT_REGISTRY.get(dewpoint_equation_name)
        dewpoint_approximation_equation = dewpoint_approximation_class(surface_type='water')

        return dewpoint_approximation_equation.calculate(temp_k=temp_k, rh=rh)

    @staticmethod
    def get_frostpoint_approximation(temp_k: Union[float, npt.ArrayLike], rh: Union[npt.ArrayLike, float],
                                     frostpoint_equation_name: Union[str, DewPointEquationName] = 'magnus') -> Union[float, npt.NDArray]:
        """
        Docstring for get_frostpoint_approximation
        
        :param temp_k: Description
        :type temp_k: Union[float, npt.ArrayLike]
        :param rh: Description
        :type rh: Union[npt.ArrayLike, float]
        :param frost_point_equation: Description
        :type frost_point_equation: Union[str, DewPointEquationName]
        :return: Description
        :rtype: float | NDArray
        """
        dewpoint_equation_name = parse_enum(value=frostpoint_equation_name, enum_class=DewPointEquationName)

        if dewpoint_equation_name not in APPROXIMATION_DEWPOINT_REGISTRY:
            raise ValueError(
                f"{dewpoint_equation_name.value} is not an approximation method"
            )
        
        dewpoint_approximation_class = APPROXIMATION_DEWPOINT_REGISTRY.get(frostpoint_equation_name)
        dewpoint_approximation_equation = dewpoint_approximation_class(surface_type='ice')

        return dewpoint_approximation_equation.calculate(temp_k=temp_k, rh=rh)

if __name__ == "__main__":
    test = Dewpoint.get_frostpoint_solver(temp_k=244.5, rh=0.6)
    print(test)