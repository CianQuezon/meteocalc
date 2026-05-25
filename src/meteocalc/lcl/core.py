"""
Docstring for meteocalc.lcl.core
"""

from typing import Union, Optional

import numpy.typing as npt
from meteocalc.lcl._types import LCL_EQUATION_MAP, LCL_SOLVER_REGISTRY
from meteocalc.lcl._enums import LclEquationName
from meteocalc.lcl._lcl_equation import IterativeLclEquation
from meteocalc.shared._enum_tools import parse_enum
from meteocalc.vapor._enums import EquationName, SurfaceType
from meteocalc.vapor.core import Vapor

class Lcl:
    """
    Docstring for LiftingCondensationLevel
    """

    @staticmethod
    def get_equations_available():
        """
        Docstring for get_equation_available
        """
        equation_list = []

        for equation in LclEquationName:
            equation_name = equation.value
            equation_list.append(equation_name)
        return equation_list
    
    @staticmethod
    def get_equation(lcl_equation_name: str):
        """
        Docstring for get_lcl_equation
        
        :param lcl_equation: Description
        :type lcl_equation: LclEquationName
        """
        lcl_equation_enum = parse_enum(lcl_equation_name, LclEquationName)
        equation_selected = LCL_EQUATION_MAP[lcl_equation_enum]

        return equation_selected
    

    @staticmethod
    def get_lcl(temp_k: Union[float, npt.ArrayLike], dewpoint_temp_k: Union[float, npt.ArrayLike], 
                pressure_hpa: Union[float, npt.ArrayLike], mixing_ratio: Optional[Union[float, npt.ArrayLike]] = None, lcl_equation_name: str = "iterative", 
                vapor_equation_name: str = "goff_gratch") -> tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Docstring for get_lcl
        
        :param temp_k: Description
        :type temp_k: Union[float, npt.ArrayLike]
        :param dewpoint_temp_k: Description
        :type dewpoint_temp_k: Union[float, npt.ArrayLike]
        :param pressure_hpa: Description
        :type pressure_hpa: Union[float, npt.ArrayLike]
        """
        lcl_equation_equation = Lcl.get_equation(lcl_equation_name)


    @staticmethod
    def get_lcl_using_solver(temp_k: Union[float, npt.ArrayLike], dewpoint_temp_k: Union[float, npt.ArrayLike], pressure_hpa: Union[float, npt.ArrayLike],
                       mixing_ratio: Union[float, npt.ArrayLike], solver_name: Union[str, LclEquationName] = 'iterative', vapor_equation_name: Union[str, EquationName] = "goff_gratch", surface_type: Union[str, SurfaceType] = 'automatic') -> tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]]:
        """
        Docstring for get_lcl_solver
        
        :param temp_k: Description
        :type temp_k: Union[float, npt.ArrayLike]
        :param dewpoint_temp_k: Description
        :type dewpoint_temp_k: Union[float, npt.ArrayLike]
        :param pressure_hpa: Description
        :type pressure_hpa: Union[float, npt.ArrayLike]
        :param mixing_ratio: Description
        :type mixing_ratio: Union[float, npt.ArrayLike]
        :param vapor_equation_name: Description
        :type vapor_equation_name: str
        :return: Description
        :rtype: tuple[float | NDArray, float | NDArray]
        """
        vapor_equation = Vapor.get_equation(equation=vapor_equation_name, phase=surface_type)
        lcl_solver_equation_enum = LC
        solver = IterativeLclEquation()
        results = solver.calculate(temp_k=temp_k, dewpoint_temp_k=dewpoint_temp_k, pressure_hpa=pressure_hpa,
                                   mixing_ratio=mixing_ratio, vapor_equation=vapor_equation)
        return results




        