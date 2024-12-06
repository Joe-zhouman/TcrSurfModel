from abc import ABCMeta, abstractmethod
from typing import Tuple, Callable, List, Optional, Dict
from os import path
from scipy.interpolate import interp1d
import pickle


class Mat(metaclass=ABCMeta):
    """
    材料属性抽象类
    """

    __mat_name__: str
    __mat_id__: int
    mat_components: Dict[str, float]
    
    def mat_name(self) -> str:
        """
        材料名称, 唯一标识符, 同数据库
        """
        return self.__mat_name__

    def mat_id(self) -> int:
        """
        材料ID, 唯一标识符, 同数据库
        """
        return self.__mat_id__

    @abstractmethod
    def get_thermal_conductivity(self, temp: float) -> float:
        """
        获取材料在指定温度下的热导率, 单位: W/mK
        """
        pass

    @abstractmethod
    def get_thermal_expansion(self, temp: float) -> float:
        """
        获取材料在指定温度下的热膨胀率, 单位: 1/K
        """
        pass

    @abstractmethod
    def get_specific_heat(self, temp: float) -> float:
        """
        获取材料在指定温度下的比热容, 单位: J/kgK
        """
        pass

    @abstractmethod
    def get_elastic_modulus(self, temp: float) -> float:
        """
        获取材料在指定温度下的杨氏模量, 单位: GPa
        """
        pass

    @abstractmethod
    def get_poisson_ratio(self, temp: float) -> float:
        """
        获取材料在指定温度下的泊松比, 单位: -
        """
        pass

    @abstractmethod
    def get_density(self, temp: float) -> float:
        """
        获取材料在指定温度下的密度, 单位: kg/m3
        """
        pass

    @abstractmethod
    def get_hardness(self, temp: float) -> float:
        """
        获取材料在指定温度下的硬度, 单位: GPa
        """
        pass

    def get_prop(
        self, temp: float
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        获取材料在指定温度下的所有属性
        """
        if temp < 0:
            raise ValueError("Invalid temperature")
        return (
            self.get_thermal_conductivity(temp),
            self.get_thermal_expansion(temp),
            self.get_elastic_modulus(temp),
            self.get_poisson_ratio(temp),
            self.get_specific_heat(temp),
            self.get_density(temp),
            self.get_hardness(temp),
        )

    def get_prop_from_interp(
        self,
        temp_data: List[float],
        prop_data: List[float],
        prop_type: str,
        temp: float,
    ) -> float:
        """
        根据温度数据和属性数据生成一维插值函数，如果已存在之前的插值结果则直接加载。

        参数:
        - temp_data: 温度数据列表，用于插值的x轴数据。
        - prop_data: 属性数据列表，用于插值的y轴数据。
        - prop_type: 保存属性类型。

        返回:
        - 插值得到的属性值。
        """
        # 检查指定的文件是否存在，如果不存在则进行插值计算并保存结果
        filename = path.join(
            path.dirname(__file__), f"db/pkl/{self.__mat_name__}-{prop_type}.pkl"
        )

        if not path.exists(filename):
            # 使用线性插值方式创建插值函数，允许外推
            interp = interp1d(
                temp_data, prop_data, kind="linear", fill_value="extrapolate"
            )
            # 打开文件以二进制写入模式，保存插值函数
            with open(filename, "wb") as f:
                pickle.dump(interp, f)
        # 打开文件以二进制读取模式，加载并返回插值函数
        with open(filename, "rb") as f:
            interp = pickle.load(f)
            return interp(temp).item()


class MatSingleProp(Mat):
    """
    属性值为单值的材料
    @ param: __prop__ 保存属性值的元组
    """
    __prop__: Tuple[float, float, float, float, float, float, float]
    """
    0.thermal conductivity
    1.thermal expansion
    2.Young's modulus
    3.Poisson's ratio
    4.specific heat
    5.density
    6.hardness
    """

    def get_thermal_conductivity(self, temp: float) -> float:
        return self.__prop__[0]

    def get_thermal_expansion(self, temp: float) -> float:
        return self.__prop__[1]

    def get_specific_heat(self, temp: float) -> float:
        return self.__prop__[4]

    def get_elastic_modulus(self, temp: float) -> float:
        return self.__prop__[2]

    def get_poisson_ratio(self, temp: float) -> float:
        return self.__prop__[3]

    def get_density(self, temp: float) -> float:
        return self.__prop__[5]

    def get_hardness(self, temp: float) -> float:
        return self.__prop__[6]
