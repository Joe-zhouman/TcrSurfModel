from typing import Tuple, Callable, List, Optional, Dict

# import Materials
from .db.Aluminium import *
from .db.Ceramic import *
from .db.Copper import *
from .db.Cobalt import *
from .db.Magnesium import *
from .db.Nickel import *
from .db.Silicon import *
from .db.Steel import *
from .db.Titanium import *
from .db.Zirconium import *


class MatDb(object):
    """
    List of available materials
    """

    __MAT_LIST__ = [
        "",         #0
        "cu",       #1
        "al",       #2
        "c",        #3
        "ss304",    #4
        "si",       #5
        "ti",       #6
        "co",       #7
        "aln",      #8
        "al6061t6", #9
        "cu110",    #10
        "brass360", #11
        "cuw30w3",  #12
        "al6061",   #13
        "al5052",   #14
        "brass",    #15
        "ni200",    #16
        "zrnb",     #17
        "zr4",      #18
        "steel2cr12nimowv", #19
        "albh137",          #20
        "tial",     #21
        "steelt10", #22
        "steelgcr15",#23
        "ss305",    #24
        "ssen58f",  #25
        "albshe20", #26
        "brass271", #27
        "ss303",    #28
        "",         #29
        "mgaz31b",  #30
        "al2024t4", #31
        "ss416",    #32
        "al75st6",  #33
        "al5a05h112",   #34
        "al3a21h112",   #35
        "al3a21",       #36
        "al6063",       #37
        "al6061h112",   #38
        "ti6al4v",      #39
        "cc",           #40
        "in600",        #41
        "ssc45",        #42
        "ms",           #43
    ]
    """
    The `__MAT_LIST__` in the `MatDb` class is a list of available materials represented by their
    abbreviations or codes. Each material is associated with a specific index in the list, making it
    easier to reference materials by index rather than their full names. This list serves as a
    reference for the materials available in the database and helps in mapping materials to their
    corresponding indices for easy retrieval and processing within the class methods.
    """

    __MAT_Instance__ = {
        "al": Al,
        "albshe20": Albshe20,
        "brass": Brass,
        "brass271": Brass271,
        "cc": None,
        "cu": Cu,
        "ni200": Ni200,
        "ss304": Steel304,
        "ss305": Steel305,
        "ssc45": SteelC45,
        "ssen58f": SteelEn58F,
        "steel2cr12nimowv": Steel2Cr12NiMoWV,
        "ti6al4v": Ti6Al4V,
        "tial": TiAl,
        "zr4": Zr4,
        "zrnb": ZrNb,
        "albh137": None,
        "in600": In600,
        "si": Si,
        "ti": Ti,
        "co": Co,
        "aln": AlN,
        "al6061t6": Al6061T6,
        "cu110": Cu110,
        "brass360": Brass360,
        "cuw30w3": CuW30W3,
        "al6061": Al6061,
        "al5052": Al5052,
        "steelt10": SteelT10,
        "steelgcr15": SteelGCr15,
        "mgaz31b": MgAZ31B,
        "al2024t4": Al2024T4,
        "ss416": Steel416,
        "al75st6": Al75ST6,
        "al5a05h112": Al5A05H112,
        "al3a21h112": Al3A21H112,
        "al3a21": Al3A21,
        "al6063": Al6063,
        "al6061h112": Al6061H112,
        "ms": MildSteel,
        "ss303": Steel303,
    }
    """
    The `__MAT_Instance__` dictionary in the `MatDb` class is mapping material abbreviations to
    their corresponding class instances. Each key in the dictionary represents a material
    abbreviation, and the value associated with that key is an instance of the class representing
    that material.
    """
    def __init__(self) -> None:
        super().__init__()

    def get_mat_list(self) -> List[str]:
        """
        获取材料列表

        此方法返回一个由字符串组成的列表，每个字符串代表一个材料名称或路径
        该方法没有参数

        返回:
            List[str]: 一个包含材料名称或路径的字符串列表
        """
        return self.__MAT_LIST__

    def get_mat_instance(self) -> Dict[str, Mat]:
        """
        获取Mat实例的字典

        此方法返回一个私有属性__MAT_Instance__，该属性是一个字典，
        其键为字符串类型，值为Mat类型。这个方法的存在是为了提供对这些
        Mat实例的访问。

        Returns:
            Dict[str, Mat]: 一个字典，包含以字符串为键的Mat实例
        """
        return self.__MAT_Instance__

    def get_prop(
        self, mat: Optional[str | int], temp: float
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        根据材料和温度获取属性。

        该方法接受一个可选的材料参数（可以是其名称或索引）和一个温度值，
        然后返回一个包含七个浮点数的元组，代表在指定温度下该材料的属性。

        参数:
        - mat (Optional[str | int]): 材料的名称（字符串）或索引（整数）。
        - temp (float): 温度值。

        返回:
        - Tuple[float, float, float, float, float, float, float]: 包含七个浮点数的元组，表示材料在指定温度下的属性。
        """
        # 检查mat参数是否为整数类型，以确定是否是通过索引查询材料
        if isinstance(mat, int):
            # 确保mat索引在有效范围内
            if mat < 0 or mat >= len(self.__MAT_LIST__):
                return 0, 0, 0, 0, 0, 0, 0
            # 根据索引获取材料名称
            mat = self.__MAT_LIST__[mat]

        # 尝试从材料实例字典中获取对应材料的实例
        mat_instance = self.__MAT_Instance__.get(mat)
        # 如果材料实例不存在，则返回默认值
        if mat_instance is None:
            return 0, 0, 0, 0, 0, 0, 0

        # 创建材料实例对象
        m = mat_instance()
        # 调用材料实例对象的方法，根据温度获取材料属性
        return m.get_prop(temp)
