from typing import Tuple, Callable, List, Optional, Dict, Union

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


class MatDb:
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

    __MAT_INSTANCE__ = {
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
        "albh137": AlBH137,
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
    __PERIODIC_TABLE__: Dict[str, int] = {
        "H": 0,
        "He": 1,
        "Li": 2,
        "Be": 3,
        "B": 4,
        "C": 5,
        "N": 6,
        "O": 7,
        "F": 8,
        "Ne": 9,
        "Na": 10,
        "Mg": 11,
        "Al": 12,
        "Si": 13,
        "P": 14,
        "S": 15,
        "Cl": 16,
        "Ar": 17,
        "K": 18,
        "Ca": 19,
        "Sc": 20,
        "Ti": 21,
        "V": 22,
        "Cr": 23,
        "Mn": 24,
        "Fe": 25,
        "Co": 26,
        "Ni": 27,
        "Cu": 28,
        "Zn": 29,
        "Ga": 30,
        "Ge": 31,
        "As": 32,
        "Se": 33,
        "Br": 34,
        "Kr": 35,
        "Rb": 36,
        "Sr": 37,
        "Y": 38,
        "Zr": 39,
        "Nb": 40,
        "Mo": 41,
        "Tc": 42,
        "Ru": 43,
        "Rh": 44,
        "Pd": 45,
        "Ag": 46,
        "Cd": 47,
        "In": 48,
        "Sn": 49,
        "Sb": 50,
        "Te": 51,
        "I": 52,
        "Xe": 53,
        "Cs": 54,
        "Ba": 55,
        "La": 56,
        "Ce": 57,
        "Pr": 58,
        "Nd": 59,
        "Pm": 60,
        "Sm": 61,
        "Eu": 62,
        "Gd": 63,
        "Tb": 64,
        "Dy": 65,
        "Ho": 66,
        "Er": 67,
        "Tm": 68,
        "Yb": 69,
        "Lu": 70,
        "Hf": 71,
        "Ta": 72,
        "W": 73,
        "Re": 74,
        "Os": 75,
        "Ir": 76,
        "Pt": 77,
        "Au": 78,
        "Hg": 79,
        "Tl": 80,
        "Pb": 81,
        "Bi": 82,
        "Po": 83,
        "At": 84,
        "Rn": 85,
        "Fr": 86,
        "Ra": 87,
        "Ac": 88,
        "Th": 89,
        "Pa": 90,
        "U": 91,
        "Np": 92,
        "Pu": 93,
        "Am": 94,
        "Cm": 95,
        "Bk": 96,
        "Cf": 97,
        "Es": 98,
        "Fm": 99,
        "Md": 100,
        "No": 101,
        "Lr": 102,
        "Rf": 103,
        "Db": 104,
        "Sg": 105,
        "Bh": 106,
        "Hs": 107,
        "Mt": 108,
        "Ds": 109,
        "Rg": 110,
        "Cn": 111,
        "Nh": 112,
        "Fl": 113,
        "Mc": 114,
        "Lv": 115,
        "Ts": 116,
        "Og": 117,
    }
    """元素周期表"""

    def __init__(self, max_length: int = 103) -> None:
        self.max_length = max_length

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
        return self.__MAT_INSTANCE__

    def get_mat_instance(self, mat: Union[str | int]) -> Mat:
        """
        根据材料名称或索引获取材料实例。

        此方法首先检查输入的材料参数是名称还是索引。如果是索引，它会根据索引从预定义的材料列表中获取对应的材料名称。
        然后，它会尝试从材料实例字典中获取该材料的实例。如果材料名称或索引无效，它会抛出一个 ValueError 异常。

        参数:
        mat (str | int): 材料名称的字符串或材料索引的整数。

        返回:
        Mat: 返回一个材料实例。

        抛出:
        ValueError: 如果材料索引超出范围或材料名称无效。
        """
        # 检查输入是否为整数类型，即是否为索引
        if isinstance(mat, int):
            # 确保材料索引在有效范围内
            if mat < 0 or mat >= len(self.__MAT_LIST__):
                raise ValueError("Invalid material index")
            # 根据索引从材料列表中获取材料名称
            mat = self.__MAT_LIST__[mat]
        # 尝试从材料实例字典中获取材料实例
        mat_instance = self.__MAT_INSTANCE__.get(mat)
        # 如果材料实例不存在，则返回默认值
        if mat_instance is None:
            raise ValueError("Invalid material name")
        # 返回材料实例
        return mat_instance()

    def get_prop(
        self, mat: Union[str | int], temp: float
    ) -> Tuple[float, float, float, float, float, float, float]:
        """
        根据材料和温度获取属性。

        本函数尝试根据提供的材料标识符和温度，获取一组与材料特性相关的属性值。
        如果无法成功获取属性（例如，由于材料标识符无效或系统中不存在相应材料的数据），
        则返回一组默认值（全部为0）。

        参数:
        - mat (Optional[str | int]): 材料的标识符，可以是字符串或整型，允许为None。
        - temp (float): 温度值，用于查询材料在特定温度下的属性。

        返回:
        - Tuple[float, float, float, float, float, float, float]: 返回一个包含七个浮点数的元组，
        代表材料在指定温度下的七个属性值。如果查询失败，所有值将为0。
            1. 热导率
            1. 热膨胀系数
            1. 杨氏模量
            1. 泊松比
            1. 比热
            1. 密度
            1. 硬度
        """
        try:
            # 尝试获取材料实例并调用其get_prop方法来获取属性值。
            return self.get_mat_instance(mat).get_prop(temp)
        except:
            # 如果发生异常（材料实例获取失败或其它原因），则返回一组默认的属性值。
            return (0, 0, 0, 0, 0, 0, 0)

    def get_components(self, mat: Union[str | int]) -> List[str]:
        """
        根据给定的材料标识符获取材料的组成成分。

        本函数通过接收一个代表材料的字符串或整数标识符，尝试获取该材料的组成成分信息。
        如果获取成功，将组成成分信息以列表的形式返回，其中元素位置根据周期表中的顺序确定。
        如果过程中遇到任何错误，将捕获异常并打印出来，然后返回一个默认的组成成分列表。

        参数:
        mat: Optional[str | int] - 材料的标识符，可以是字符串或整数类型，表示要查询组成的材料。

        返回值:
        List[str] - 一个包含材料组成的列表，每个元素对应周期表中的一种元素的组成比例。
        """
        # 初始化一个默认的组成成分列表，长度为材料组成可能包含的最大元素数量，
        # 每个元素初始化为0.0，表示尚未赋值的组成比例。
        comp = [0.0 for i in range(self.max_length)]

        try:
            # 尝试根据给定的材料标识符获取材料实例，并调用该实例的get_components方法获取组成成分字典。
            comp_dict = self.get_mat_instance(mat).get_components()

            # 遍历组成成分字典，将每种元素的组成比例赋值到对应的位置上。
            for elem in comp_dict:
                comp[self.__PERIODIC_TABLE__[elem]] = comp_dict[elem]

        except Exception as e:
            # 捕获任何可能发生的异常，并将异常信息打印出来。
            print(f"{e}")

        # 无论过程是否成功，最终都返回组成成分列表。
        return comp
