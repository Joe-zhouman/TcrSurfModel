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
        "",
        "cu",
        "al",
        "c",
        "ss304",
        "si",
        "ti",
        "co",
        "aln",
        "al6061t6",
        "cu110",
        "brass360",
        "cuw30w3",
        "al6061",
        "al5052",
        "brass",
        "ni200",
        "zrnb",
        "zr4",
        "steel2cr12nimowv",
        "albh137",
        "tial",
        "steelt10",
        "steelgcr15",
        "ss305",
        "ssen58f",
        "albshe20",
        "brass271",
        "ss303",
        "",
        "mgaz31b",
        "al2024t4",
        "ss416",
        "al75st6",
        "al5a05h112",
        "al3a21h112",
        "al3a21",
        "al6063",
        "al6061h112",
        "ti6al4v",
        "cc",
        "in600",
        "ssc45",
        "ms",
    ]

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

    def __init__(self) -> None:
        super().__init__()

    def get_mat_list(self) -> List[str]:
        return self.__MAT_LIST__

    def get_mat_instance(self) -> Dict[str, Mat]:
        return self.__MAT_Instance__

    def get_prop(
        self, mat: Optional[str | int], temp: float
    ) -> Tuple[float, float, float, float, float, float, float]:
        if isinstance(mat, int):
            if mat < 0 or mat >= len(self.__MAT_LIST__):
                return 0, 0, 0, 0, 0, 0, 0
            mat = self.__MAT_LIST__[mat]

        mat_instance = self.__MAT_Instance__.get(mat)
        if mat_instance is None:
            return 0, 0, 0, 0, 0, 0, 0

        m = mat_instance()
        return m.get_prop(temp)
