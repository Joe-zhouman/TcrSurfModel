from ..Mat import Mat, MatSingleProp


class Steel2Cr12NiMoWV(Mat):
    """
    Chinese GB Steel 2Cr12NiMoWV(old) / 22Cr12NiMoWV(new)

    similar to  UNS S42200, AISI 422, AISI 616
    """

    __mat_name__ = "steel2cr12nimowv"
    __mat_id__ = 19

    __temp__data = [
        293.15,
        473.15,
        573.15,
        673.15,
        773.15,
        873.15,
    ]

    def get_thermal_conductivity(self, temp: float) -> float:
        conductivity_data = [
            24.3,
            27.7,
            28.1,
            29.1,
            29.1,
            29.7,
        ]

        return self.get_prop_from_interp(
            self.__temp__data, conductivity_data, "thermal_conductivity", temp
        )

    def get_thermal_expansion(self, temp: float) -> float:
        expansion_data = [
            0.00001038,
            0.00001082,
            0.00001121,
            0.00001149,
            0.00001182,
            0.00001206,
        ]

        return self.get_prop_from_interp(
            self.__temp__data, expansion_data, "expansion", temp
        )

    def get_specific_heat(self, temp: float) -> float:
        specific_heat_data = [
            530,
            585,
            627,
            663,
            721,
            860,
        ]
        return self.get_prop_from_interp(
            self.__temp__data, specific_heat_data, "specific_heat", temp
        )

    def get_elastic_modulus(self, temp: float) -> float:
        elastic_data = [
            216,
            205,
            198,
            189,
            178,
            170,
        ]
        return self.get_prop_from_interp(
            self.__temp__data, elastic_data, "elastic", temp
        )

    def get_poisson_ratio(self, temp: float) -> float:
        return (
            0.3793141
            - 9.068872e-4 * temp**1
            + 1.666833e-6 * temp**2
            - 1.053181e-9 * temp**3
        )

    def get_density(self, temp: float) -> float:
        return 7819.218 - 0.2229658 * temp**1 - 4.526658e-5 * temp**2

    def get_hardness(self, temp: float) -> float:
        temp_data = [
            573.4061475,
            601.5311475,
            628.2729508,
            647.1766393,
            666.5413934,
            681.2954918,
            695.127459,
            710.8036885,
            728.7852459,
            743.5393443,
            758.2934426,
            770.7422131,
            780.8856557,
            791.9512295,
            802.0946721,
            824.6868852,
            847.2790984,
            867.104918,
            892.9245902,
            923.354918,
            971.7668033,
            300,
        ]
        hardness_data = [
            4.29330846,
            4.27938252,
            4.25741484,
            4.24868661,
            4.25741484,
            4.27938252,
            4.34381451,
            4.42266279,
            4.53112821,
            4.59016635,
            4.62959049,
            4.60987842,
            4.55574378,
            4.471992,
            4.37343165,
            4.12609911,
            3.85140504,
            3.65379399,
            3.46746099,
            3.27034029,
            2.97681678,
            2.15754,
        ]
        return self.get_prop_from_interp(temp_data, hardness_data, "hardness", temp)


class Steel303(MatSingleProp):
    """
    AISI Steel 303
    """

    __mat_name__ = "ss303"
    __mat_id__ = 28
    __prop__ = (
        16.3,
        1.72e-05,
        193,
        0.25,
        500,
        8030,
        1.637769,
    )


class Steel304(Mat):
    """
    AISI Steel 304
    """

    __mat_id__ = 4
    __mat_name__ = "ss304"

    def get_thermal_conductivity(self, temp: float) -> float:
        temp_data = [
            2.613060581,
            3.628069103,
            5.159770191,
            6.560209803,
            8.681331231,
            12.15034597,
            16.73549146,
            22.68484443,
            29.54281752,
            40.36690871,
            56.95135425,
            80.34939643,
            118.937459,
            169.1505696,
            236.7422562,
            344.8731056,
            494.4135702,
            680.9891752,
            852.0696314,
            1145.766648,
            1399.594791,
        ]
        conductivity_data = [
            0.123620282,
            0.198390404,
            0.328941936,
            0.463318429,
            0.679750395,
            1.038792788,
            1.600480161,
            2.291365928,
            3.227414766,
            4.545850115,
            6.402881182,
            8.111308308,
            10.10933411,
            11.90037927,
            13.5591169,
            16.22383358,
            18.94304639,
            22.11801575,
            25.61536746,
            30.64944878,
            36.97316754,
        ]
        return self.get_prop_from_interp(
            temp_data, conductivity_data, "thermal_conductivity", temp
        )

    def get_thermal_expansion(self, temp: float) -> float:
        temp_data = [
            373.15,
            473.15,
            573.15,
            673.15,
            773.15,
            873.15,
            973.15,
            1073.15,
        ]
        expansion_data = [
            0.0000163,
            0.0000167,
            0.0000171,
            0.0000176,
            0.000018,
            0.0000183,
            0.000019,
            0.00002,
        ]

        return self.get_prop_from_interp(temp_data, expansion_data, "expansion", temp)

    def get_specific_heat(self, temp: float) -> float:
        temp_data = [
            293.15,
            363.15,
            473.15,
            593.15,
            703.15,
            813.15,
            923.15,
            1033.15,
            1143.15,
        ]
        specific_heat_data = [
            456,
            490,
            532,
            557,
            574,
            586,
            599,
            620,
            645,
        ]
        return self.get_prop_from_interp(
            temp_data, specific_heat_data, "specific_heat", temp
        )

    def get_elastic_modulus(self, temp: float) -> float:
        temp_data = [
            293.15,
            373.15,
            473.15,
            573.15,
            673.15,
            773.15,
        ]
        modulus_data = [
            200,
            194,
            186,
            179,
            172,
            165,
        ]
        return self.get_prop_from_interp(temp_data, modulus_data, "elastic", temp)

    def get_poisson_ratio(self, temp: float) -> float:
        temp_data = [
            423.15,
            533.15,
            643.15,
            753.15,
            863.15,
            973.15,
            1093.15,
        ]
        poisson_data = [
            0.28,
            0.3,
            0.32,
            0.28,
            0.29,
            0.28,
            0.25,
        ]
        return self.get_prop_from_interp(temp_data, poisson_data, "poisson_ratio", temp)

    def get_density(self, temp: float) -> float:
        temp_data = [
            293.15,
            363.15,
            473.15,
            593.15,
            703.15,
            813.15,
            923.15,
            1033.15,
            1143.15,
        ]
        density_data = [
            7910,
            7880,
            7840,
            7790,
            7740,
            7690,
            7640,
            7590,
            7540,
        ]
        return self.get_prop_from_interp(temp_data, density_data, "density", temp)

    def get_hardness(self, temp: float) -> float:
        temp_data = [
            3.018108652,
            9.356136821,
            13.8832998,
            17.80684105,
            23.23943662,
            30.18108652,
            37.72635815,
            44.06438632,
            49.79879276,
            56.74044266,
            62.17303823,
            66.09657948,
            69.11468813,
            71.83098592,
            73.34004024,
            75.15090543,
            77.86720322,
            82.3943662,
            88.4305835,
            97.18309859,
            107.444668,
            127.3641851,
            162.6760563,
            191.9517103,
            209.1549296,
            229.6780684,
            251.1066398,
            288.832998,
            292.3205069,
            373.4264977,
            473.7029954,
            573.2421659,
            673.5186636,
            773.0578341,
            873.3343318,
            973.6108295,
            1073.15,
        ]

        hardness_data = [
            4.905660377,
            4.578616352,
            4.402515723,
            4.238993711,
            4.062893082,
            3.987421384,
            3.974842767,
            4.012578616,
            4.088050314,
            4.188679245,
            4.339622642,
            4.490566038,
            4.641509434,
            4.465408805,
            4.238993711,
            3.987421384,
            3.761006289,
            3.58490566,
            3.471698113,
            3.371069182,
            3.308176101,
            3.144654088,
            2.943396226,
            2.691823899,
            2.566037736,
            2.402515723,
            2.226415094,
            1.962264151,
            1.879675,
            1.70455,
            1.5411,
            1.365975,
            1.2667375,
            1.23755,
            1.109125,
            0.9807,
            0.665475,
        ]
        return self.get_prop_from_interp(temp_data, hardness_data, "hardness", temp)


class Steel305(Steel304):
    """
    AISI steel 305
    """

    __mat_name__ = "ss305"
    __mat_id__ = 24

    def get_hardness(self, temp: float) -> float:
        return super().get_hardness(temp) * 2 / 3


class Steel416(MatSingleProp):
    """
    AISI Steel 416
    """

    __mat_id__ = 32
    __mat_name__ = "ss416"
    __prop__ = (
        24.9,
        9.90e-06,
        200,
        0.285,
        460,
        7700,
        1.8486195,
    )


class SteelC45(Steel304):
    """
    %0.45 Carbon, AISI steel 1045

    """

    __mat_name__ = "ssc45"
    __mat_id__ = 42

    def get_hardness(self, temp: float) -> float:
        return super().get_hardness(temp) / 2


class SteelEn58F(Steel304):
    """
    Steel En 58F

    similar to the AISI 302/304
    """

    __mat_name__ = "ssen58f"
    __mat_id__ = 25


class SteelGCr15(MatSingleProp):
    """
    Chinese GB steel GCr15 high carbon Cr-steel


    """

    __mat_id__ = 23
    __mat_name__ = "steelgcr15"
    __prop__ = (
        46.6,
        1.19e-05,
        200,
        0.285,
        475,
        7810,
        8.316336,
    )


class MildSteel(MatSingleProp):
    """
    Mild Steel

    Similar to AISI
    """

    __mat_name__ = "ms"
    __mat_id__ = 43

    __prop__ = (
        51.9,
        1.00e-05,
        200,
        0.29,
        486,
        7870,
        1.284717,
    )


class SteelT10(MatSingleProp):
    """
    Chinese GB steel T10

    Equalivent to USA W110, EU 1.1535
    """

    __mat_name__ = "steelt10"
    __mat_id__ = 22
    __prop__ = (
        48.3,
        1.44e-05,
        200,
        0.285,
        477,
        7850,
        5.207517,
    )
