from ..mat import Mat, MatSingleProp


class Steel2Cr12NiMoWV(Mat):
    """
    Chinese GB Steel 2Cr12NiMoWV(old) / 22Cr12NiMoWV(new)

    similar to  UNS S42200, AISI 422, AISI 616
    """

    __mat_name__ = "steel2cr12nimowv"
    __mat_id__ = 19
    __components__ = {
        "C": 0.00225,
        "Si": 0.0025,
        "Mn": 0.0075,
        "P": 0.0002,
        "S": 0.00015,
        "Cr": 0.12,
        "Ni": 0.01,
        "Mo": 0.01,
        "W": 0.01,
        "V": 0.3,
    }
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


class Steel303(Mat):
    """
    AISI Steel 303
    """

    __mat_name__ = "ss303"
    __mat_id__ = 28
    __components__ = {
        "C": 0.00075,
        "Si": 0.005,
        "Mn": 0.01,
        "P": 0.001,
        "S": 0.00075,
        "Cr": 0.18,
        "Ni": 0.09,
        "Mo": 0.003,
        "Fe": 0.7095,
    }
    def get_thermal_conductivity(self, temp: float) -> float:
        # 4-1173
        if temp < 10.0:
            return (
                -0.6609524
                + 0.4874603 * temp**1
                - 0.1045833 * temp**2
                + 0.01111111 * temp**3
                - 4.166667e-4 * temp**4
            )
        if temp < 40.0:
            return (
                -0.43828
                + 0.1089687 * temp**1
                + 6.012966e-4 * temp**2
                - 1.105985e-5 * temp**3
                - 5.725505e-8 * temp**4
            )
        if temp < 373.0:
            return (
                -0.7621893
                + 0.1386544 * temp**1
                - 5.102006e-4 * temp**2
                + 8.967006e-7 * temp**3
                - 5.960198e-10 * temp**4
            )
        if temp < 473.0:
            return (
                3.006336
                + 0.08012783 * temp**1
                - 1.915e-4 * temp**2
                + 1.666667e-7 * temp**3
            )
        return 7.202893 + 0.01796429 * temp**1

    def get_thermal_expansion(self, temp: float) -> float:
        # 0-922
        if temp < 20.0:
            return 1.032014e-5 + 3.611886e-8 * temp**1
        return (
            1.012469e-5
            + 4.819421e-8 * temp**1
            - 1.21971e-10 * temp**2
            + 1.381049e-13 * temp**3
            - 5.570179e-17 * temp**4
        )

    def get_specific_heat(self, temp: float) -> float:
        return 500

    def get_elastic_modulus(self, temp: float) -> float:
        # 0.0	593.0
        return (2.126806e11 - 6.761325e7 * temp**1 - 14194.14 * temp**2) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        return 0.25

    def get_density(self, temp: float) -> float:
        # 0-922
        if temp < 20.0:
            return 7901.461 - 0.01112026 * temp**1
        return (
            7900.121
            + 0.09441926 * temp**1
            - 0.001994032 * temp**2
            + 3.88437e-6 * temp**3
            - 3.687574e-9 * temp**4
            + 1.329766e-12 * temp**5
        )

    def get_hardness(self, temp: float) -> float:
        return 1.637769


class Steel304(Mat):
    """
    AISI Steel 304
    """

    __mat_id__ = 4
    __mat_name__ = "ss304"
    __components__ = {
        "C": 0.0004,
        "Si": 0.005,
        "Mn": 0.01,
        "P": 0.000225,
        "S": 0.00015,
        "Cr": 0.19,
        "Ni": 0.095,
        "Fe": 0.68345,
    }
    def get_thermal_conductivity(self, temp: float) -> float:
        # 1-887K
        if temp < 45.0:
            return (
                -0.03740871
                + 0.06460546 * temp**1
                + 0.003720604 * temp**2
                - 8.390067e-5 * temp**3
                + 6.006594e-7 * temp**4
            )
        if temp < 283.0:
            return (
                -1.031521
                + 0.1813807 * temp**1
                - 0.001088656 * temp**2
                + 3.411681e-6 * temp**3
                - 3.988389e-9 * temp**4
            )
        return (
            -1.258169
            + 0.1023945 * temp**1
            - 2.189547e-4 * temp**2
            + 2.312931e-7 * temp**3
            - 8.903937e-11 * temp**4
        )

    def get_thermal_expansion(self, temp: float) -> float:
        # 4-1700K
        if temp < 249.0:
            return (
                1.017247e-5
                + 4.390756e-8 * temp**1
                - 9.800292e-11 * temp**2
                - 9.637483e-14 * temp**3
                + 4.333114e-16 * temp**4
            )

        return 1.356267e-5 + 6.945622e-9 * temp**1 - 1.444708e-12 * temp**2

    def get_specific_heat(self, temp: float) -> float:
        # 4-1311K
        if temp < 14.0:
            return (
                4.80349978
                - 1.97398861 * temp**1
                + 0.433444409 * temp**2
                - 0.0314324757 * temp**3
                + 8.32403453e-4 * temp**4
            )
        if temp < 47.0:
            return (
                -0.224295746
                + 0.760568357 * temp**1
                - 0.0400750758 * temp**2
                + 0.00218176061 * temp**3
                - 1.83602372e-5 * temp**4
            )
        if temp < 128.0:
            return (
                8.9262753
                - 2.90098686 * temp**1
                + 0.147079315 * temp**2
                - 0.00125489708 * temp**3
                + 3.41401137e-6 * temp**4
            )
        if temp < 310.0:
            return (
                270.215021
                - 1.21051111 * temp**1
                + 0.0215156635 * temp**2
                - 7.51184063e-5 * temp**3
                + 8.13679634e-8 * temp**4
            )
        return (
            109.207295
            + 2.5717751 * temp**1
            - 0.00652809855 * temp**2
            + 7.78752439e-6 * temp**3
            - 4.16791252e-9 * temp**4
            + 8.09061335e-13 * temp**5
        )

    def get_elastic_modulus(self, temp: float) -> float:
        # 5-1173K
        if temp < 60.0:
            return (2.078364e11 + 7.414909e7 * temp**1) / 1e9
        if temp < 295.0:
            return (
                2.077579e11
                + 1.570552e8 * temp**1
                - 1651323.0 * temp**2
                + 5296.944 * temp**3
                - 6.38774 * temp**4
            ) / 1e9
        return (2.242833e11 - 8.92503e7 * temp**1) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        # 5-1173K
        if temp < 60.0:
            return 0.2797496 - 4.402647e-5 * temp**1
        if temp < 295.0:
            return (
                0.2752731
                + 2.459471e-5 * temp**1
                + 1.356199e-7 * temp**2
                - 1.850087e-10 * temp**3
            )
        return 0.2652448 + 8.224213e-5 * temp**1

    def get_density(self, temp: float) -> float:
        # 4-1700K
        if temp < 93.0:
            return (
                7930.967
                + 0.03300298 * temp**1
                - 9.663581e-4 * temp**2
                - 2.917178e-6 * temp**3
            )
        return (
            7945.333
            - 0.1981948 * temp**1
            - 3.713764e-4 * temp**2
            + 2.213069e-7 * temp**3
            - 5.128456e-11 * temp**4
        )

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


class Steel305(Mat):
    """
    AISI steel 305
    """

    __mat_name__ = "ss305"
    __mat_id__ = 24
    __components__ = {
        "Cr": 0.0006,
        "Cr": 0.18,
        "Ni": 0.12,
        "Mg": 0.01,
        "Si": 0.00375,
        "P": 0.000225,
        "S": 0.00015,
    }
    def get_hardness(self, temp: float) -> float:
        return Steel304().get_hardness(temp) * 0.7358

    def get_thermal_conductivity(self, temp: float) -> float:
        # 293.0	773.0
        return 11.451 + 0.013 * temp**1

    def get_thermal_expansion(self, temp: float) -> float:
        # 293.0	773.0
        return 1.4335e-5 + 5.0e-9 * temp**1

    def get_specific_heat(self, temp: float) -> float:
        # 40-300
        if temp < 18.0:
            return (
                0.363452365
                + 0.26377074 * temp**1
                + 0.0493134608 * temp**2
                - 0.0038147097 * temp**3
                + 1.19554913e-4 * temp**4
            )
        if temp < 50.0:
            return (
                -14.0795868
                + 2.9024659 * temp**1
                - 0.153359541 * temp**2
                + 0.004588802 * temp**3
                - 3.66629778e-5 * temp**4
            )
        if temp < 140.0:
            return (
                -20.5016084
                - 0.832746541 * temp**1
                + 0.0955618906 * temp**2
                - 7.74522415e-4 * temp**3
                + 1.944414e-6 * temp**4
            )
        if temp < 300.0:
            return (
                -75.5829977
                + 5.00692586 * temp**1
                - 0.0164947547 * temp**2
                + 2.02748649e-5 * temp**3
            )
        return Steel304().get_specific_heat(temp)

    def get_elastic_modulus(self, temp: float) -> float:
        # 293.0	673.0
        return (2.256255e11 - 9.345614e7 * temp**1 + 20467.84 * temp**2) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        return Steel304().get_poisson_ratio(temp)

    def get_density(self, temp: float) -> float:
        # 293.0	773.0
        return 8001.375 - 0.3162285 * temp**1 - 1.011579e-4 * temp**2


class Steel410(Mat):
    """
    AISI Steel 410
    """

    #! THIS MATERIAL IS NOT LIST IN DATABASE
    #! SO IT IS JUST A PLACEHOLDER AND NOT BE FULLY IMPLEMENTED

    def get_thermal_conductivity(self, temp: float) -> float:
        # 25-1600K
        if temp < 94.0:
            return (
                -10.05532
                + 0.7824252 * temp**1
                - 0.00772505 * temp**2
                + 3.110218e-5 * temp**3
            )
        if temp < 250.0:
            return (
                0.6567181
                + 0.3436629 * temp**1
                - 0.001574372 * temp**2
                + 2.428653e-6 * temp**3
            )
        if temp < 336.0:
            return 24.73655 + 0.005541401 * temp**1
        return (
            29.69488
            - 0.0418147 * temp**1
            + 1.656612e-4 * temp**2
            - 2.562328e-7 * temp**3
            + 1.675486e-10 * temp**4
            - 3.852217e-14 * temp**5
        )

    def get_thermal_expansion(self, temp: float) -> float:
        # 0-921K
        if temp < 20.0:
            return 6.044369e-6 + 2.096765e-8 * temp**1
        return (
            5.930321e-6
            + 2.802308e-8 * temp**1
            - 5.650723e-11 * temp**2
            + 5.290395e-14 * temp**3
            - 1.860061e-17 * temp**4
        )

    def get_specific_heat(self, temp: float) -> float:
        # 255-1366K
        if temp < 1005.0:
            return 475.908662 - 0.233225909 * temp**1 + 6.59091294e-4 * temp**2
        if temp < 1116.0:
            return (
                -1.43817176e7
                + 53613.9015 * temp**1
                - 74.8565716 * temp**2
                + 0.0463964178 * temp**3
                - 1.07711432e-5 * temp**4
            )
        if temp < 1200.0:
            return -71009.3836 + 125.406572 * temp**1 - 0.0547314061 * temp**2
        return 1366.0 - 742.930705 + 2.08570601 * temp**1 - 7.60158325e-4 * temp**2

    def get_elastic_modulus(self, temp: float) -> float:
        # 20-1173K
        if temp < 293.0:
            return (
                2.192748e11
                - 3.992256e7 * temp**1
                + 465394.5 * temp**2
                - 2049.499 * temp**3
                + 2.239098 * temp**4
                - 1.059059e-4 * temp**5
            ) / 1e9
        return (2.263948e11 - 3.264378e7 * temp**1 - 53828.09 * temp**2) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        # 293.0	1173.0
        return 0.2524185 + 1.215255e-4 * temp**1

    def get_density(self, temp: float) -> float:
        # 0-921
        if temp < 20.0:
            return 7690.789 - 0.007508503 * temp**1
        return (
            7689.941
            + 0.0550138 * temp**1
            - 0.001051382 * temp**2
            + 1.745929e-6 * temp**3
            - 1.460243e-9 * temp**4
            + 4.842266e-13 * temp**5
        )

    def get_hardness(self, temp: float) -> float:
        #! NOT IMPLEMENTED
        raise NotImplementedError


class Steel416(Mat):
    """
    AISI Steel 416
    """

    __mat_id__ = 32
    __mat_name__ = "ss416"
    __components__ = {
        "C": 0.00075,
        "Mn": 0.00625,
        "Si": 0.005,
        "P": 0.0003,
        "S": 0.0015,
        "Cr": 0.13,
        "Mo": 0.003,
        "Fe": 0.8532,
    }
    def get_thermal_conductivity(self, temp: float) -> float:
        # 316.0	473.0
        if 316 < temp < 473:
            return 25.31899 - 3.157169e-4 * temp**1
        else:
            return Steel410().get_thermal_conductivity(temp)

    def get_thermal_expansion(self, temp: float) -> float:
        # 0-780K
        if temp < 20.0:
            return 6.421949e-6 + 2.130395e-8 * temp**1
        return (
            6.233979e-6
            + 3.273792e-8 * temp**1
            - 1.04808e-10 * temp**2
            + 1.478016e-13 * temp**3
            - 7.278695e-17 * temp**4
        )

    def get_specific_heat(self, temp: float) -> float:
        return Steel410().get_specific_heat(temp)

    def get_elastic_modulus(self, temp: float) -> float:
        return Steel410().get_elastic_modulus(temp)

    def get_poisson_ratio(self, temp: float) -> float:
        return Steel410().get_poisson_ratio(temp)

    def get_density(self, temp: float) -> float:
        # 0-780K
        if temp < 20.0:
            return 7683.301 - 0.01413985 * temp**1
        return (
            7682.029
            + 0.07720145 * temp**1
            - 0.001463129 * temp**2
            + 3.423345e-6 * temp**3
            - 3.904904e-9 * temp**4
            + 1.680162e-12 * temp**5
        )

    def get_hardness(self, temp: float) -> float:
        return 1.8486195


class SteelC45(Mat):
    """
    %0.45 Carbon, AISI steel 1045

    """

    __mat_name__ = "ssc45"
    __mat_id__ = 42
    __components__ = {
        "C": 0.0045,
        "Fe": 0.987,
        "Mn": 0.00075,
        "R": 0.0002,
        "S": 0.00025,
    }
    def get_hardness(self, temp: float) -> float:
        return Steel304().get_hardness(temp) * 0.6981

    def get_thermal_conductivity(self, temp: float) -> float:
        # 273-1473
        if temp < 1073.0:
            return (
                51.89812
                + 0.01329408 * temp**1
                - 5.098658e-5 * temp**2
                + 1.414141e-8 * temp**3
            )
        return 27.32279 - 0.01296 * temp**1 + 1.0e-5 * temp**2

    def get_thermal_expansion(self, temp: float) -> float:
        # 0-1144K
        if temp < 680.0:
            return (
                6.731035e-6
                + 3.918624e-8 * temp**1
                - 1.534406e-10 * temp**2
                + 3.517965e-13 * temp**3
                - 3.827522e-16 * temp**4
                + 1.565546e-19 * temp**5
            )
        if temp < 977.0:
            return (
                8.614185e-6
                + 1.013631e-8 * temp**1
                - 2.26181e-12 * temp**2
                - 1.569696e-15 * temp**3
            )
        if temp < 1061.0:
            return 6.292031e-5 - 4.915637e-8 * temp**1
        return -1.563678e-6 + 1.162025e-8 * temp**1

    def get_specific_heat(self, temp: float) -> float:
        # 293.0	948.0
        return (
            490.246811
            - 0.252192106 * temp**1
            + 7.49883258e-4 * temp**2
            - 1.80281531e-7 * temp**3
        )

    def get_elastic_modulus(self, temp: float) -> float:
        # 4-1500K
        if temp < 273.0:
            return (
                2.217366e11
                + 5020008.0 * temp**1
                - 305140.4 * temp**2
                + 926.6601 * temp**3
                - 1.145454 * temp**4
            ) / 1e9
        if temp < 1050.0:
            return (2.109875e11 + 3.572844e7 * temp**1 - 106319.6 * temp**2) / 1e9
        return (2.024261e11 - 6.77381e7 * temp**1) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        if temp < 120.0:
            return 0.2850355 - 1.662951e-6 * temp**1
        if temp < 273.0:
            return 0.2848011 - 7.147353e-6 * temp**1 + 6.558945e-8 * temp**2
        if temp < 1053.0:
            return (
                0.2712267
                + 7.030261e-5 * temp**1
                - 3.856929e-8 * temp**2
                + 1.246582e-11 * temp**3
            )
        return 0.3165268 - 1.242823e-6 * temp**1 + 1.661461e-9 * temp**2

    def get_density(self, temp: float) -> float:
        # 0-1144
        if temp < 60.0:
            return 7907.978 - 0.01549395 * temp**1
        if temp < 977.0:
            return (
                7911.3
                - 0.01678428 * temp**1
                - 8.018711e-4 * temp**2
                + 1.172796e-6 * temp**3
                - 1.015971e-9 * temp**4
                + 3.677737e-13 * temp**5
            )
        if temp < 1061.0:
            return 7116.994 + 0.5195388 * temp**1
        return 8166.523 - 0.4696497 * temp**1


class SteelEn58F(Steel304):
    """
    Steel En 58F

    similar to the AISI 302/304
    """

    __mat_name__ = "ssen58f"
    __mat_id__ = 25


class SteelGCr15(Mat):
    """
    Chinese GB steel GCr15 high carbon Cr-steel
    GB/T 18254-2016
    """

    __mat_id__ = 23
    __mat_name__ = "steelgcr15"
    __components__ = {
        "C": 0.01,
        "Si": 0.0025,
        "Mn": 0.0035,
        "Cr": 0.015,
        "Mo": 0.0005,
        "P": 0.000125,
        "S": 0.000125,
        "Ni": 0.0015,
        "Cu": 0.00125,
    }
    def get_thermal_conductivity(self, temp: float) -> float:
        # 293.0	973.0
        return 33.74749 - 0.002610848 * temp**1 + 2.037179e-7 * temp**2

    def get_thermal_expansion(self, temp: float) -> float:
        # 293.0	922.0
        return 4.656794e-7 + 2.77374e-8 * temp**1 - 1.402175e-11 * temp**2

    def get_specific_heat(self, temp: float) -> float:
        return 475

    def get_elastic_modulus(self, temp: float) -> float:
        return 200

    def get_poisson_ratio(self, temp: float) -> float:
        return 0.285

    def get_density(self, temp: float) -> float:
        # 293.0	922.0
        return (
            7804.373
            + 0.172543 * temp**1
            - 7.362584e-4 * temp**2
            + 3.294825e-7 * temp**3
        )

    def get_hardness(self, temp: float) -> float:
        return 8.316336


class Steel1018(Mat):
    __mat_name__ = "ss1018"
    __components__ = {
        "C": 0.00175,
        "Mn": 0.0075,
        "P": 0.0002,
        "S": 0.00025,
        "Si": 0.0025,
        "Zn": 0.0015,
        "Cu": 0.001,
        "Fe": 0.9853,
    }
    def get_thermal_conductivity(self, temp: float) -> float:
        # 123.0	813.0
        return (
            36.11328
            + 0.1134704 * temp**1
            - 2.36859e-4 * temp**2
            + 1.238751e-7 * temp**3
        )

    def get_thermal_expansion(self, temp: float) -> float:
        # 0-1144
        if temp < 680.0:
            return (
                6.731035e-6
                + 3.918624e-8 * temp**1
                - 1.534406e-10 * temp**2
                + 3.517965e-13 * temp**3
                - 3.827522e-16 * temp**4
                + 1.565546e-19 * temp**5
            )
        if temp < 977.0:
            return (
                8.614185e-6
                + 1.013631e-8 * temp**1
                - 2.26181e-12 * temp**2
                - 1.569696e-15 * temp**3
            )
        if temp < 1061.0:
            return 6.292031e-5 - 4.915637e-8 * temp**1
        return -1.563678e-6 + 1.162025e-8 * temp**1

    def get_specific_heat(self, temp: float) -> float:
        # 293.0	848.0
        return (
            -215.730638
            + 6.0184999 * temp**1
            - 0.0183429321 * temp**2
            + 2.414973e-5 * temp**3
            - 1.07882432e-8 * temp**4
        )

    def get_elastic_modulus(self, temp: float) -> float:
        # 4-1500
        if temp < 273.0:
            return (
                2.217366e11
                + 5020008.0 * temp**1
                - 305140.4 * temp**2
                + 926.6601 * temp**3
                - 1.145454 * temp**4
            ) / 1e9
        if temp < 1050.0:
            return (2.109875e11 + 3.572844e7 * temp**1 - 106319.6 * temp**2) / 1e9
        return (2.024261e11 - 6.77381e7 * temp**1) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        # 4-1500
        if temp < 120.0:
            return 0.2850355 - 1.662951e-6 * temp**1
        if temp < 273.0:
            return 0.2848011 - 7.147353e-6 * temp**1 + 6.558945e-8 * temp**2
        if temp < 1053.0:
            return (
                0.2712267
                + 7.030261e-5 * temp**1
                - 3.856929e-8 * temp**2
                + 1.246582e-11 * temp**3
            )
        return 0.3165268 - 1.242823e-6 * temp**1 + 1.661461e-9 * temp**2

    def get_density(self, temp: float) -> float:
        # 0-1144
        if temp < 60.0:
            return 7907.978 - 0.01549395 * temp**1
        if temp < 977.0:
            return (
                7911.3
                - 0.01678428 * temp**1
                - 8.018711e-4 * temp**2
                + 1.172796e-6 * temp**3
                - 1.015971e-9 * temp**4
                + 3.677737e-13 * temp**5
            )
        if temp < 1061.0:
            return 7116.994 + 0.5195388 * temp**1
        return 8166.523 - 0.4696497 * temp**1

    def get_hardness(self, temp: float) -> float:
        return 1.284717


class MildSteel(Steel1018):
    """
    Mild Steel

    Similar to AISI 1018
    """

    __mat_name__ = "ms"
    __mat_id__ = 43


class SteelT10(Mat):
    """
    Chinese GB steel T10

    Equalivent to USA W1 Tool Steel, EU 1.1535
    """

    __mat_name__ = "steelt10"
    __mat_id__ = 22
    __components__ = {
        "C": 0.01,
        "Si": 0.0027,
        "Mn": 0.005,
        "S": 0.0001,
        "P": 0.00015,
        "Cr": 0.00125,
        "Ni": 0.001,
        "Cu": 0.0015,
        "Fe": 0.9783,
    }
    def get_thermal_conductivity(self, temp: float) -> float:
        # 293-973
        if temp < 715.0:
            return (
                92.92772
                - 0.1995037 * temp**1
                + 2.622541e-4 * temp**2
                - 1.292243e-7 * temp**3
            )
        return 44.89882 + 0.004904424 * temp**1 - 2.207792e-5 * temp**2

    def get_thermal_expansion(self, temp: float) -> float:
        # 100.0	973.0
        return 8.687241e-6 + 5.904846e-9 * temp**1 + 3.9615e-13 * temp**2

    def get_specific_heat(self, temp: float) -> float:
        # 1-1809
        if temp < 20.0:
            return (
                0.0164077652
                + 0.0734379027 * temp**1
                + 0.00354351328 * temp**2
                + 3.43411423e-5 * temp**3
                + 8.78682677e-6 * temp**4
            )
        if temp < 130.0:
            return (
                -4.12996699
                + 0.900549934 * temp**1
                - 0.0558817132 * temp**2
                + 0.00197232379 * temp**3
                - 1.8078495e-5 * temp**4
                + 5.25176517e-8 * temp**5
            )
        if temp < 500.0:
            return (
                -143.811779
                + 5.19120595 * temp**1
                - 0.0178093048 * temp**2
                + 2.86828262e-5 * temp**3
                - 1.68314078e-8 * temp**4
            )
        if temp < 1000.0:
            return (
                3998.51031
                - 21.8908093 * temp**1
                + 0.050237037 * temp**2
                - 4.9900476e-5 * temp**3
                + 1.85286976e-8 * temp**4
            )
        if temp < 1042.0:
            return (
                -560642.611
                + 1783.88815 * temp**1
                - 1.88950988 * temp**2
                + 6.67236706e-4 * temp**3
            )
        if temp < 1184.0:
            return (
                5927267.78
                - 20371.737 * temp**1
                + 26.2589723 * temp**2
                - 0.0150426432 * temp**3
                + 3.23124797e-6 * temp**4
            )
        if temp < 1665.0:
            return 429.530277 + 0.149681052 * temp**1
        return 440.284412 + 0.177820042 * temp**1

    def get_elastic_modulus(self, temp: float) -> float:
        # 4-1500
        if temp < 273.0:
            return (
                2.217366e11
                + 5020008.0 * temp**1
                - 305140.4 * temp**2
                + 926.6601 * temp**3
                - 1.145454 * temp**4
            ) / 1e9
        if temp < 1050.0:
            return (2.109875e11 + 3.572844e7 * temp**1 - 106319.6 * temp**2) / 1e9
        return (2.024261e11 - 6.77381e7 * temp**1) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        # 4-1500
        if temp < 120.0:
            return 0.2850355 - 1.662951e-6 * temp**1
        if temp < 273.0:
            return 0.2848011 - 7.147353e-6 * temp**1 + 6.558945e-8 * temp**2
        if temp < 1053.0:
            return (
                0.2712267
                + 7.030261e-5 * temp**1
                - 3.856929e-8 * temp**2
                + 1.246582e-11 * temp**3
            )
        return 0.3165268 - 1.242823e-6 * temp**1 + 1.661461e-9 * temp**2

    def get_density(self, temp: float) -> float:
        # 100.0	973.0
        return (
            7882.37
            - 0.09435002 * temp**1
            - 2.977779e-4 * temp**2
            + 1.001517e-7 * temp**3
        )

    def get_hardness(self, temp: float) -> float:
        return 5.207517
