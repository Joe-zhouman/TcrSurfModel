from ..Mat import Mat


class Brass(Mat):
    """
    Brass with Cu:Zn=3:1
    """

    __mat_name__ = "brass"
    __mat_id__ = 15
    __components__ = {"Cu": 0.66, "Zn": 0.34}
    def get_thermal_conductivity(self, temp: float) -> float:
        temp_data = [
            1.570410857,
            1.930697729,
            2.391869834,
            2.940617505,
            3.479591323,
            4.310736681,
            5.634178017,
            6.616038239,
            7.592742493,
            9.053377534,
            10.8778947,
            13.17047168,
            16.69518985,
            21.00192659,
            25.82023157,
            32.73031357,
            36.70997199,
            48.72004056,
            69.26782666,
            88.47974327,
            114.7626755,
            151.1478568,
            180.2245515,
            226.7157687,
            276.6052795,
            347.9591323,
            451.3204888,
            559.1242204,
        ]
        kappa_data = [
            1.064209244,
            1.323188207,
            1.69718713,
            2.193896909,
            2.644218244,
            3.365325118,
            4.487733171,
            5.623413252,
            6.88395207,
            8.693390519,
            10.97843767,
            13.65007807,
            17.7827941,
            22.80909784,
            27.49091463,
            33.91606082,
            37.81836096,
            43.50242966,
            52.02549442,
            59.3811561,
            70.46492159,
            81.68873338,
            91.79897623,
            107.2520015,
            122.4159023,
            134.3935353,
            145.2653926,
            154.5927736,
        ]
        return self.get_prop_from_interp(
            temp_data, kappa_data, "thermal_conductivity", temp
        )

    def get_thermal_expansion(self, temp: float) -> float:
        temp_data = [
            31.14186851,
            58.82352941,
            91.69550173,
            122.8373702,
            158.3044983,
            185.9861592,
            220.5882353,
            246.5397924,
            275.0865052,
            302.7681661,
            335.6401384,
            371.9723183,
            408.3044983,
            443.7716263,
            475.7785467,
            512.9757785,
            550.1730104,
            576.9896194,
            608.9965398,
            646.1937716,
            682.5259516,
            721.4532872,
            760.3806228,
            801.0380623,
            844.2906574,
            888.4083045,
            921.2802768,
            955.8823529,
        ]
        expansion_data = [
            -3.85255e-07,
            -3.71314e-07,
            -3.36461e-07,
            -3.05094e-07,
            -2.4933e-07,
            -2.07507e-07,
            -1.48257e-07,
            -9.24933e-08,
            -3.67292e-08,
            1.55496e-08,
            7.47989e-08,
            1.23592e-07,
            1.89812e-07,
            2.52547e-07,
            3.08311e-07,
            3.71046e-07,
            4.54692e-07,
            5.06971e-07,
            5.76676e-07,
            6.53351e-07,
            7.30027e-07,
            8.20643e-07,
            9.00804e-07,
            9.7748e-07,
            1.06113e-06,
            1.15871e-06,
            1.22493e-06,
            1.29464e-06,
        ]
        return self.get_prop_from_interp(temp_data, expansion_data, "expansion", temp)

    def get_specific_heat(self, temp: float) -> float:
        temp_data = [
            25.47770701,
            114.6496815,
            210.1910828,
            286.6242038,
            356.6878981,
            418.2590234,
            489.3842887,
            562.6326964,
            618.895966,
            682.5902335,
            750.5307856,
            820.5944798,
            869.4267516,
            907.6433121,
        ]
        specific_heat_data = [
            396.3949843,
            410.031348,
            423.1974922,
            434.9529781,
            444.3573668,
            452.3510972,
            462.6959248,
            472.1003135,
            480.0940439,
            488.5579937,
            497.492163,
            506.8965517,
            512.0689655,
            516.3009404,
        ]
        return self.get_prop_from_interp(
            temp_data, specific_heat_data, "specific_heat", temp
        )

    def get_elastic_modulus(self, temp: float) -> float:
        temp_data = [
            148.9324491,
            175.7926426,
            202.6528362,
            231.2647814,
            255.789306,
            283.2334168,
            308.9257758,
            339.8733901,
            370.8210043,
            396.5133634,
            419.2861361,
            440.8910744,
            468.3351852,
            490.5240407,
            512.7128962,
        ]
        elastic_modulus_data = [
            103.066,
            101.9998,
            100.5782,
            99.3343,
            97.9127,
            96.8465,
            95.6026,
            94.181,
            92.7594,
            91.8709,
            90.4493,
            89.5608,
            88.3169,
            87.2507,
            86.3622,
        ]
        return self.get_prop_from_interp(
            temp_data, elastic_modulus_data, "elastic", temp
        )

    def get_poisson_ratio(self, temp: float) -> float:
        return 0.334

    def get_density(self, temp: float) -> float:
        temp_data = [
            40.60913706,
            95.43147208,
            158.3756345,
            217.2588832,
            282.2335025,
            343.1472081,
            418.2741117,
            509.6446701,
            609.1370558,
            710.6598985,
            797.9695431,
            858.8832487,
        ]
        density_data = [
            8241.435563,
            8226.75367,
            8197.389886,
            8177.814029,
            8158.238173,
            8133.768352,
            8104.404568,
            8075.040783,
            8021.207178,
            7977.161501,
            7933.115824,
            7903.752039,
        ]
        return self.get_prop_from_interp(temp_data, density_data, "density", temp)

    def get_hardness(self, temp: float) -> float:
        temp_data = [
            1.492537313,
            13.13432836,
            27.46268657,
            38.50746269,
            52.8358209,
            66.26865672,
            79.10447761,
            87.46268657,
            93.13432836,
            99.10447761,
            105.3731343,
            112.5373134,
            122.3880597,
            130.4477612,
            143.2835821,
            153.4328358,
            164.1791045,
            178.2089552,
            191.9402985,
            204.4776119,
            219.7014925,
            234.3283582,
            254.6268657,
            292.5373134,
        ]
        hardness_data = [
            2.092154421,
            2.079701121,
            2.092154421,
            2.123287671,
            2.204234122,
            2.297633873,
            2.415940224,
            2.584059776,
            2.739726027,
            3.03237858,
            2.801992528,
            2.602739726,
            2.453300125,
            2.347447073,
            2.210460772,
            2.110834371,
            1.99875467,
            1.867995019,
            1.755915318,
            1.662515567,
            1.562889166,
            1.488169365,
            1.438356164,
            1.407222914,
        ]
        return self.get_prop_from_interp(temp_data, hardness_data, "hardness", temp)


class Brass271(Brass):
    """
    anaconda alloy 271
    """

    __mat_name__ = "brass271"
    __mat_id__ = 27
    __components__ = {"Cu": 0.62, "Zn": 0.35, "Pb": 0.03}

class Brass360(Mat):
    """
    Brass 360,C110, GB-HPb60-2,UNS C37700

    """

    __mat_name__ = "brass360"
    __mat_id__ = 11
    __components__ = {"Cu": 0.59, "Fe": 0.003, "Pb": 0.02, "Zn": 0.387}
    def get_thermal_conductivity(self, temp: float) -> float:

        # 293.0	500.0
        return 48.3474 + 0.2323907 * temp**1 - 1.064614e-4 * temp**2

    def get_thermal_expansion(self, temp: float) -> float:
        return 2.05e-05

    def get_specific_heat(self, temp: float) -> float:
        return 380.16144

    def get_elastic_modulus(self, temp: float) -> float:
        return 97

    def get_poisson_ratio(self, temp: float) -> float:
        return 0.31

    def get_density(self, temp: float) -> float:
        return 8490

    def get_hardness(self, temp: float) -> float:
        return 1.383


class Cu(Mat):
    """Pure Copper"""

    __mat_id__ = 1
    __mat_name__ = "cu"
    __components__ = {"Cu": 1}
    def get_thermal_conductivity(self, temp: float) -> float:
        # 1-1358 K
        if temp < 40.0:
            return (
                12.55868
                + 36.66487 * temp**1
                + 1.387207 * temp**2
                - 0.07168113 * temp**3
                + 6.99799e-4 * temp**4
            )
        if temp < 70.0:
            return (
                2174.919
                - 45.25448 * temp**1
                + 0.3738471 * temp**2
                - 9.504397e-4 * temp**3
            )
        if temp < 100.0:
            return (
                2545.87
                - 67.53869 * temp**1
                + 0.8176488 * temp**2
                - 0.004470238 * temp**3
                + 9.22619e-6 * temp**4
            )
        if temp < 300.0:
            return (
                555.4
                - 2.116905 * temp**1
                + 0.008971429 * temp**2
                - 1.266667e-5 * temp**3
            )
        return (
            423.7411
            - 0.3133575 * temp**1
            + 0.001013916 * temp**2
            - 1.570451e-6 * temp**3
            + 1.06222e-9 * temp**4
            - 2.64198e-13 * temp**5
        )

    def get_thermal_expansion(self, temp: float) -> float:
        # 4-1273 K
        if temp < 100.0:
            return 1.104402e-5 + 4.812192e-8 * temp**1 - 1.223083e-10 * temp**2
        if temp < 210.0:
            return (
                1.276495e-5
                + 1.849516e-8 * temp**1
                + 1.203963e-11 * temp**2
                - 1.023671e-13 * temp**3
            )
        if temp < 800.0:
            return 1.47252e-5 + 8.137386e-9 * temp**1 - 4.58414e-12 * temp**2
        return 1.83456e-5 - 1.577095e-9 * temp**1 + 1.908643e-12 * temp**2

    def get_specific_heat(self, temp: float) -> float:
        # 1-1300 K
        if temp < 17.5:
            return (
                0.00816805501
                + 0.00104457033 * temp**1
                + 0.00344121866 * temp**2
                + 2.84703334e-4 * temp**3
                + 2.24642893e-5 * temp**4
            )
        if temp < 62.0:
            return (
                29.059721
                - 3.76716858 * temp**1
                + 0.154053918 * temp**2
                - 0.00104836396 * temp**3
                + 3.01020641e-7 * temp**4
            )
        if temp < 300.0:
            return (
                -215.281402
                + 8.23639228 * temp**1
                - 0.0473210818 * temp**2
                + 1.29111169e-4 * temp**3
                - 1.35703145e-7 * temp**4
            )
        return (
            342.764033
            + 0.133834821 * temp**1
            + 5.53525209e-5 * temp**2
            - 1.97122089e-7 * temp**3
            + 1.1407471e-10 * temp**4
        )

    def get_elastic_modulus(self, temp: float) -> float:
        return (
            1.396274e11
            - 5077626.0 * temp**1
            - 191131.5 * temp**2
            + 290.7333 * temp**3
            - 0.2058552 * temp**4
            + 5.385261e-5 * temp**5
        ) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        # 4-1300 K
        if temp < 20.0:
            return 0.3413951 - 9.643661e-5 * temp**1 + 2.151898e-6 * temp**2
        return 0.339846 + 2.405498e-5 * temp**1

    def get_density(self, temp: float) -> float:
        if temp < 90.0:
            return (
                9028.155
                + 0.001936185 * temp**1
                - 4.310034e-4 * temp**2
                - 8.227902e-6 * temp**3
            )
        if temp < 250.0:
            return (
                9034.264
                - 0.05885933 * temp**1
                - 0.001406238 * temp**2
                + 1.736657e-6 * temp**3
            )
        if temp < 800.0:
            return 9062.242 - 0.3913962 * temp**1 - 8.947644e-5 * temp**2
        return 9038.962 - 0.3593546 * temp**1 - 9.31574e-5 * temp**2

    def get_hardness(self, temp: float) -> float:
        temp_data = [
            3.880597015,
            15.52238806,
            32.53731343,
            48.35820896,
            62.3880597,
            71.34328358,
            83.58208955,
            91.04477612,
            94.92537313,
            97.91044776,
            103.5820896,
            111.3432836,
            117.0149254,
            125.3731343,
            132.8358209,
            143.5820896,
            152.238806,
            159.4029851,
            171.9402985,
            184.4776119,
            197.0149254,
            213.7313433,
            230.1492537,
            249.2537313,
            269.8507463,
            293.7313433,
        ]
        hardness_data = [
            1.768368618,
            1.762141968,
            1.755915318,
            1.799501868,
            1.867995019,
            1.98007472,
            2.123287671,
            2.322540473,
            2.496886675,
            2.677459527,
            2.503113325,
            2.303860523,
            2.160647572,
            1.98007472,
            1.855541719,
            1.712328767,
            1.587795766,
            1.488169365,
            1.400996264,
            1.307596513,
            1.257783313,
            1.226650062,
            1.201743462,
            1.164383562,
            1.145703611,
            1.139476961,
        ]
        return self.get_prop_from_interp(temp_data, hardness_data, "hardness", temp)


class Cu110(Cu):
    """
    Pure Copper

    The same as Cu, but with a different name.
    """

    __mat_name__ = "cu110"
    __mat_id__ = 10


class CuW30W3(Mat):
    """
    Elkonite copper-tungsten alloy 30W3
    """

    __mat_name__ = "cuw30w3"
    __mat_id__ = 12
    __components__ = {"Cu": 0.2, "W": 0.8}
    def get_thermal_conductivity(self, temp: float) -> float:
        # 293.0	1273.0
        return (
            212.4297
            - 0.2114542 * temp**1
            + 1.963361e-4 * temp**2
            - 6.177219e-8 * temp**3
        )

    def get_thermal_expansion(self, temp: float) -> float:
        return 7.50e-06

    def get_specific_heat(self, temp: float) -> float:
        return 177.5

    def get_elastic_modulus(self, temp: float) -> float:
        return 241

    def get_poisson_ratio(self, temp: float) -> float:
        return 0.3

    def get_density(self, temp: float) -> float:
        return 15670

    def get_hardness(self, temp: float) -> float:
        return 2.736153
