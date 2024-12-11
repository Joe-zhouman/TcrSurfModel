from ..mat import Mat


class AlN(Mat):
    """
    AlN
    """
    __mat_name__ = "aln"
    __mat_id__ = 8
    __components__ = {"Al": 0.6583, "N": 0.3417}
    def get_thermal_conductivity(self, temp: float) -> float:
        # 15-1800
        if temp < 60.0:
            return (
                -1039.309
                + 182.7191 * temp**1
                - 3.229445 * temp**2
                + 0.01807475 * temp**3
            )
        if temp < 300.0:
            return (
                2406.915
                + 5.150704 * temp**1
                - 0.1846174 * temp**2
                + 7.596644e-4 * temp**3
                - 9.337017e-7 * temp**4
            )
        if temp < 600.0:
            return 960.0 - 3.33 * temp**1 + 0.00405 * temp**2 - 1.5e-6 * temp**3
        return (
            272.0825
            - 0.4303125 * temp**1
            + 2.609375e-4 * temp**2
            - 5.46875e-8 * temp**3
        )

    def get_thermal_expansion(self, temp: float) -> float:
        # 60-1269
        if temp < 250.0:
            return (
                1.160353e-6
                + 2.25342e-9 * temp**1
                + 1.745694e-11 * temp**2
                + 1.299743e-14 * temp**3
            )
        return (
            1.736106e-6
            + 6.193034e-9 * temp**1
            - 4.709742e-12 * temp**2
            + 1.658835e-15 * temp**3
        )

    def get_specific_heat(self, temp: float) -> float:
        # 5-2700
        if temp < 82.0:
            return (
                -1.960527
                + 0.51489601 * temp**1
                - 0.0269594492 * temp**2
                + 6.82541778e-4 * temp**3
                - 3.27756945e-6 * temp**4
            )
        if temp < 215.0:
            return (
                -41.9948498
                + 0.142483693 * temp**1
                + 0.0212085747 * temp**2
                - 4.52792062e-5 * temp**3
            )
        if temp < 700.0:
            return (
                170.209681
                - 2.01770471 * temp**1
                + 0.0318480222 * temp**2
                - 8.95685198e-5 * temp**3
                + 1.03194302e-7 * temp**4
                - 4.35243947e-11 * temp**5
            )
        return (
            654.446636
            + 1.1318335 * temp**1
            - 9.27345526e-4 * temp**2
            + 3.55076624e-7 * temp**3
            - 4.72263979e-11 * temp**4
        )

    def get_elastic_modulus(self, temp: float) -> float:
        # 0-1200
        if temp < 130.0:
            return (3.102093e11 - 316827.8 * temp**1) / 1e9
        if temp < 410.0:
            return (3.101374e11 + 3423625.0 * temp**1 - 24536.23 * temp**2) / 1e9
        return (3.139299e11 - 1.418573e7 * temp**1 - 4093.343 * temp**2) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        0.25

    def get_density(self, temp: float) -> float:
        # 70-1269
        if temp < 200.0:
            return (
                3243.143
                - 0.003943618 * temp**1
                + 2.743787e-5 * temp**2
                - 1.681385e-7 * temp**3
            )
        return (
            3243.892
            + 0.006269394 * temp**1
            - 8.252462e-5 * temp**2
            + 5.575774e-8 * temp**3
            - 1.676819e-11 * temp**4
        )

    def get_hardness(self, temp: float) -> float:
        11.287


class Cc(Mat):
    __mat_name__ = "cc"
    __mat_id__ = 40
    __components__ = {"C": 1}

    def get_thermal_conductivity(self, temp: float) -> float:
        temp_data = [
            472.0450276,
            541.1058011,
            599.1168508,
            690.2770718,
            786.9621547,
            900.2218232,
            1052.155525,
            1217.901381,
            1422.321271,
            1670.940055,
            1938.895856,
            2148.840608,
            2369.835083,
            2579.779834,
            2684.75221,
            2753.812983,
        ]
        thermal_conductivity_data = [
            110.2209945,
            94.94475138,
            82.76243094,
            70,
            59.8480663,
            51.33977901,
            43.60497238,
            37.03038674,
            31.13259669,
            26.20165746,
            23.97790055,
            23.49447514,
            23.49447514,
            22.0441989,
            21.4640884,
            22.23756906,
        ]
        return self.get_prop_from_interp(
            temp_data, thermal_conductivity_data, "thermal_conductivity", temp
        )

    def get_thermal_expansion(self, temp: float) -> float:
        temp_data = [
            374.4863029,
            471.3682628,
            573.8181514,
            669.5865256,
            767.5820713,
            881.1678174,
            968.0275056,
            1064.909465,
            1167.359354,
            1265.3549,
        ]
        thermal_expansion_data = [
            -3.15249e-07,
            -2.44868e-07,
            -2.03226e-07,
            -1.12317e-07,
            -2.37537e-08,
            3.54839e-08,
            8.82698e-08,
            1.34604e-07,
            1.76833e-07,
            2.21994e-07,
        ]
        return self.get_prop_from_interp(
            temp_data, thermal_expansion_data, "expansion", temp
        )

    def get_specific_heat(self, temp: float) -> float:
        temp_data = [
            480.8177316,
            802.9689563,
            1138.432215,
            1391.360863,
            1665.588765,
            2030.338498,
            2328.528062,
            2512.234132,
            2653.341693,
            2757.175559,
            2861.009425,
        ]
        specific_heat_data = [
            1425.323741,
            1524.460432,
            1672.230216,
            1803.165468,
            1924.748201,
            2072.517986,
            2225.899281,
            2326.906475,
            2444.748201,
            2491.510791,
            2528.920863,
        ]
        return self.get_prop_from_interp(
            temp_data, specific_heat_data, "specific_heat", temp
        )

    def get_elastic_modulus(self, temp: float) -> float:
        temp_data = [
            308.4072498,
            784.927362,
            1084.646726,
            1285.020907,
            1483.711272,
            1680.71782,
            1877.724369,
            1980.437184,
        ]
        elastic_modulus_data = [
            49.22882427,
            53.07206068,
            61.8710493,
            65.91656131,
            66.52338812,
            60.7585335,
            47.71175727,
            41.94690265,
        ]
        return self.get_prop_from_interp(
            temp_data, elastic_modulus_data, "elastic_modulus", temp
        )

    def get_poisson_ratio(self, temp: float) -> float:
        return 0.1

    def get_density(self, temp: float) -> float:
        return (
            self.get_thermal_conductivity(temp)
            / self.get_specific_heat(temp)
            / self.get_thermal_diffusion(temp)
        )

    def get_hardness(self, temp: float) -> float:
        return 7.848

    def get_thermal_diffusion(self, temp: float) -> float:
        temp_data = [
            409.0923275,
            495.3722222,
            652.7796296,
            847.2240741,
            1044.754938,
            1300.927778,
            1572.532716,
            1773.15,
            2041.668519,
            2202.162346,
            2375.001852,
            2557.100617,
            2683.643827,
            2834.878395,
        ]
        thermal_diffusion_data = [
            4.51519e-05,
            3.96537e-05,
            3.10215e-05,
            2.41032e-05,
            1.96157e-05,
            1.61253e-05,
            1.4349e-05,
            1.33206e-05,
            1.21988e-05,
            1.15132e-05,
            1.09761e-05,
            1.02355e-05,
            1.00173e-05,
            1.06094e-05,
        ]

        return self.get_prop_from_interp(
            temp_data, thermal_diffusion_data, "thermal_diffusion", temp
        )
