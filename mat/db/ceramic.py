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
