from ..Mat import Mat


class MgAZ31B(Mat):
    """
    Magnesium AZ31B
    """

    __mat_name__ = "mgaz31b"
    __mat_id__ = 30
    __components__ = {
        "Al": 0.003,
        "Zn": 0.01,
        "Mn": 0.006,
        "Si": 0.0004,
        "Cu": 0.00005,
        "Ni": 0.000005,
        "Mg": 0.980545,
    }
    def get_thermal_conductivity(self, temp: float) -> float:
        # 173.0	523.0
        return 31.62642 + 0.1913696 * temp**1 - 1.106685e-4 * temp**2

    def get_thermal_expansion(self, temp: float) -> float:
        # 88.0	811.0
        return (
            1.816435e-5
            + 2.829976e-8 * temp**1
            - 3.073053e-11 * temp**2
            + 1.625552e-14 * temp**3
        )

    def get_specific_heat(self, temp: float) -> float:
        # 425.0	838.0
        return 791.051726 + 0.711512212 * temp**1 + 8.27588087e-5 * temp**2

    def get_elastic_modulus(self, temp: float) -> float:
        # 293.0	723.0
        return (4.759698e10 - 1.042497e7 * temp**1 - 15462.79 * temp**2) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        # 293.0	723.0
        return (
            0.2644563
            + 1.544979e-5 * temp**1
            + 2.189063e-8 * temp**2
            + 4.554008e-11 * temp**3
        )

    def get_density(self, temp: float) -> float:
        # 88.0	811.0
        return (
            1810.206
            - 0.07580797 * temp**1
            - 1.054187e-4 * temp**2
            + 4.024841e-8 * temp**3
        )

    def get_hardness(self, temp: float) -> float:
        return 0.813981
