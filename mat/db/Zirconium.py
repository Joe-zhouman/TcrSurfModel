from ..Mat import Mat


class Zr4(Mat):
    """
    Zr4 alloy
    """

    __mat_name__ = "zr4"
    __mat_id__ = 18

    def get_thermal_conductivity(self, temp: float) -> float:
        return 12.767 - 5.4348e-4 * temp**1 + 8.9818e-6 * temp**2

    def get_thermal_expansion(self, temp: float) -> float:
        return 0 - 0.002077956 + 0.000007092 * temp**1

    def get_specific_heat(self, temp: float) -> float:
        return 261.444855 + 0.0868976215 * temp**1

    def get_elastic_modulus(self, temp: float) -> float:
        return 1.168844e2 - 6.417e-2 * temp**1

    def get_poisson_ratio(self, temp: float) -> float:
        return 0.2869477 + 8.72093e-5 * temp**1

    def get_density(self, temp: float) -> float:
        return 6673.019 - 0.1474608 * temp**1

    def get_hardness(self, temp: float) -> float:
        return 390.56 - 0.92 * temp + 8.39e-4 * temp**2 - 2.88e-7 * temp**3


class ZrNb(Mat):
    """
    Zr 2.5% Nb
    """

    __mat_id__ = 17
    __mat_name__ = "zrnb"

    def get_thermal_conductivity(self, temp: float) -> float:
        if temp < 1130.0:
            return 16.85 - 0.002186 * temp**1 + 8.899e-6 * temp**2
        return 5.0 + 0.02 * temp**1

    def get_thermal_expansion(self, temp: float) -> float:
        if temp < 1130.0:
            return 16.85 - 0.002186 * temp**1 + 8.899e-6 * temp**2
        return 5.0 + 0.02 * temp**1

    def get_specific_heat(self, temp: float) -> float:
        return (
            258.798851
            + 0.0528641287 * temp**1
            + 9.04576198e-5 * temp**2
            - 7.31582023e-8 * temp**3
            + 1.70974223e-11 * temp**4
        )

    def get_elastic_modulus(self, temp: float) -> float:
        return 1.1368e2 - 6.0e-2 * temp**1

    def get_poisson_ratio(self, temp: float) -> float:
        return 0.3407154 - 9.585938e-7 * temp**1 - 3.334472e-9 * temp**2

    def get_density(self, temp: float) -> float:
        return 6601 / (1 + self.get_thermal_expansion(temp))

    def get_hardness(self, temp: float) -> float:
        return 400.91 - 0.78 * temp + 5.81e-4 * temp**2 - 1.76e-7 * temp**3
