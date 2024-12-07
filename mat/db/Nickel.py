from ..Mat import Mat


class Ni200(Mat):
    """
    Nickel 200
    """

    __mat_name__ = "ni200"
    __mat_id__ = 16
    __components__ = {
        "Ni": 0.99,
        "C": 0.00075,
        "Cu": 0.00125,
        "Fe": 0.002,
        "Mn": 0.00175,
        "S": 0.00005,
        "Si": 0.0175,
    }
    def get_thermal_conductivity(self, temp: float) -> float:
        # 173-1273 K
        if temp < 673.0:
            return (
                76.12158
                + 0.02717507 * temp**1
                - 2.126458e-4 * temp**2
                + 1.876168e-7 * temp**3
            )
        return 40.623 + 0.02201643 * temp**1 - 3.571429e-7 * temp**2

    def get_thermal_expansion(self, temp: float) -> float:
        # 73-1373 K
        return (
            8.852306e-6
            + 1.80633e-8 * temp**1
            - 2.211481e-11 * temp**2
            + 1.505531e-14 * temp**3
            - 3.884635e-18 * temp**4
        )

    def get_specific_heat(self, temp: float) -> float:
        # 293 - 1726 K
        if temp < 633.0:
            return 292.88 + 0.50208 * temp**1
        return 418.4 + 0.1284488 * temp**1

    def get_elastic_modulus(self, temp: float) -> float:
        # 0-1323 K
        if temp < 773.0:
            return 2.198604e2 - 4.976173e-2 * temp**1 - 6940.452e-9 * temp**2
        return (
            3.675636e11 - 5.264939e8 * temp**1 + 511021.8 * temp**2 - 191.9755 * temp**3
        ) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        # 293-773 K
        return 0.2970106 - 4.70622e-5 * temp**1 + 3.902138e-8 * temp**2

    def get_density(self, temp: float) -> float:
        return (
            8964.214
            - 0.1681755 * temp**1
            - 3.536041e-4 * temp**2
            + 2.01714e-7 * temp**3
            - 4.919056e-11 * temp**4
        )

    def get_hardness(self, temp: float) -> float:
        temp_data = [
            281.8456522,
            351.4108696,
            422.7152174,
            495.7586957,
            575.7586957,
            633.15,
            709.6717391,
            774.0195652,
            838.3673913,
            899.2369565,
            953.15,
            998.3673913,
            1041.845652,
            1087.063043,
        ]
        hardness_data = [
            2.34402812,
            2.326792619,
            2.292321617,
            2.257850615,
            2.102731107,
            1.878669596,
            1.654608084,
            1.465017575,
            1.292662566,
            1.103072056,
            0.930717047,
            0.77559754,
            0.706655536,
            0.672184534,
        ]
        return self.get_prop_from_interp(temp_data, hardness_data, "hardness", temp)


class In600(Mat):
    """
    Inconel 600 Nickel Alloy
    """

    __mat_id__ = 41
    __mat_name__ = "in600"
    __components__ = {
        "C": 0.0005,
        "Ni": 0.72,
        "Cr": 0.155,
        "Fe": 0.08,
        "Mg": 0.005,
        "S": 0.000075,
        "Cu": 0.0025,
    }
    def get_thermal_conductivity(self, temp: float) -> float:
        return 8.95499 + 0.01654954 * temp**1

    def get_thermal_expansion(self, temp: float) -> float:
        return (
            1.156732e-5
            + 2.324533e-9 * temp**1
            + 2.517045e-12 * temp**2
            - 7.894455e-16 * temp**3
        )

    def get_specific_heat(self, temp: float) -> float:
        if temp < 473.0:
            return (
                193.0052
                + 1.03727385 * temp**1
                + 5.98189827e-4 * temp**2
                - 6.7018015e-6 * temp**3
                + 7.61250767e-9 * temp**4
            )
        return (
            -139.761541
            + 3.94221333 * temp**1
            - 0.009847567 * temp**2
            + 1.20913374e-5 * temp**3
            - 6.94019745e-9 * temp**4
            + 1.49384657e-12 * temp**5
        )

    def get_elastic_modulus(self, temp: float) -> float:
        if temp < 75.0:
            return 2.264875e2 - 8716981.0e-9 * temp**1
        if temp < 300.0:
            return 2.28581e2 - 3.226934e-2 * temp**1 - 58773.58e-9 * temp**2
        return 2.212243e2 - 1.402353e-2 * temp**1 - 37225.36e-9 * temp**2

    def get_poisson_ratio(self, temp: float) -> float:
        if temp < 75.0:
            return 0.3181427 + 1.322061e-6 * temp**1
        if temp < 300.0:
            return 0.3172927 + 8.914209e-6 * temp**1 + 4.510509e-8 * temp**2
        return (
            0.356154
            - 1.411953e-4 * temp**1
            + 1.395181e-7 * temp**2
            - 8.561041e-11 * temp**3
            + 3.915795e-14 * temp**4
        )

    def get_density(self, temp: float) -> float:
        return (
            8426.01
            - 0.2674595 * temp**1
            - 8.229108e-5 * temp**2
            - 7.341464e-9 * temp**3
        )

    def get_hardness(self, temp: float) -> float:
        temp_data = [
            1223.15,
            1273.15,
            1323.15,
            1373.15,
            1423.15,
            1473.15,
            300.15,
        ]
        hardness_data = [
            1.735839,
            1.76526,
            1.784874,
            1.637769,
            1.363173,
            1.235682,
            2.000628,
        ]
        return self.get_prop_from_interp(temp_data, hardness_data, "hardness", temp)
