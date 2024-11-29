from ..Mat import Mat, MatSingleProp


class Ti(MatSingleProp):
    """
    pure Ti
    """

    __mat_id__ = 6
    __mat_name__ = "ti"

    __prop__ = (
        21.9,
        9.68e-06,
        116,
        0.32,
        1402.58,
        4502,
        1.6,
    )


class TiAl(Mat):
    """
    gamma-TiAl
    """

    __mat_id__ = 21
    __mat_name__ = "tial"

    def get_thermal_conductivity(self, temp: float) -> float:
        return 6.483477 + 0.01826249 * temp**1 - 4.533913e-6 * temp**2

    def get_thermal_expansion(self, temp: float) -> float:
        return 9.54688e-6 + 4.884527e-9 * temp**1

    def get_specific_heat(self, temp: float) -> float:
        return (
            -379.854105
            + 4.97534084 * temp**1
            - 0.00947159694 * temp**2
            + 9.47436675e-6 * temp**3
            - 4.72033859e-9 * temp**4
            + 9.30436665e-13 * temp**5
        )

    def get_elastic_modulus(self, temp: float) -> float:
        return 1.829266e2 - 3.42e-2 * temp**1

    def get_poisson_ratio(self, temp: float) -> float:
        return 0.2315375 + 5.008234e-6 * temp**1

    def get_density(self, temp: float) -> float:
        return 3661.056 - 0.09177033 * temp**1 - 4.830411e-5 * temp**2

    def get_hardness(self, temp: float) -> float:
        temp_data = [
            296.1866492,
            669.3803665,
            770.741623,
            872.1028796,
            972.3123037,
            1072.521728,
            1183.097644,
            1270.636911,
        ]
        hardness_data = [
            2.85995086,
            2.55036855,
            2.432432432,
            2.395577396,
            2.167076167,
            1.695331695,
            1.275184275,
            0.928746929,
        ]
        return self.get_prop_from_interp(temp_data, hardness_data, "hardness", temp)


class Ti6Al4V(Mat):
    """
    Ti6Al4V alloy
    """

    __mat_id__ = 39
    __mat_name__ = "ti6al4v"

    def get_thermal_conductivity(self, temp: float) -> float:
        if temp < 9.0:
            return -0.08721429 + 0.1026071 * temp**1 - 1.785714e-4 * temp**2
        if temp < 311.0:
            return (
                0.1560505
                + 0.07648919 * temp**1
                - 2.883179e-4 * temp**2
                + 3.68138e-7 * temp**3
            )
        return (
            8.114005
            - 0.01485211 * temp**1
            + 4.468662e-5 * temp**2
            - 2.273481e-8 * temp**3
        )

    def get_thermal_expansion(self, temp: float) -> float:
        if temp < 24.0:
            return -0.00173578
        if temp < 293.0:
            return (
                -0.0017112276
                - 2.134531e-6 * temp**1
                + 4.801472e-8 * temp**2
                - 7.088608e-11 * temp**3
                - 3.146896e-16 * temp**4
            )
        return -0.002539514 + 8.201155e-6 * temp**1 + 1.590887e-9 * temp**2

    def get_specific_heat(self, temp: float) -> float:
        if temp < 95.0:
            return (
                15.6349846
                - 2.33167626 * temp**1
                + 0.10906734 * temp**2
                - 5.76425078e-4 * temp**3
            )
        if temp < 303.0:
            return (
                -167.173226
                + 6.75429814 * temp**1
                - 0.0235237743 * temp**2
                + 2.95625708e-5 * temp**3
            )
        return (
            383.351385
            + 0.670881806 * temp**1
            - 5.35234016e-4 * temp**2
            + 1.63517247e-7 * temp**3
        )

    def get_elastic_modulus(self, temp: float) -> float:
        if temp < 300.0:
            return 1.196706e2 - 4.940015e-2 * temp**1
        return 1.197517e2 - 4.967055e-2 * temp**1

    def get_poisson_ratio(self, temp: float) -> float:
        return 0.3216426 + 6.33075e-5 * temp**1

    def get_density(self, temp: float) -> float:
        if temp < 24.0:
            return 4453.153
        if temp < 300.0:
            return (
                4452.817
                + 0.02869485 * temp**1
                - 6.448869e-4 * temp**2
                + 9.646377e-7 * temp**3
                - 1.720215e-11 * temp**4
            )
        return 4467.094 - 0.119171 * temp**1 - 1.275079e-5 * temp**2

    def get_hardness(self, temp: float) -> float:
        temp_data = [
            1.030927835,
            100,
            196.9072165,
            297.9381443,
            398.9690722,
            498.9690722,
            598.9690722,
            698.9690722,
            800,
            898.9690722,
            998.9690722,
        ]
        hardness_data = [
            3.855923077,
            3.739076923,
            3.690076923,
            3.697615385,
            3.740961538,
            3.676884615,
            3.663692308,
            3.614692308,
            3.580769231,
            3.661807692,
            3.999153846,
        ]
        return self.get_prop_from_interp(temp_data, hardness_data, "hardness", temp)
