from ..Mat import Mat


class Si(Mat):
    """
    pure Si
    """

    __mat_id__ = 3
    __mat_name__ = "si"
    __prop__ = (
        149,
        2.56e-06,
        130,
        0.27,
        706,
        2329,
        13,
    )

    def get_thermal_conductivity(self, temp: float) -> float:
        # 0-1685
        if temp < 8.0:
            return (
                -0.2498417 * temp**1
                + 1.481083 * temp**2
                + 5.733086 * temp**3
                - 0.3609236 * temp**4
            )
        if temp < 35.0:
            return (
                2955.562
                - 1163.24 * temp**1
                + 195.2953 * temp**2
                - 10.93608 * temp**3
                + 0.2648092 * temp**4
                - 0.002396056 * temp**5
            )
        if temp < 100.0:
            return (
                14141.66
                - 506.8115 * temp**1
                + 8.440371 * temp**2
                - 0.06860715 * temp**3
                + 2.162919e-4 * temp**4
            )
        if temp < 273.0:
            return (
                5061.885
                - 79.31105 * temp**1
                + 0.5061928 * temp**2
                - 0.001469493 * temp**3
                + 1.607847e-6 * temp**4
            )
        if temp < 1000.0:
            return (
                810.9131
                - 4.713958 * temp**1
                + 0.01248387 * temp**2
                - 1.698128e-5 * temp**3
                + 1.152568e-8 * temp**4
                - 3.094028e-12 * temp**5
            )
        return (
            393.8803
            - 0.980598 * temp**1
            + 9.947652e-4 * temp**2
            - 4.55816e-7 * temp**3
            + 7.898531e-11 * temp**4
        )

    def get_thermal_expansion(self, temp: float) -> float:
        # 0-1000
        if temp < 30.0:
            return 7.35637e-7 + 2.453566e-9 * temp**1 + 1.20482e-11 * temp**2
        if temp < 130.0:
            return (
                7.713685e-7
                + 2.098318e-10 * temp**1
                + 4.628581e-11 * temp**2
                + 7.569451e-14 * temp**3
                - 8.713366e-16 * temp**4
            )
        if temp < 293.0:
            return (
                -3.223163e-7
                + 2.257142e-8 * temp**1
                - 9.684044e-11 * temp**2
                + 2.835316e-13 * temp**3
                - 3.440569e-16 * temp**4
            )
        return (
            6.772622e-7
            + 9.501405e-9 * temp**1
            - 1.271286e-11 * temp**2
            + 8.584038e-15 * temp**3
            - 2.241706e-18 * temp**4
        )

    def get_specific_heat(self, temp: float) -> float:
        # 1-1685
        if temp < 7.0:
            return (
                -4.8321811e-5
                + 7.68448084e-5 * temp**1
                - 3.41813386e-5 * temp**2
                + 2.80830708e-4 * temp**3
                - 3.12897302e-7 * temp**4
            )
        if temp < 20.0:
            return (
                0.0525075264
                - 0.0396481488 * temp**1
                + 0.0100460936 * temp**2
                - 7.81251542e-4 * temp**3
                + 3.9615568e-5 * temp**4
            )
        if temp < 50.0:
            return (
                -1.80567549
                + 0.761903471 * temp**1
                - 0.0865373791 * temp**2
                + 0.0037353614 * temp**3
                - 3.33397563e-5 * temp**4
            )
        if temp < 293.0:
            return (
                -82.9482602
                + 2.71223532 * temp**1
                + 0.0140475122 * temp**2
                - 7.97769138e-5 * temp**3
                + 1.07990546e-7 * temp**4
            )
        if temp < 900.0:
            return (
                63.0442191
                + 3.7706731 * temp**1
                - 0.00694853616 * temp**2
                + 5.9532044e-6 * temp**3
                - 1.91438418e-9 * temp**4
            )
        return 769.459775 + 0.187175131 * temp**1 - 3.18395957e-5 * temp**2

    def get_elastic_modulus(self, temp: float) -> float:
        # 73.0	923.0
        return (
            1.640074e11 + 216812.4 * temp**1 - 16643.99 * temp**2 + 6.096174 * temp**3
        ) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        # 73.0	923.0
        #
        return 0.2238761 - 6.012478e-6 * temp**1

    def get_density(self, temp: float) -> float:
        # 0-1000
        if temp < 30.0:
            return 2331.507 - 7.113612e-5 * temp**1 + 3.674386e-6 * temp**2
        if temp < 130.0:
            return (
                2331.592
                - 0.005873649 * temp**1
                + 1.206114e-4 * temp**2
                - 5.479876e-7 * temp**3
                + 1.606517e-10 * temp**4
            )
        if temp < 293.0:
            return (
                2330.436
                + 0.02130626 * temp**1
                - 9.544145e-5 * temp**2
                + 4.607415e-8 * temp**3
                + 4.840886e-11 * temp**4
            )
        return (
            2332.565
            + 0.003839515 * temp**1
            - 5.433308e-5 * temp**2
            + 4.287211e-8 * temp**3
            - 1.366545e-11 * temp**4
        )

    def get_hardness(self, temp: float) -> float:
        return 13
