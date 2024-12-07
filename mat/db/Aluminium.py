from ..Mat import Mat, MatSingleProp


class Al(Mat):
    """
    纯铝

    pure Al
    """

    __mat_name__ = "al"
    __mat_id__ = 2

    def get_thermal_conductivity(self, temp: float) -> float:
        # 0-933 K
        if temp < 14:
            return (
                3895.7 * temp + 203.42 * temp**2 - 56.434 * temp**3 + 2.0664 * temp**4
            )
        if temp < 50:
            return 49148.0 - 2950.9 * temp + 63.175 * temp**2 - 0.46605 * temp**3
        if temp < 82:
            return (
                15117.0
                - 626.0 * temp
                + 10.348 * temp**2
                - 0.078676 * temp**3
                + 2.2917e-4 * temp**4
            )
        if temp < 297.0:
            return (
                913.09
                - 12.076 * temp
                + 0.080875 * temp**2
                - 2.3988e-4 * temp**3
                + 2.6487e-7 * temp**4
            )
        return (
            39.646
            + 1.684 * temp
            - 0.0054134 * temp**2
            + 8.4313e-6 * temp**3
            - 6.537e-9 * temp**4
            + 2.002e-12 * temp**5
        )

    def get_thermal_expansion(self, temp: float) -> float:
        # 20-933 K
        if temp < 220:
            return (
                1.371347e-5
                + 7.808536e-8 * temp
                - 2.568882e-10 * temp**2
                + 3.615726e-13 * temp**3
            )
        if temp < 610:
            return (
                5.760185e-6
                + 1.707141e-7 * temp
                - 6.548135e-10 * temp**2
                + 1.220625e-12 * temp**3
                - 1.064883e-15 * temp**4
                + 3.535918e-19 * temp**5
            )
        return 1.9495e-5 + 9.630182e-9 * temp + 9.462013e-13 * temp**2

    def get_elastic_modulus(self, temp: float) -> float:
        return (
            7.659324e10
            + 2007396.0 * temp**1
            - 186458.4 * temp**2
            + 419.2175 * temp**3
            - 0.3495083 * temp**4
        ) / 1e9

    def get_specific_heat(self, temp: float) -> float:
        # 100-933 K
        if temp < 320:
            return (
                -290.416126
                + 11.1810036 * temp**1
                - 0.0412540099 * temp**2
                + 7.11275398e-5 * temp**3
                - 4.60821994e-8 * temp**4
            )
        return (
            595.658507
            + 1.51302896 * temp**1
            - 0.00207006538 * temp**2
            + 1.30360846e-6 * temp**3
        )

    def get_poisson_ratio(self, temp: float) -> float:
        return (
            0.3238668
            + 3.754548e-6 * temp**1
            + 2.213647e-7 * temp**2
            - 6.565023e-10 * temp**3
            + 4.21277e-13 * temp**4
            + 3.170505e-16 * temp**5
        )

    def get_density(self, temp: float) -> float:
        if temp < 130:
            return (
                2734.317
                - 0.02751647 * temp**1
                + 0.001016054 * temp**2
                - 1.700864e-5 * temp**3
                + 5.734155e-8 * temp**4
            )
        return (
            2736.893
            - 0.006011681 * temp**1
            - 7.012444e-4 * temp**2
            + 1.3582e-6 * temp**3
            - 1.367828e-9 * temp**4
            + 5.177991e-13 * temp**5
        )

    def get_hardness(self, temp: float) -> float:
        return 0.16


class Al3A21(Mat):
    """
    AISI Al 3A21, Similar to Al 3003
    """

    __mat_id__ = 36
    __mat_name__ = "al3a21"

    def get_thermal_conductivity(self, temp: float) -> float:
        # 4-300
        if temp < 60.0:
            return (
                -1.530129
                + 2.972428 * temp**1
                + 0.004582021 * temp**2
                - 3.032544e-4 * temp**3
            )
        return (
            75.77884
            + 1.278889 * temp**1
            - 0.008262703 * temp**2
            + 2.487536e-5 * temp**3
            - 2.814856e-8 * temp**4
        )

    def get_thermal_expansion(self, temp: float) -> float:
        # 4-775
        if temp < 250.0:
            return (
                1.408647e-5
                + 5.850421e-8 * temp**1
                - 1.003649e-10 * temp**2
                + 1.818267e-15 * temp**3
            )
        if temp < 350.0:
            return 1.967679e-5 + 1.206864e-8 * temp**1 - 3.632489e-12 * temp**2
        return (
            1.794192e-5
            + 2.296167e-8 * temp**1
            - 2.52962e-11 * temp**2
            + 1.369558e-14 * temp**3
        )

    def get_specific_heat(self, temp: float) -> float:
        # 4-300
        if temp < 16.0:
            return (
                0.153508701
                - 0.104679998 * temp**1
                + 0.0454540137 * temp**2
                - 0.00328604666 * temp**3
                + 1.21017723e-4 * temp**4
            )
        if temp < 47.0:
            return (
                -7.85853106
                + 1.97771153 * temp**1
                - 0.1627576 * temp**2
                + 0.00632458042 * temp**3
                - 5.22112155e-5 * temp**4
            )
        if temp < 130.0:
            return (
                -74.7637286
                - 1.13922203 * temp**1
                + 0.176562541 * temp**2
                - 0.00147785574 * temp**3
                + 3.9391632e-6 * temp**4
            )
        return (
            -264.783185
            + 9.29321629 * temp**1
            - 0.0121044208 * temp**2
            - 6.74100976e-5 * temp**3
            + 1.6547628e-7 * temp**4
        )

    def get_elastic_modulus(self, temp: float) -> float:
        # 0.0	773.0
        return (
            7.770329e10
            + 2036488.0 * temp**1
            - 189160.7 * temp**2
            + 425.2931 * temp**3
            - 0.3545736 * temp**4
        ) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        # 0.0	773.0
        return (
            0.3238668
            + 3.754548e-6 * temp**1
            + 2.213647e-7 * temp**2
            - 6.565023e-10 * temp**3
            + 4.21277e-13 * temp**4
            + 3.170505e-16 * temp**5
        )

    def get_density(self, temp: float) -> float:
        # 4.0	775.0
        return (
            2764.943
            + 0.005111833 * temp**1
            - 6.260194e-4 * temp**2
            + 8.285936e-7 * temp**3
            - 4.225989e-10 * temp**4
        )

    def get_hardness(self, temp: float) -> float:
        return 0.568806


class Al3A21H112(Al3A21):
    """
    AISI Al 3A21 H112

    Same properties as Al3A21
    """

    __mat_name__ = "al3a21h112"
    __mat_id__ = 3


class Al5A05(Mat):
    """
    AISI Al 5A05, similar to Al 5005
    """

    __mat_id__ = 0
    __mat_name__ = "al5a05"

    def get_thermal_conductivity(self, temp: float) -> float:
        # 4-849
        if temp < 130.0:
            return (
                -1.051689
                + 0.9981587 * temp**1
                - 0.004145406 * temp**2
                + 8.950483e-6 * temp**3
            )
        if temp < 300.0:
            return (
                11.817
                + 0.6909558 * temp**1
                - 0.001587888 * temp**2
                + 1.597797e-6 * temp**3
            )
        return (
            30.01698
            + 0.4662148 * temp**1
            - 6.570887e-4 * temp**2
            + 3.182847e-7 * temp**3
        )

    def get_thermal_expansion(self, temp: float) -> float:
        # 4-575
        if temp < 100.0:
            return (
                1.389086e-5
                + 4.41266e-8 * temp**1
                + 2.944067e-10 * temp**2
                - 2.285788e-12 * temp**3
            )
        if temp < 245.0:
            return (
                1.375093e-5
                + 7.248418e-8 * temp**1
                - 2.348375e-10 * temp**2
                + 3.133913e-13 * temp**3
            )
        return (
            1.640353e-5
            + 3.06124e-8 * temp**1
            - 3.563214e-11 * temp**2
            + 1.742129e-14 * temp**3
        )

    def get_specific_heat(self, temp: float) -> float:
        # 4-300
        if temp < 16.0:
            return (
                0.153508701
                - 0.104679998 * temp**1
                + 0.0454540137 * temp**2
                - 0.00328604666 * temp**3
                + 1.21017723e-4 * temp**4
            )
        if temp < 47.0:
            return (
                -7.85853106
                + 1.97771153 * temp**1
                - 0.1627576 * temp**2
                + 0.00632458042 * temp**3
                - 5.22112155e-5 * temp**4
            )
        if temp < 130.0:
            return (
                -74.8892486
                - 1.13922203 * temp**1
                + 0.176562541 * temp**2
                - 0.00147785574 * temp**3
                + 3.9391632e-6 * temp**4
            )
        return (
            -263.527985
            + 9.29321629 * temp**1
            - 0.0121044208 * temp**2
            - 6.74100976e-5 * temp**3
            + 1.6547628e-7 * temp**4
        )

    def get_elastic_modulus(self, temp: float) -> float:
        # 0.0	773.0
        return (
            7.770329e10
            + 2036488.0 * temp**1
            - 189160.7 * temp**2
            + 425.2931 * temp**3
            - 0.3545736 * temp**4
        ) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        # 0.0	773.0
        #
        return (
            0.3238668
            + 3.754548e-6 * temp**1
            + 2.213647e-7 * temp**2
            - 6.565023e-10 * temp**3
            + 4.21277e-13 * temp**4
            + 3.170505e-16 * temp**5
        )

    def get_density(self, temp: float) -> float:
        # 0-575
        if temp < 100.0:
            return (
                2698.788
                + 2.658375e-4 * temp**1
                - 1.50635e-4 * temp**2
                + 3.814654e-6 * temp**3
                - 9.941487e-8 * temp**4
                + 4.307162e-10 * temp**5
            )
        return (
            2699.543
            + 0.03368339 * temp**1
            - 9.256705e-4 * temp**2
            + 2.004492e-6 * temp**3
            - 2.270152e-9 * temp**4
            + 1.007566e-12 * temp**5
        )

    def get_hardness(self, temp: float) -> float:
        return 0.7


class Al5A05H112(Al5A05):
    """
    AISI Al 5A05 H112
    """

    __mat_id__ = 34
    __mat_name__ = "al5a05h112"


class Al75ST6(Mat):
    """
    AISI Al 75S T6

    Equal to AISI 7075-T6
    """

    __mat_id__ = 33
    __mat_name__ = "al75st6"

    def get_thermal_conductivity(self, temp: float) -> float:
        # 116-700
        if temp < 477.0:
            return (
                7.820747
                + 0.8194394 * temp**1
                - 0.00216484 * temp**2
                + 2.440757e-6 * temp**3
            )
        return -8.842465 + 0.644486 * temp**1 - 5.607477e-4 * temp**2

    def get_thermal_expansion(self, temp: float) -> float:
        # 0-700
        if temp < 20.0:
            return 1.422526e-5 + 5.056226e-8 * temp**1
        return (
            1.390523e-5
            + 7.07427e-8 * temp**1
            - 2.219089e-10 * temp**2
            + 3.272338e-13 * temp**3
            - 1.6939e-16 * temp**4
        )

    def get_specific_heat(self, temp: float) -> float:
        # 116.0	700.0
        return (
            153.496734
            + 4.88875714 * temp**1
            - 0.0128256043 * temp**2
            + 1.62660406e-5 * temp**3
            - 7.07317334e-9 * temp**4
        )

    def get_elastic_modulus(self, temp: float) -> float:
        # 0.0	773.0
        #
        return (
            7.881334e10
            + 2065581.0 * temp**1
            - 191863.0 * temp**2
            + 431.3687 * temp**3
            - 0.3596389 * temp**4
        ) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        # 0.0	773.0
        return (
            0.3238668
            + 3.754548e-6 * temp**1
            + 2.213647e-7 * temp**2
            - 6.565023e-10 * temp**3
            + 4.21277e-13 * temp**4
            + 3.170505e-16 * temp**5
        )

    def get_density(self, temp: float) -> float:
        # 0-700
        if temp < 20.0:
            return 2754.296 - 0.003592712 * temp**1
        return (
            2753.524
            + 0.05647875 * temp**1
            - 0.001127433 * temp**2
            + 2.657999e-6 * temp**3
            - 3.148685e-9 * temp**4
            + 1.417919e-12 * temp**5
        )

    def get_hardness(self, temp: float) -> float:
        return 1.716225


class Al2024T4(Mat):
    """
    AISI Al 2024 T4
    """

    __mat_id__ = 31
    __mat_name__ = "al2024t4"

    def get_thermal_conductivity(self, temp: float) -> float:
        # 4-700
        if temp < 50.0:
            return -0.6765738 + 0.938525 * temp**1 - 0.002896119 * temp**2
        if temp < 120.0:
            return 10.25948 + 0.6163078 * temp**1 - 8.031264e-4 * temp**2
        return (
            -12.17384
            + 1.076003 * temp**1
            - 0.003922898 * temp**2
            + 7.72158e-6 * temp**3
            - 5.396842e-9 * temp**4
        )

    def get_thermal_expansion(self, temp: float) -> float:
        # 0.0	755.0
        return (
            1.36751e-5
            + 6.437183e-8 * temp**1
            - 1.468985e-10 * temp**2
            + 1.595694e-13 * temp**3
            - 5.82147e-17 * temp**4
        )

    def get_specific_heat(self, temp: float) -> float:
        # 116.0	700.0
        return (
            198.819161
            + 3.94185811 * temp**1
            - 0.0073841575 * temp**2
            + 5.2182848e-6 * temp**3
        )

    def get_elastic_modulus(self, temp: float) -> float:
        # 0.0	773.0
        return (
            8.103343e10
            + 2123766.0 * temp**1
            - 197267.6 * temp**2
            + 443.52 * temp**3
            - 0.3697696 * temp**4
        ) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        # 0.0	773.0
        return (
            0.3238668
            + 3.754548e-6 * temp**1
            + 2.213647e-7 * temp**2
            - 6.565023e-10 * temp**3
            + 4.21277e-13 * temp**4
            + 3.170505e-16 * temp**5
        )

    def get_density(self, temp: float) -> float:
        # 0.0	755.0
        return (
            2813.898
            + 0.02810992 * temp**1
            - 7.443022e-4 * temp**2
            + 1.039896e-6 * temp**3
            - 5.689519e-10 * temp**4
        )

    def get_hardness(self, temp: float) -> float:
        return 1.343559


class Al5052(Mat):
    """
    AISI Al 5052
    """

    __mat_id__ = 14
    __mat_name__ = "al5052"

    def get_thermal_conductivity(self, temp: float) -> float:
        # 4-300
        if temp < 60.0:
            return (
                -0.9315578
                + 1.309154 * temp**1
                + 0.001629908 * temp**2
                - 8.388012e-5 * temp**3
            )
        return (
            25.30397
            + 0.7796799 * temp**1
            - 0.002014995 * temp**2
            + 2.275133e-6 * temp**3
        )

    def get_thermal_expansion(self, temp: float) -> float:
        # 200.0	773.0
        return (
            1.525318e-5
            + 3.577384e-8 * temp**1
            - 4.285906e-11 * temp**2
            + 2.203519e-14 * temp**3
        )

    def get_specific_heat(self, temp: float) -> float:
        return Al5A05().get_specific_heat(temp)

    def get_elastic_modulus(self, temp: float) -> float:
        # 0.0	773.0
        return (
            7.770329e10
            + 2036488.0 * temp**1
            - 189160.7 * temp**2
            + 425.2931 * temp**3
            - 0.3545736 * temp**4
        ) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        # 0.0	773.0
        #
        return (
            0.3238668
            + 3.754548e-6 * temp**1
            + 2.213647e-7 * temp**2
            - 6.565023e-10 * temp**3
            + 4.21277e-13 * temp**4
            + 3.170505e-16 * temp**5
        )

    def get_density(self, temp: float) -> float:
        # 200.0	773.0
        return (
            2730.231
            - 0.1048146 * temp**1
            - 1.621319e-4 * temp**2
            + 6.893765e-8 * temp**3
        )

    def get_hardness(self, temp: float) -> float:
        return 0.666876


class Al6061(Mat):
    """
    AISI Aluminum alloy 6061
    """

    __mat_name__ = "al6061"
    __mat_id__ = 13

    def get_thermal_conductivity(self, temp: float) -> float:
        # 4-811 K
        if temp < 130:
            return (
                -1.35437
                + 1.660905 * temp**1
                - 0.008948206 * temp**2
                + 2.250241e-5 * temp**3
            )
        if temp < 300:
            return (
                30.71577
                + 0.8750948 * temp**1
                - 0.002142331 * temp**2
                + 2.05104e-6 * temp**3
            )
        return 79.478 + 0.3355001 * temp**1 - 2.701638e-4 * temp**2

    def get_thermal_expansion(self, temp: float) -> float:
        # 22-588 K
        if temp < 88:
            return (
                1.3648e-5
                + 5.61487e-8 * temp**1
                + 7.71396e-11 * temp**2
                - 9.540526e-13 * temp**3
            )
        return (
            1.316461e-5
            + 8.302515e-8 * temp**1
            - 2.909984e-10 * temp**2
            + 4.99184e-13 * temp**3
            - 3.127291e-16 * temp**4
        )

    def get_specific_heat(self, temp: float) -> float:
        # 30-811 K
        if temp < 30:
            return (
                -1.17115557
                + 0.487569051 * temp**1
                - 0.0443149197 * temp**2
                + 0.00232206728 * temp**3
                - 3.42445087e-6 * temp**4
            )
        if temp < 47:
            return (
                26.582634
                - 1.95690617 * temp**1
                + 0.00352123139 * temp**2
                + 0.00324321212 * temp**3
                - 3.10753881e-5 * temp**4
            )
        if temp < 130:
            return (
                -75.1821286
                - 1.13922203 * temp**1
                + 0.176562541 * temp**2
                - 0.00147785574 * temp**3
                + 3.9391632e-6 * temp**4
            )
        if temp < 145:
            return (
                -264.783185
                + 9.29321629 * temp**1
                - 0.0121044208 * temp**2
                - 6.74100976e-5 * temp**3
                + 1.6547628e-7 * temp**4
            )
        return (
            -5.60596587
            + 6.74472515 * temp**1
            - 0.0153591209 * temp**2
            + 1.61975861e-5 * temp**3
            - 6.21481318e-9 * temp**4
        )

    def get_elastic_modulus(self, temp: float) -> float:
        return (
            7.659324e10
            + 2007396.0 * temp**1
            - 186458.4 * temp**2
            + 419.2175 * temp**3
            - 0.3495083 * temp**4
        ) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        return (
            0.3238668
            + 3.754548e-6 * temp**1
            + 2.213647e-7 * temp**2
            - 6.565023e-10 * temp**3
            + 4.21277e-13 * temp**4
            + 3.170505e-16 * temp**5
        )

    def get_density(self, temp: float) -> float:
        # 88-588K
        if temp < 88:
            return (
                2733.392
                - 0.04347255 * temp**1
                + 0.001455455 * temp**2
                - 2.077034e-5 * temp**3
                + 7.111e-8 * temp**4
            )
        return (
            2736.099
            - 0.009406248 * temp**1
            - 6.040342e-4 * temp**2
            + 8.988964e-7 * temp**3
            - 5.405225e-10 * temp**4
        )

    def get_hardness(self, temp: float) -> float:
        temp_data = [273.15, 448.15, 458.15, 468.15, 493.15, 623.15, 693.15]
        hardness_data = [
            0.9355878,
            0.87301914,
            0.97942509,
            0.972903435,
            0.936715605,
            0.36589917,
            0.36992004,
        ]

        return self.get_prop_from_interp(temp_data, hardness_data, "hardness", temp)


class Al6061H112(Al6061):
    """
    AISI Aluminum Alloy 6061 H112
    """

    __mat_name__ = "al6061h112"
    __mat_id__ = 38


class Al6061T6(Al6061):
    """
    AISI Aluminum 6061-T6

    The same as Al6061 in properties
    """

    __mat_id__ = 9
    __mat_name__ = "al6061t6"


class Al6063(Mat):
    """
    AISI Al 6063
    """

    __mat_id__ = 37
    __mat_name__ = "al6063"

    def get_thermal_conductivity(self, temp: float) -> float:
        # 4-477
        if temp < 60.0:
            return (
                14.86825
                + 3.30989 * temp**1
                + 0.4701955 * temp**2
                - 0.0162375 * temp**3
                + 1.915255e-4 * temp**4
                - 7.883266e-7 * temp**5
            )
        if temp < 150.0:
            return (
                437.2957
                - 4.00863 * temp**1
                + 0.02222813 * temp**2
                - 4.051341e-5 * temp**3
            )
        return (
            183.9345
            + 0.1808882 * temp**1
            - 6.043173e-4 * temp**2
            + 5.676149e-7 * temp**3
        )

    def get_thermal_expansion(self, temp: float) -> float:
        # 223.0	573.0
        return 1.948056e-5 + 1.021437e-8 * temp**1 + 8.175717e-13 * temp**2

    def get_specific_heat(self, temp: float) -> float:
        return Al6061.get_specific_heat(temp)

    def get_elastic_modulus(self, temp: float) -> float:
        # 0.0	773.0
        #
        return (
            7.659324e10
            + 2007396.0 * temp**1
            - 186458.4 * temp**2
            + 419.2175 * temp**3
            - 0.3495083 * temp**4
        ) / 1e9

    def get_poisson_ratio(self, temp: float) -> float:
        # 0.0	773.0
        return (
            0.3238668
            + 3.754548e-6 * temp**1
            + 2.213647e-7 * temp**2
            - 6.565023e-10 * temp**3
            + 4.21277e-13 * temp**4
            + 3.170505e-16 * temp**5
        )

    def get_density(self, temp: float) -> float:
        # 223.0	573.0
        return 2736.641 - 0.1364685 * temp**1 - 7.761706e-5 * temp**2

    def get_hardness(self, temp: float) -> float:
        return 0.813981


class Albshe20(Al6061):
    """
    Aluminum alloy BS He 20

    Similar to Al6061.
    """

    __mat_name__ = "albshe20"
    __mat_id__ = 26


class AlBH137(Mat):
    __mat_id__ = 20
    __mat_name__ = "albh137"
    __temp_data__ = [t + 273.15 for t in [20, 100, 200, 300, 350, 450]]

    def get_thermal_conductivity(self, temp: float) -> float:
        thermal_conductivity_data = [120, 122, 126, 131, 133, 135]
        return self.get_prop_from_interp(
            self.__temp_data__, thermal_conductivity_data, "thermal_conductivity", temp
        )

    def get_thermal_expansion(self, temp: float) -> float:
        thermal_expansion_data = [20.5, 21.5, 22, 22.5, 23, 24] * 1e-6
        return self.get_prop_from_interp(
            self.__temp_data__, thermal_expansion_data, "thermal_expansion", temp
        )

    def get_specific_heat(self, temp: float) -> float:
        return 902

    def get_elastic_modulus(self, temp: float) -> float:
        elastic_modulus_data = [82.5, 80, 76, 71.5, 68.5, 63.5]
        return self.get_prop_from_interp(
            self.__temp_data__, elastic_modulus_data, "elastic_modulus", temp
        )

    def get_poisson_ratio(self, temp: float) -> float:
        return 0.32

    def get_density(self, temp: float) -> float:
        density_data = [2770, 2760, 2750, 2730, 2720, 2700]
        return self.get_prop_from_interp(
            self.__temp_data__, density_data, "density", temp
        )

    def get_hardness(self, temp: float) -> float:
        return 0.666879
