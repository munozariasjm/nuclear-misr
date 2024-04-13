import numpy as np

# Masses of the particles in MeV
M_e = 0.511
M_n = 939.565
M_p = 938.272
# Parameters of the semi-empirical mass formula (SEMF) in MeV
a1 = 15.8
a2 = 18.3
a3 = 0.714
a4 = 23.2


def semf_be(Z, N):
    A = Z + N
    if N % 2 == 0 and Z % 2 == 0:
        delta = 12 / (A**0.5)
    elif N % 2 == 1 and Z % 2 == 1:
        delta = -12 / (A**0.5)
    else:
        delta = 0
    B = (
        a1 * A
        - a2 * A ** (2 / 3)
        - a3 * Z * (Z - 1) / A ** (1 / 3)
        - a4 * (N - Z) ** 2 / A
    ) + delta
    return B


def seeger_be(Z, N):
    A = Z + N
    # Given constants
    W_0 = 15.645  # MeV
    gamma = 19.655  # MeV
    beta = 30.586  # MeV
    eta = 53.767  # MeV
    delta_coeff = 10.609  # MeV
    r_0 = 1.2025  # fm
    N = A - Z  # Number of neutrons

    # Calculating the terms of the Seeger model
    volume_term = W_0 * A
    surface_term = -gamma * A ** (2 / 3)
    coulomb_term = (
        -(0.86 / r_0)
        * (Z**2)
        * (A ** (-1 / 3))
        * (1 - 0.76361 * (Z ** (-1 / 3)) - 2.543 * (r_0 ** (-2)) * (A ** (-2 / 3)))
    )
    symmetry_term = -(eta * (A ** (-4 / 3)) - (beta * A ** (-1))) * ((N - Z) ** 2)
    pairing_term = (
        delta_coeff
        / 2
        * (A ** (-1 / 2))
        * (0 if (Z % 2) != (N % 2) else (1 if Z % 2 == 0 else -1))
    )
    asymmetry_term = 7 * np.exp(-6 * abs(N - Z) / A)
    additional_term = 14.33 * 10 ** (-6) * Z ** (2.39)

    # Summing all terms to get the binding energy
    B_A = (
        volume_term
        + surface_term
        + coulomb_term
        + symmetry_term
        + pairing_term
        + asymmetry_term
        + additional_term
    )

    return B_A
