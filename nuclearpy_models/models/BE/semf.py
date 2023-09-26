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
