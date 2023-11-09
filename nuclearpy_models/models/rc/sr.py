from nuclearpy_models.utils.physics import PhysicsQualities
from nuclearpy_models.models.rc.mnp import mnp_rc

pq = PhysicsQualities()


def sr_rc(Z, N):
    I = pq.I(N, Z)
    A = N + Z
    P = pq.compute_P(Z, N)
    d = pq.compute_d(Z, N)
    K = (Z + N) ** (1 / 3)
    u = pq.protons_in_shell(Z)
    v = pq.neutrons_in_shell(N)
    m = pq.protons_for_shell(Z)
    n = pq.neutrons_for_shell(N)
    x = u - v
    sr = mnp_rc(Z, N)
    sr += (
        (
            (((-2.4128800890460504 * -0.33651388812597666) ** I) / 0.8349193432167763)
            + (
                (1.1823759708727454e-7 * (v - m))
                * (
                    (
                        n
                        * (
                            (u - (-(0.9897962173011227**m)))
                            - (v * 0.7420746696139493)
                        )
                    )
                    - Z
                )
            )
        )
        * (m + ((u - (u**d)) * P))
    ) + -0.003402575685507848
    return sr
