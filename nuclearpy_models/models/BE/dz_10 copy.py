from __future__ import division
import numpy as np
from math import sqrt


class DuffloZuker10:
    def __init__(self) -> None:
        self.coeffs = (
            np.array(
                [
                    0.7043,
                    17.7418,
                    16.2562,
                    37.5562,
                    53.9017,
                    0.4711,
                    2.1307,
                    0.0210,
                    40.5356,
                    6.0632,
                ],
                order="F",
            ),
        )

    def __call__(self, Z, N):
        return self.binfing_energy(Z, N)

    def binfing_energy(self, Z, N):
        mass_excess = self.mass_excess(Z, N)
        E = Z * 7.28903 + N * 8.07138 - mass_excess
        return E  # MeV

    def init_vars(self):
        term = np.zeros(10, order="F")
        onp = np.zeros((9, 2, 2), order="F")
        noc = np.zeros((18, 2), order="F")
        op, n2, dx, qx, os, oei, dei, pp, y = [np.zeros(2) for _ in range(9)]
        return term, onp, noc, op, n2, dx, qx, os, oei, dei, pp, y

    def get_internal_features(self, Z: int, N: int):
        nuclei = [N, Z]
        a = sum(nuclei)
        t = abs(N - Z)
        r = a ** (1 / 3)
        rc = r * (1 - 0.25 * (t / a) ** 2)  # Charge radius
        ra = rc**2 / r
        z2 = Z * (Z - 1)
        term_0 = (-z2 + 0.76 * z2 ** (2 / 3)) / rc  # Coulomb energy
        return nuclei, a, t, r, rc, ra, z2, term_0

    def binfing_energy(
        self,
        Z: int,
        N: int,
    ):
        term, onp, noc, op, n2, dx, qx, os, oei, dei, pp, y = self.init_vars()
        nuclei, a, t, r, rc, ra, z2, term_0 = self.get_internal_features(Z, N)
        term[0] = term_0
        for deformed in [0, 1]:  # deformed=0  spherical, deformed=1  deformed
            ju = 4 if deformed else 0  # nucleons associated to deform.
            map(lambda x: x.fill(0), [term[1:], noc, onp, os, op])  # init with 0
            for I3 in [0, 1]:
                n2[I3] = 2 * (nuclei[I3] // 2)  # (for pairing calculation)
                ncum = i = 0
                while True:
                    i += 1  # sub-shells (ssh) j and r filling
                    idd = (i + 1) if i % 2 else (i * (i - 2) // 4)
                    ncum += idd
                    if ncum < nuclei[I3]:
                        noc[i - 1, I3] = idd  # nb of nucleons in each ssh
                    else:
                        break
                i_max = i + 1  # i_max = last subshell nb
                ip = (i - 1) // 2  # HO number (p)
                ipm = i // 2
                pp[I3] = ip
                moc = nuclei[I3] - ncum + idd
                noc[i - 1, I3] = moc - ju  # nb of nucleons in last ssh
                noc[i, I3] = ju
                if i % 2:  # 'i' is odd, ssh j
                    oei[I3] = moc + ip * (ip - 1)  # nb of nucleons in last EI shell
                    dei[I3] = ip * (ip + 1) + 2  # size of the EI shell
                else:  # ssh r
                    oei[I3] = moc - ju  # nb of nucleons in last EI shell
                    dei[I3] = (ip + 1) * (ip + 2) + 2  # size of the EI shell
                qx[I3] = (
                    oei[I3] * (dei[I3] - oei[I3] - ju) / dei[I3]
                )  # n*(D-n)/D        S3(j)
                dx[I3] = qx[I3] * (2 * oei[I3] - dei[I3])  # n*(D-n)*(2n-D)/D  Q
                if deformed:
                    qx[I3] /= sqrt(dei[I3])  # scaling for deformed
                for i in range(1, i_max + 1):  # Amplitudes
                    ip = (i - 1) // 2
                    fact = sqrt((ip + 1) * (ip + 2))
                    onp[ip, 0, I3] += noc[i - 1, I3] / fact  # for FM term
                    vm = -1.0
                    if i % 2:
                        vm = 0.5 * ip  # for spin-orbit term
                    onp[ip, 1, I3] += noc[i - 1, I3] * vm
                for ip in range(0, ipm + 1):  # FM and SO terms
                    den = ((ip + 1) * (ip + 2)) ** (3.0 / 2)
                    op[I3] = op[I3] + onp[ip, 0, I3]  # FM
                    os[I3] = (
                        os[I3]
                        + onp[ip, 1, I3] * (1 + onp[ip, 0, I3]) * (ip * ip / den)
                        + onp[ip, 1, I3] * (1 - onp[ip, 0, I3]) * ((4 * ip - 5) / den)
                    )  # SO
                op[I3] = op[I3] * op[I3]

            term[1] = sum(op)  # Master term (FM): volume
            term[2] = -term[1] / ra  # surface
            term[1] += sum(os)  # FM + SO
            term[3] = -t * (t + 2) / r**2
            term[4] = -term[3] / ra  # surface
            if not deformed:  # spherical
                term[5] = sum(dx)  # S3  volume
                term[6] = -term[5] / ra  # surface
                px = sqrt(pp[0]) + sqrt(pp[1])
                term[7] = qx[0] * qx[1] * (2**px)  # QQ sph.
            else:  # deformed
                term[8] = qx[0] * qx[1]  # QQ deform.
            term[4] = t * (1 - t) / (a * ra**3) + term[4]  # Wigner term
            condition = (N > Z, n2[0] == nuclei[0], n2[1] == nuclei[1])  # PAIRING
            term[9] = {
                (True, True, False): 1 - t / a,
                (True, False, True): 1,
                (False, False, False): 1,
                (True, False, False): t / a,
                (False, False, True): 1 - t / a,
                (False, True, True): 2 - t / a,
                (True, True, True): 2 - t / a,
            }.get(condition, 0)
            term[1:] /= ra
            y[deformed] = np.dot(term, self.coeffs)
        return y[0] if Z < 50 else max(y)  # y[0]->spherical, y[1]->deformed

    @staticmethod
    def compute_nucleons(nuclei):
        n2 = [0, 0]
        for I3 in range(2):
            n2[I3] = 2 * (nuclei[I3] // 2)

        return n2

    @staticmethod
    def compute_subshell_params(nuclei, n2, ju):
        noc = np.zeros((len(nuclei), 2))
        ip_maxs = []
        for I3 in range(2):
            ncum = i = 0
            while True:
                i += 1
                idd = (i + 1) if i % 2 else (i * (i - 2) // 4)
                ncum += idd
                if ncum < nuclei[I3]:
                    noc[i - 1, I3] = idd
                else:
                    break
            ip_max = i + 1
            ip_maxs.append(ip_max)
            ip = (i - 1) // 2
            ipm = i // 2
            moc = nuclei[I3] - ncum + idd
            noc[i - 1, I3] = moc - ju
            noc[i, I3] = ju
        return noc, ip_maxs, ipm

    @staticmethod
    def compute_ei_shell_params(i, ju, moc, ip, ipm):
        oei, dei, qx, dx = [0, 0], [0, 0], [0, 0], [0, 0]
        for I3 in range(2):
            if i % 2:
                oei[I3] = moc + ip * (ip - 1)
                dei[I3] = ip * (ip + 1) + 2
            else:
                oei[I3] = moc - ju
                dei[I3] = (ip + 1) * (ip + 2) + 2
            qx[I3] = oei[I3] * (dei[I3] - oei[I3] - ju) / dei[I3]
            dx[I3] = qx[I3] * (2 * oei[I3] - dei[I3])

        return oei, dei, qx, dx

    @staticmethod
    def compute_amplitudes(i_max, noc, ipm, ju):
        onp = np.zeros((len(noc), 2, 2))
        for I3 in range(2):
            for i in range(1, i_max[I3] + 1):
                ip = (i - 1) // 2
                fact = sqrt((ip + 1) * (ip + 2))
                onp[ip, 0, I3] += noc[i - 1, I3] / fact
                vm = -1.0 if i % 2 else 0.5 * ip
                onp[ip, 1, I3] += noc[i - 1, I3] * vm

        return onp

    @staticmethod
    def compute_fm_and_so_terms(onp, ipm):
        op, os = [0, 0], [0, 0]
        for I3 in range(2):
            for ip in range(ipm + 1):
                den = ((ip + 1) * (ip + 2)) ** (3.0 / 2)
                op[I3] += onp[ip, 0, I3]
                os[I3] += onp[ip, 1, I3] * (1 + onp[ip, 0, I3]) * (ip * ip / den)
                os[I3] += onp[ip, 1, I3] * (1 - onp[ip, 0, I3]) * ((4 * ip - 5) / den)
            op[I3] *= op[I3]

        return op, os

    def compute_spherical_and_deformed(nuclei):
        for deformed in [0, 1]:
            ju = 4 if deformed else 0
            term, noc, onp, os, op = [0] * 5, [0] * 5, [0] * 5, [0] * 5, [0] * 5
            n2 = compute_nucleons(nuclei)
            noc, ip_maxs, ipm = compute_subshell_params(nuclei, n2, ju)
            oei, dei, qx, dx = compute_ei_shell_params(
                ip_maxs[0], ju, noc[ip_maxs[0] - 1, 0], ip_maxs[1] // 2, ip_maxs[1] // 2
            )
            oei, dei, qx, dx = compute_ei_shell_params(
                ip_maxs[0], ju, noc[ip_maxs[0] - 1, 0], ip_maxs[1] // 2, ip_maxs[1] // 2
            )
            if deformed:
                qx = [q / sqrt(d) for q, d in zip(qx, dei)]
            onp = compute_amplitudes(ip_maxs, noc, ip_maxs[1] // 2, ju)
            op, os = compute_fm_and_so_terms(onp, ip_maxs[1] // 2)

        return op, os


dz_be = DuffloZuker10()
