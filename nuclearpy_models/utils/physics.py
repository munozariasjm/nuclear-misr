import numpy as np


class PhysicsQualities:
    def __init__(self) -> None:
        pass

    @staticmethod
    def S(N, Z):
        # Isospin symmetry breaking
        return N - Z / (N + Z)

    @staticmethod
    def I(N, Z):
        # Isospin symmetry breaking
        return N - Z / (N + Z)

    @staticmethod
    def protons_in_shell(Z):
        """
        Compute the number of protons in the shell closest to the Fermi
        """
        p_magic_numbers = [2, 8, 20, 28, 50, 82, 114, 122, 124, 164]
        closest_smaller = [x for x in p_magic_numbers if x <= Z]
        return Z - closest_smaller[-1]

    @staticmethod
    def neutrons_in_shell(N):
        n_magic_numbers = [2, 8, 20, 28, 50, 82, 126, 184, 196, 236, 318]
        closest_smaller = [x for x in n_magic_numbers if x <= N]
        return N - closest_smaller[-1]

    @staticmethod
    def protons_for_shell(Z):
        """
        Returns the total number of protons required to fill the shell closest to the Fermi surface
        """
        p_magic_numbers = [2, 8, 20, 28, 50, 82, 114, 122, 124, 164]
        closest_bigger = [x for x in p_magic_numbers if x >= Z]
        return closest_bigger[0] - Z

    @staticmethod
    def z_shell(Z):
        """Returns the index of the shell closest to the Fermi surface"""
        p_magic_numbers = [2, 8, 20, 28, 50, 82, 114, 122, 124, 164]
        closest_sma = [x for x in p_magic_numbers if x <= Z]
        return p_magic_numbers.index(closest_sma[0])

    def n_shell(self, N):
        """Returns the index of the shell closest to the Fermi surface"""
        n_magic_numbers = [2, 8, 20, 28, 50, 82, 126, 184, 196, 236, 318]
        closest_sma = [x for x in n_magic_numbers if x <= N]
        return n_magic_numbers.index(closest_sma[0])

    def neutrons_for_shell(self, N):
        """
        Returns the total number of neutrons required to fill the shell closest to the Fermi surface
        """
        n_magic_numbers = [2, 8, 20, 28, 50, 82, 126, 184, 196, 236, 318]
        closest_bigger = [x for x in n_magic_numbers if x >= N]
        return closest_bigger[0] - N

    @np.vectorize
    def compute_P(self, Z, N):
        """promiscuity factor"""
        z_magic_numbers = [2, 8, 20, 28, 50, 82, 126]
        n_magic_numbers = [2, 8, 20, 28, 50, 82, 126, 184]
        # νp(n) is the difference between the proton (neutron) number
        # of a particular nucleus and the nearest magic number.
        clossest_z = min(z_magic_numbers, key=lambda x: abs(x - Z))
        clossest_n = min(n_magic_numbers, key=lambda x: abs(x - N))
        vp = abs(Z - clossest_z)
        vn = abs(N - clossest_n)
        return (vp * vn) / (vp + vn + 1e-6)

    @staticmethod
    def compute_d(Z, N):
        """d is the difference between the number of protons and the number of neutrons"""
        if (Z % 2 == 0) and (N % 2 == 0):
            return 1
        elif (Z % 2 == 1) and (N % 2 == 1):
            return -1
        else:
            return 0

    def compute_Jp(self, Z, N):
        """Compute the nuclear angular momentum of any nuclei and its parity"""
        # Define the shell structure with their angular momentum j and max capacity
        # Define the shell capacities and orbital angular momenta
        shells = [(1, 0), (2, 1), (3, 0), (4, 1), (5, 0), (6, 1), (7, 0), (8, 1)]
        capacities = [2, 8, 20, 28, 50, 82, 126, 184]

        def find_last_filled(nucleons):
            last_cap = [x for x in capacities if x <= nucleons]
            return shells[last_cap.index(max(last_cap))]

        A = Z + N
        if Z % 2 == 0 and N % 2 == 0:  # even-even
            return 0, 1

        if Z % 2 != N % 2:  # even-odd or odd-even
            # Using the shell model, the total angular momentum of an even-odd or odd-even nucleus
            # Counting the number of nucleons in the last filled shell
            # Based on the fermi levels
            valence_z = self.protons_in_shell(Z)
            valence_n = self.neutrons_in_shell(N)
            valence_nucleons = valence_z + valence_n
            # Find the last filled shell

            # Find the angular momentum of the last filled shell
            last_filled_shell_z = find_last_filled(Z)
            last_filled_shell_n = find_last_filled(N)
            last_filled_idx_z = shells.index(last_filled_shell_z)
            last_filled_idx_n = shells.index(last_filled_shell_n)
            # Find the total number of nucleons in the last filled shell
            total_last_filled_idx = last_filled_idx_z + last_filled_idx_n
            # Find the total angular momentum of the last filled shell
            # In even-odd and odd-even nuclei, the half integral spin of a single “extra” nucleon should be
            # combined with the total angular momentum of the rest of the nucleus for a half integral total angular momentum.
            # Total angular momentum will be equal to the half integral angular momentum j of the unpaired particle.
            j = last_filled_shell_z[1] + last_filled_shell_n[1] + 1 / 2
            l = last_filled_shell_z[0] + last_filled_shell_n[0]
            p = (-1) ** (l)

        else:  # odd-odd
            # Odd-odd nuclei each have an extra neutron and an extra proton whose half-integral spins should
            # yield integral total angular momenta.
            valence_z = self.protons_in_shell(Z)
            valence_n = self.neutrons_in_shell(N)
            valence_nucleons = valence_z + valence_n
            # Find the last filled shell
            last_filled_shell_z = find_last_filled(Z)
            last_filled_shell_n = find_last_filled(N)

            l_protons = last_filled_shell_z[0]
            l_neutrons = last_filled_shell_n[0]

            j_protons = last_filled_shell_z[1]
            j_neutrons = last_filled_shell_n[1]

            if (l_protons + l_neutrons + j_neutrons + j_protons) % 2 == 0:
                j = np.abs(j_protons - j_neutrons)
            else:
                j = j_protons + j_neutrons

            p = (-1) ** (l_protons + l_neutrons)

        return j, p

    def compute_j(self, Z, N):
        """j is the total angular momentum of the valence nucleon"""
        j, _ = self.compute_Jp(Z, N)
        return j

    def parity(self, Z, N):
        """p is the parity of the valence nucleon"""
        _, p = self.compute_Jp(Z, N)
        return p

    def compute_Phi(self, Z, N):
        """Phi is the parity of the valence nucleon times the parity of the nucleus"""
        return self.parity(Z, N) * self.compute_d(Z, N)

    # Distributions
    def rho(self, Z, rc):
        """Spherical charge density"""
        return Z / (rc**3)

    def coulumb(self, Z, rc):
        """Coulumb energy kind"""
        return (Z**2) / (rc)

    # Vibrational indications
    def monopole(self, J, p):
        """Monopole energy kind indicator"""
        return 1 if (J == 0 and p == 1) else 0

    def dipole(self, J, p):
        """Dipole energy kind indicator"""
        return 1 if (J == 1 and p == -1) else 0

    def quadrupole(self, J, p):
        """Quadrupole energy kind indicator"""
        return 1 if (J % 2 == 0 and p == 1) else 0

    def p2(self, N, Z):
        z_magic_numbers = [2, 8, 20, 28, 50, 82, 114, 122, 124, 164]
        n_magic_numbers = [8, 20, 28, 50, 82, 126, 184, 196, 236, 318]
        # νp(n) is the difference between the proton (neutron) number
        # of a particular nucleus and the nearest magic number.
        clossest_z = min(z_magic_numbers, key=lambda x: abs(x - Z))
        clossest_n = min(n_magic_numbers, key=lambda x: abs(x - N))
        vp = abs(Z - clossest_z)
        vn = abs(N - clossest_n)
        return (vp * vn) / (vp + vn + 1)
