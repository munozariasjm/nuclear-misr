###
# Implementation of the model described in: https://arxiv.org/pdf/0912.0882.pdf
#

import numpy as np
import math


def mass10(N, Z):
    b = np.array([0.7043,17.7418,16.2562,37.5562,53.9017,0.4711,2.1307,0.0210,40.5356,6.0632])
    dyda = np.zeros(10)
    op = np.zeros(2)
    n2 = np.zeros(2)
    dx = np.zeros(2)
    qx = np.zeros(2)
    os = np.zeros(2)
    onp = np.zeros((9,3,3))
    oei = np.zeros(2)
    dei = np.zeros(2)
    nn = np.zeros(2)
    noc = np.zeros((19,3))
    pp = np.zeros(2)
    y = np.zeros(2)

    nn[0] = N
    nn[1] = Z
    a = N + Z
    t = abs(N - Z)
    r = a**(1./3.)
    s = r*r
    rc = r*(1. - .25*(t/a)**2)
    ra = (rc*rc)/r
    z2 = Z*(Z-1)
    dyda[0] = (-z2 + .76*z2**(2./3.))/rc
    for ndef in range(2):
        ju = 0
        y[ndef] = 0.
        if ndef == 1:
            ju = 4
        dyda[1:] = 0.
        for j in range(2):
            noc[:,j] = 0
            onp[:,:,j] = 0.
            n2[j] = 2*(nn[j]//2)
            ncum = 0
            i = 0
            while True:
                i += 1
                i2 = (i//2)*2
                if i2 != i:
                    id = i + 1
                else:
                    id = i * (i - 2) // 4
                ncum += id
                if ncum < nn[j]:
                    noc[i,j] = id
                else:
                    break
            imax = i + 1
            ip = (i - 1)//2
            ipm = i//2
            pp[j] = ip
            moc = nn[j] - ncum + id
            noc[i,j] = moc - ju
            noc[i + 1,j] = ju
            if i2 != i:
                oei[j] = moc + ip * (ip - 1)
                dei[j] = ip * (ip + 1) + 2
            else:
                oei[j] = moc - ju
                dei[j] = (ip + 1) * (ip + 2) + 2
            qx[j] = oei[j] * (dei[j] - oei[j] - ju) / dei[j]
            dx[j] = qx[j] * (2 * oei[j] - dei[j])
            if ndef == 1:
                qx[j] = qx[j] / math.sqrt(dei[j])
            for i in range(imax):
                ip = (i - 1)//2
                fact = math.sqrt((ip + 1.) * (ip + 2.))
                onp[ip,0,j] += noc[i,j]/fact
                vm = -1.
                if 2 * (i//2) == i:
                    vm = .5 * ip
                onp[ip,1,j] += noc[i,j] * vm
            op[j] = 0.
            os[j] = 0.
            for ip in range(ipm + 1):
                pi = ip
                den = ((pi + 1) * (pi + 2))**(3./2.)
                op[j] += onp[ip,0,j]
                os[j] += onp[ip,1,j] * (1. + onp[ip,0,j]) * (pi * pi / den)
                os[j] += onp[ip,1,j] * (1. - onp[ip,0,j]) * ((4 * pi - 5) / den)
            op[j] *= op[j]
        dyda[1] = op[0] + op[1]
        dyda[2] = -dyda[1]/ra
        dyda[1] += os[0] + os[1]
        dyda[3] = -t * (t + 2) / (r * r)
        dyda[4] = -t * (t + 2) / (r * r)
        dyda[5] = 8 * t / (ra * r)
        dyda[6] = -4 * (t + 1) / ra
        dyda[7] = 16 * t / (ra * ra)
        for j in range(2):
            op[j] = 0.
            for i in range(op[j]):
                ip = (i - 1)//2
                fact = math.sqrt((ip + 1) * (ip + 2))
                onp[ip,0,j] += noc[i,j] / fact
                vm = -1.
                if 2 * (i//2) == i:
                    vm = .5 * ip
                onp[ip,1,j] += noc[i,j] * vm
        dyda[8] = -2 * (op[0] + op[1]) * (op[0] + op[1]) / r
        dyda[9] = 2 * (dx[0] + dx[1]) / ra
        for i in range(10):
            y[ndef] += b[i] * dyda[i]
    return y