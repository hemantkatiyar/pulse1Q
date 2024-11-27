import numpy as np
from functools import reduce
from scipy.interpolate import CubicSpline as spline
import copy
from scipy.linalg import expm
import matplotlib.pyplot as plt


# matrix multiplication
# check wihtout list
def prod(*arg):
    return reduce(np.matmul, list(arg))


# element wisse multiplication
def eprod(*arg):
    return reduce(np.multiply, list(arg))


# tensor product
def mkron(arg):
    return reduce(np.kron, arg)


# Conjugate transpose
def dagger(x):
    return x.conj().T


def get_spinops(nions, crosstalk):
    sx_temp = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    sy_temp = np.asarray([[0.0, -1.0j], [1.0j, 0.0]])
    sz_temp = np.asarray([[1.0, 0.0], [0.0, -1.0]])

    I = []
    for i in range(3):
        I.append(np.zeros((nions, 2**nions, 2**nions), dtype=complex))

    for l in range(nions):
        list_temp = []
        for i in range(nions):
            list_temp.append(np.eye(2))
        list_temp[l] = sx_temp
        I[0][l] = mkron(list_temp)
        list_temp[l] = sy_temp
        I[1][l] = mkron(list_temp)
        list_temp[l] = sz_temp
        I[2][l] = mkron(list_temp)

    sx = I[0]
    sy = I[1]
    sz = I[2]

    ssx = copy.deepcopy(sx[0])
    ssy = copy.deepcopy(sy[0])
    ssz = copy.deepcopy(sz[0])
    Had = (1 / np.sqrt(2)) * np.asarray([[1, 1], [1, -1]])
    if nions >= 2:
        for ion in range(1, nions):
            ssx += sx[ion] * crosstalk
            ssy += sy[ion] * crosstalk
            ssz += sz[ion] * crosstalk
            Had = np.kron(Had, (1 / np.sqrt(2)) * np.asarray([[1, 1], [1, -1]]))

    return sx, sy, sz, ssx, ssy, np.diag(ssz), Had


def eval_grad(x, z, a):
    return -2 * np.real(np.trace(x) * z) / 2 ** (2 * a)


def calc_uf_eg(u, nsegments, rfi, ssx, ssy, ssz, delT, nions, amp_max, deltaz):
    ux = u[0:nsegments] * amp_max / np.sqrt(2)
    uy = u[nsegments:] * amp_max / np.sqrt(2)
    Uf = np.zeros((len(rfi), nsegments, 2**nions, 2**nions), dtype=complex)

    for rfindex in range(len(rfi)):
        for segment in range(nsegments):
            if segment == 0:
                Uf[rfindex][segment] = expm(
                    -1.0j
                    * delT
                    * rfi[rfindex]
                    * (
                        deltaz * (ssz / 2)
                        + ux[segment] * (ssx / 2)
                        + uy[segment] * (ssy / 2)
                    )
                )
            else:
                Uf[rfindex][segment] = np.matmul(
                    expm(
                        -1.0j
                        * delT
                        * rfi[rfindex]
                        * (
                            deltaz * (ssz / 2)
                            + ux[segment] * (ssx / 2)
                            + uy[segment] * (ssy / 2)
                        )
                    ),
                    Uf[rfindex][segment - 1],
                )

    return Uf


def calc_grad_eg(u, nsegments, rfi, ssx, ssy, ssz, Ut, delT, nions, amp_max, deltaz):
    # Backward propagator
    Uf = calc_uf_eg(u, nsegments, rfi, ssx, ssy, ssz, delT, nions, amp_max, deltaz)
    Ub = np.zeros((len(rfi), nsegments, 2**nions, 2**nions), dtype=complex)

    for rfindex in range(len(rfi)):
        for segment in range(nsegments):
            Ub[rfindex][segment] = np.matmul(
                Uf[rfindex][-1], dagger(Uf[rfindex][segment])
            )

    w0 = 1
    w1 = 1

    grad1 = np.zeros(nsegments)
    grad2 = np.zeros(nsegments)

    for rfindex in range(len(rfi)):
        if rfi[rfindex] < 1 / 4:
            trxjpj = np.trace(np.matmul(dagger(Uf[rfindex][-1]), np.eye(2)))
            for segment in range(nsegments):
                XX = np.matmul(Uf[rfindex][segment], dagger(np.eye(2)))
                grad1[segment] += w0 * eval_grad(
                    prod(Ub[rfindex][segment], ssx / 2, XX), 1.0j * delT * trxjpj, nions
                )
                grad2[segment] += w0 * eval_grad(
                    prod(Ub[rfindex][segment], ssy / 2, XX), 1.0j * delT * trxjpj, nions
                )
        else:
            trxjpj = np.trace(np.matmul(dagger(Uf[rfindex][-1]), Ut))
            for segment in range(nsegments):
                XX = np.matmul(Uf[rfindex][segment], dagger(Ut))
                grad1[segment] += w1 * eval_grad(
                    prod(Ub[rfindex][segment], ssx / 2, XX), 1.0j * delT * trxjpj, nions
                )
                grad2[segment] += w1 * eval_grad(
                    prod(Ub[rfindex][segment], ssy / 2, XX), 1.0j * delT * trxjpj, nions
                )

    grad = np.append(grad1, grad2) / len(rfi)
    return (-grad) * 1e4


def calc_fidelity_ed(
    u, nsegments, rfi, ssx, ssy, ssz, Ut, delT, nions, amp_max, deltaz
):
    ux = u[0:nsegments] * amp_max / np.sqrt(2)
    uy = u[nsegments:] * amp_max / np.sqrt(2)
    Uf = np.zeros((len(rfi), nsegments, 2**nions, 2**nions), dtype=complex)

    w0 = 1
    w1 = 1

    for rfindex in range(len(rfi)):
        for segment in range(nsegments):
            if segment == 0:
                Uf[rfindex][segment] = expm(
                    -1.0j
                    * delT
                    * rfi[rfindex]
                    * (
                        deltaz * (ssz / 2)
                        + ux[segment] * (ssx / 2)
                        + uy[segment] * (ssy / 2)
                    )
                )
            else:
                Uf[rfindex][segment] = np.matmul(
                    expm(
                        -1.0j
                        * delT
                        * rfi[rfindex]
                        * (
                            deltaz * (ssz / 2)
                            + ux[segment] * (ssx / 2)
                            + uy[segment] * (ssy / 2)
                        )
                    ),
                    Uf[rfindex][segment - 1],
                )
    Fid = 0

    for rfindex in range(len(rfi)):
        if rfi[rfindex] < 1 / 4:
            Fid += w0 * abs(np.trace(np.matmul(np.eye(2), Uf[rfindex][-1]))) / 2**nions
        else:
            Fid += w1 * abs(np.trace(np.matmul(dagger(Ut), Uf[rfindex][-1]))) / 2**nions

    infid = (1 - Fid / len(rfi)) * 1e4
    # infid = (1-Fid)*1e4
    # print(infid)
    return infid


def calc_fidelity_ed_noisy(u, nsegments, rfi, ssx, ssy, ssz, Ut, delT, nions, amp_max):
    ux = u[0:nsegments] * amp_max / np.sqrt(2)
    uy = u[nsegments:] * amp_max / np.sqrt(2)
    Uf = np.zeros((len(rfi), nsegments, 2**nions, 2**nions), dtype=complex)

    w0 = 1
    w1 = 1
    a = -2 * np.pi * 200
    b = 2 * np.pi * 200
    deltaz = (b - a) * np.random.rand(nsegments) + a

    for rfindex in range(len(rfi)):
        for segment in range(nsegments):
            if segment == 0:
                Uf[rfindex][segment] = expm(
                    -1.0j
                    * delT
                    * rfi[rfindex]
                    * (
                        deltaz[segment] * (ssz / 2)
                        + ux[segment] * (ssx / 2)
                        + uy[segment] * (ssy / 2)
                    )
                )
            else:
                Uf[rfindex][segment] = np.matmul(
                    expm(
                        -1.0j
                        * delT
                        * rfi[rfindex]
                        * (
                            deltaz[segment] * (ssz / 2)
                            + ux[segment] * (ssx / 2)
                            + uy[segment] * (ssy / 2)
                        )
                    ),
                    Uf[rfindex][segment - 1],
                )
    Fid = 0

    for rfindex in range(len(rfi)):
        if rfi[rfindex] < 1 / 4:
            Fid += w0 * abs(np.trace(np.matmul(np.eye(2), Uf[rfindex][-1]))) / 2**nions
        else:
            Fid += abs(np.trace(np.matmul(dagger(Ut), Uf[rfindex][-1]))) / 2**nions

    infid = (1 - Fid / len(rfi)) * 1e4
    # infid = (1-Fid)*1e4
    # print(infid)
    return infid


def penalty(u, PenaltyRange):
    umUbound = u - PenaltyRange
    HeavSideU = np.heaviside(u - PenaltyRange, 0)
    umLbound = u
    HeavSideL = np.heaviside(-u, 0)

    GrPenal = np.zeros((len(u),))
    PenalFid = 0

    GrPenal = 2 * umUbound * HeavSideU + 2 * umLbound * HeavSideL
    PenalFid += np.sum(2 * (umUbound**2) * HeavSideU + 2 * (umLbound**2) * HeavSideL)

    return GrPenal, PenalFid


def rand_u(R, N, maxAmp):
    x = [1] + list(range(R, N, R - 1)) + [N]
    # y = [0] + ( maxAmp * np.random.rand(len(x) - 2)).tolist() + [0]
    y = (maxAmp * np.random.rand(len(x))).tolist()
    xx = np.arange(1, N + 1)
    cs = spline(x, y)
    u1_r = cs(xx)

    return u1_r


def make_controls(R, N, maxAmp, PenaltyRange):
    u = np.zeros((N,))
    U_penal = PenaltyRange
    L_penal = np.zeros((len(PenaltyRange.tolist())))
    u = np.maximum(np.minimum(U_penal, np.transpose(rand_u(R, N, maxAmp))), L_penal)

    return u


def smooth_pulse(u, nsegments, newnum):
    ux = u[0:nsegments]
    uy = u[nsegments:]

    x = list(range(1, newnum * nsegments + 1, newnum))
    y1 = ux
    y2 = uy
    xx = np.arange(1, newnum * nsegments)
    cs1 = spline(x, y1)
    cs2 = spline(x, y2)

    ux_new = cs1(xx)
    uxx = np.append(ux_new, ux_new[-1])
    uy_new = cs2(xx)
    uyy = np.append(uy_new, uy_new[-1])

    u_new = np.append(uxx, uyy)

    return u_new
