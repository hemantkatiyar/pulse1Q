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
    return reduce(np.multiply,list(arg))

# tensor product
def mkron(arg):
    return reduce(np.kron, arg)

# Conjugate transpose
def dagger(x):
    return x.conj().T

# def get_fourier_pulse(n, amplitude=1.0, frequency=1.0, phase=0.0, num_points=1000):
def get_fourier_pulse( x , del_t, nsegments, amp_max,penalty_range):

    T = nsegments * del_t
    freq = 1/T
    t = np.linspace(0, T , nsegments)  # Generate 'num_points' data points from 0 to 2π
    pulse = np.zeros(nsegments)  # Initialize the signal to zeros

   
    n = int(len(x.tolist())/3)
    for i in range(n):
        amp = x[3*i]
        fre = x[3*i+1]
        phase = x[3*i+2]
        component = amp * np.sin(2 * np.pi * freq * fre * 2 * t +  phase)
        pulse += component

    pulse = amp_max*np.abs(pulse)/np.max(np.abs(pulse))
    
    for i in range(nsegments):
        if pulse[i] > penalty_range[i]:
            pulse[i] = penalty_range[i]
            
    return pulse

def get_supergaussian_pulse(x,del_t, nsegments, amp_max):
    tau = nsegments * del_t
    timestamps = np.linspace(0, tau, nsegments)

    # generate amplitude before applying the penalty
    sigma = tau/x[1]
    pulse = amp_max*x[0]*np.exp(-((timestamps-tau/2)**2/(2*sigma**2))**x[2])
    # plt.plot(pulse)
    return pulse
   
def get_gaussian_pulse(x,del_t, nsegments, amp_max):
    tau = nsegments * del_t
    timestamps = np.linspace(0, tau, nsegments)

    # generate amplitude before applying the penalty
    pulse = np.exp(-((timestamps-tau/2)**2/(2*(tau*x[1]/6)**2)))
    pulse = pulse/np.max(pulse)
    pulse = x[0]*amp_max * pulse
    return pulse 

def get_der_gaussian_pulse(x,del_t, nsegments,amp_max):
    tau = nsegments * del_t
    timestamps = np.linspace(0, tau, nsegments)

    # generate amplitude before applying the penalty
    pulse = np.exp(-((timestamps-tau/2)**2/(2*(tau*x[1]/6)**2)))
    ux = x[0] * amp_max * pulse
    pulse_uy = pulse * (-((timestamps-tau/2)/((tau*x[1]/6)**2)))
    uy = x[2] * amp_max * pulse_uy / np.max(pulse_uy) / 5
    
    return ux, uy 

def get_spinops(nions,crosstalk):
    sx_temp = np.asarray([[0.0, 1.0],[1.0, 0.0]])
    sy_temp = np.asarray([[0.0, -1.0j],[1.0j, 0.0]])
    sz_temp = np.asarray([[1.0, 0.0],[0.0, -1.0]])
    
    I = []
    for i in range(3):
        I.append(
            np.zeros((nions, 2**nions, 2**nions), dtype=complex))
        
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
    sz= I[2]
    
    ssx = copy.deepcopy(sx[0])
    ssy = copy.deepcopy(sy[0])
    ssz = copy.deepcopy(sz[0])
    Had = (1 / np.sqrt(2)) * np.asarray([[1,1],[1,-1]])
    if nions >= 2:
        for ion in range(1,nions):
            ssx += sx[ion] * crosstalk
            ssy += sy[ion] * crosstalk
            ssz += sz[ion] * crosstalk
            Had = np.kron(Had,(1 / np.sqrt(2)) * np.asarray([[1,1],[1,-1]]));
    
    return sx, sy, sz, ssx, ssy, np.diag(ssz), Had

def calc_fidelity(u, nsegments, rfi, ssx, W1, W2, Ut, delT,nions):
    
    Uf = (np.zeros((rfi.shape[0],nsegments,2**nions,2**nions), dtype=complex))
    
    for rfindex in range(rfi.shape[0]):
        for segment in range(nsegments):
            if segment == 0:
                # Uf[rfindex][segment] =  np.matmul(W1 , eprod(
                #     np.exp(-1.0j * delT * u[segment] * rfi[rfindex, 0] * (ssz / 2)) , W2))
                Uf[rfindex][segment] = expm(-1.0j * delT * u[segment] * rfi[rfindex,0] * (ssx / 2))
            else:
                Uf[rfindex][segment] = np.matmul(expm(-1.0j * delT * u[segment] 
                                                      * rfi[rfindex,0] * (ssx / 2)),Uf[rfindex][segment-1])
                # Uf[rfindex][segment] =  np.matmul(np.matmul(W1 , eprod(
                #     np.exp(-1.0j * delT * u[segment] * rfi[rfindex, 0] * (ssz / 2)) , W2))
                #                                   ,Uf[rfindex][segment-1])
    Fid = 0
    for rfindex in range(rfi.shape[0]):
        Fid += rfi[rfindex, 1] * abs(np.trace(np.matmul(dagger(Ut),Uf[rfindex][-1]))) / 2**nions
    
    return Fid, Uf

def calc_fidelity2(u, nsegments, rfi, ssx, Ut, delT, nions):
    
    Uf = (np.zeros((rfi.shape[0],nsegments,2**nions,2**nions), dtype=complex))
    
    for rfindex in range(rfi.shape[0]):
        for segment in range(nsegments):
            if segment == 0:
                Uf[rfindex][segment] = expm(-1.0j * delT * u[segment] * rfi[rfindex,0] * (ssx / 2))
            else:
                Uf[rfindex][segment] = np.matmul(expm(-1.0j * delT * u[segment] 
                                                      * rfi[rfindex,0] * (ssx / 2)),Uf[rfindex][segment-1])

    Fid = 0
    for rfindex in range(rfi.shape[0]):
        Fid += rfi[rfindex, 1] * abs(np.trace(np.matmul(dagger(Ut),Uf[rfindex][-1]))) / 2**nions
    
    infid = (1-Fid)*1e4
    return infid


def calc_fidelity3(x, nsegments, rfi, ssx, Ut, delT, nions, amp_max):
    
    u = get_gaussian_pulse(x, delT, nsegments, amp_max)
    Uf = (np.zeros((rfi.shape[0],nsegments,2**nions,2**nions), dtype=complex))
    
    for rfindex in range(rfi.shape[0]):
        for segment in range(nsegments):
            if segment == 0:
                Uf[rfindex][segment] = expm(-1.0j * delT * u[segment] * rfi[rfindex,0] * (ssx / 2))
            else:
                Uf[rfindex][segment] = np.matmul(expm(-1.0j * delT * u[segment] 
                                                      * rfi[rfindex,0] * (ssx / 2)),Uf[rfindex][segment-1])

    Fid = 0
    for rfindex in range(rfi.shape[0]):
        Fid += rfi[rfindex, 1] * abs(np.trace(np.matmul(dagger(Ut),Uf[rfindex][-1]))) / 2**nions
    
    infid = (1-Fid)*1e4
    return infid
    
    # infid1 = (1-Fid)*1e4
    # numslice = 5
    # Fcost = 0
    # coef_slice = np.linspace(0,1/300,numslice)
    # for j in range(numslice):
    #     Uf_temp = (np.zeros((rfi.shape[0],nsegments,2**nions,2**nions), dtype=complex))
    #     for segment in range(nsegments):
    #         if segment == 0:
    #             Uf_temp[rfindex][segment] = expm(-1.0j * delT * coef_slice[j] * (u[segment] * rfi[rfindex,0] * (ssx / 2) 
    #                                                      ))
    #         else:
    #             Uf_temp[rfindex][segment] = np.matmul( expm(-1.0j * delT * coef_slice[j] * (u[segment] * rfi[rfindex,0] * (ssx / 2) 
    #                                                                 )),
    #                                                                    Uf_temp[rfindex][segment-1])
        
    #     # print(Uf_temp[rfindex][-1])
    #     Fcost += abs(np.trace(np.matmul(np.eye(2),Uf_temp[rfindex][-1]))) / 2**nions
    
    # nums = 20
    # w0 = 1/nums
    # w1 = (nums-1)/nums
    
    # infid2 = (1-Fcost/numslice)*1e4
    # print(infid2)
    
    # infid = (w0 * infid1 + w1 * infid2)
    # return infid

def calc_fidelity4(x, nsegments, rfi, ssx, ssy, Ut, delT, nions, amp_max):
    
    ux, uy = get_der_gaussian_pulse(x, delT, nsegments, amp_max)
    Uf = (np.zeros((rfi.shape[0],nsegments,2**nions,2**nions), dtype=complex))
    
    for rfindex in range(rfi.shape[0]):
        for segment in range(nsegments):
            if segment == 0:
                Uf[rfindex][segment] = expm(-1.0j * delT * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                         +  uy[segment] * rfi[rfindex,0] * (ssy / 2)))
            else:
                Uf[rfindex][segment] = np.matmul( expm(-1.0j * delT * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                                    +  uy[segment] * rfi[rfindex,0] * (ssy / 2))),
                                                                       Uf[rfindex][segment-1])

    Fid = 0
    for rfindex in range(rfi.shape[0]):
        Fid += rfi[rfindex, 1] * abs(np.trace(np.matmul(dagger(Ut),Uf[rfindex][-1]))) / 2**nions
    
    infid = (1-Fid)*1e4
    return infid

def calc_fidelity5(u, nsegments, rfi, ssx, ssy, Ut, delT, nions, amp_max):
    
    ux = u[0:nsegments] * amp_max / np.sqrt(2)
    uy = u[nsegments:] * amp_max / np.sqrt(2)
    Uf = (np.zeros((rfi.shape[0],nsegments,2**nions,2**nions), dtype=complex))
    
    for rfindex in range(rfi.shape[0]):
        for segment in range(nsegments):
            if segment == 0:
                Uf[rfindex][segment] = expm(-1.0j * delT * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                         +  uy[segment] * rfi[rfindex,0] * (ssy / 2)))
            else:
                Uf[rfindex][segment] = np.matmul( expm(-1.0j * delT * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                                    +  uy[segment] * rfi[rfindex,0] * (ssy / 2))),
                                                                       Uf[rfindex][segment-1])

    Fid = 0
    for rfindex in range(rfi.shape[0]):
        Fid += rfi[rfindex, 1] * abs(np.trace(np.matmul(dagger(Ut),Uf[rfindex][-1]))) / 2**nions
    
    infid = (1-Fid)*1e4
    return infid

def calc_fidelity_5_ampphi(amp, phi , nsegments, rfi, ssx, ssy, Ut, delT, nions):
    
    ux = amp * np.cos(phi)
    uy = amp * np.sin(phi)
    Uf = (np.zeros((rfi.shape[0],nsegments,2**nions,2**nions), dtype=complex))
    
    for rfindex in range(rfi.shape[0]):
        for segment in range(nsegments):
            if segment == 0:
                Uf[rfindex][segment] = expm(-1.0j * delT * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                         +  uy[segment] * rfi[rfindex,0] * (ssy / 2)))
            else:
                Uf[rfindex][segment] = np.matmul( expm(-1.0j * delT * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                                    +  uy[segment] * rfi[rfindex,0] * (ssy / 2))),
                                                                       Uf[rfindex][segment-1])

    Fid = 0
    for rfindex in range(rfi.shape[0]):
        Fid += rfi[rfindex, 1] * abs(np.trace(np.matmul(dagger(Ut),Uf[rfindex][-1]))) / 2**nions
    
    infid = (1-Fid)*1e4
    return infid

def calc_fidelity6(u, nsegments, rfi, ssx, ssy, Ut, delT, nions, amp_max):
    
    ux = u[0:nsegments] * amp_max / np.sqrt(2)
    uy = u[nsegments:] * amp_max / np.sqrt(2)
    Uf = (np.zeros((rfi.shape[0],nsegments,2**nions,2**nions), dtype=complex))
    
    for rfindex in range(rfi.shape[0]):
        for segment in range(nsegments):
            if segment == 0:
                Uf[rfindex][segment] = expm(-1.0j * delT * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                         +  uy[segment] * rfi[rfindex,0] * (ssy / 2)))
            else:
                Uf[rfindex][segment] = np.matmul( expm(-1.0j * delT * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                                    +  uy[segment] * rfi[rfindex,0] * (ssy / 2))),
                                                                       Uf[rfindex][segment-1])

    Fid = 0
    for rfindex in range(rfi.shape[0]):
        Fid += rfi[rfindex, 1] * abs(np.trace(np.matmul(dagger(Ut),Uf[rfindex][-1]))) / 2**nions
    
    infid1 = (1-Fid)*1e4
    
    numslice = 5
    Fcost = 0
    coef_slice = np.linspace(0,1/300,numslice)
    for j in range(numslice):
        Uf_temp = (np.zeros((rfi.shape[0],nsegments,2**nions,2**nions), dtype=complex))
        for rfindex in range(rfi.shape[0]):
            for segment in range(nsegments):
                if segment == 0:
                    Uf_temp[rfindex][segment] = expm(-1.0j * delT * coef_slice[j] * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                            +  uy[segment] * rfi[rfindex,0] * (ssy / 2)))
                else:
                    Uf_temp[rfindex][segment] = np.matmul( expm(-1.0j * delT * coef_slice[j] * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                                        +  uy[segment] * rfi[rfindex,0] * (ssy / 2))),
                                                                        Uf_temp[rfindex][segment-1])
        
        Fid_temp = 0
        for rfindex in range(rfi.shape[0]):
            Fid_temp += rfi[rfindex, 1] * abs(np.trace(np.matmul(np.eye(2),Uf_temp[rfindex][-1]))) / 2**nions
    
    
        # print(Uf_temp[rfindex][-1])
        Fcost += Fid_temp
    
    nums = 20
    w0 = 1/nums
    w1 = (nums-1)/nums
    
    infid2 = (1-Fcost/numslice)*1e4
    print(u)
    print(infid2)
     
    infid = (w0 * infid1 + w1 * infid2)
    print(infid)
    # print(infid1)
    return infid
    
    
def calc_fidelity7(x, nsegments, rfi, ssx, ssy, Ut, delT, nions, amp_max):
    
    ux, uy = get_der_gaussian_pulse(x, delT, nsegments, amp_max)
    Uf = (np.zeros((rfi.shape[0],nsegments,2**nions,2**nions), dtype=complex))
    
    for rfindex in range(rfi.shape[0]):
        for segment in range(nsegments):
            if segment == 0:
                Uf[rfindex][segment] = expm(-1.0j * delT * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                         +  uy[segment] * rfi[rfindex,0] * (ssy / 2)))
            else:
                Uf[rfindex][segment] = np.matmul( expm(-1.0j * delT * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                                    +  uy[segment] * rfi[rfindex,0] * (ssy / 2))),
                                                                       Uf[rfindex][segment-1])

    Fid = 0
    for rfindex in range(rfi.shape[0]):
        Fid += rfi[rfindex, 1] * abs(np.trace(np.matmul(dagger(Ut),Uf[rfindex][-1]))) / 2**nions
    
    infid = (1-Fid)*1e4
    infid1 = (1-Fid)*1e4
    
    numslice = 5
    Fcost = 0
    coef_slice = np.linspace(0,1/300,numslice)
    for j in range(numslice):
        Uf_temp = (np.zeros((rfi.shape[0],nsegments,2**nions,2**nions), dtype=complex))
        for segment in range(nsegments):
            if segment == 0:
                Uf[rfindex][segment] = expm(-1.0j * delT * coef_slice[j] * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                         +  uy[segment] * rfi[rfindex,0] * (ssy / 2)))
            else:
                Uf[rfindex][segment] = np.matmul( expm(-1.0j * delT * coef_slice[j] * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                                    +  uy[segment] * rfi[rfindex,0] * (ssy / 2))),
                                                                       Uf[rfindex][segment-1])
        
        Fcost += abs(np.trace(np.matmul(np.eye(2),Uf_temp[rfindex][-1]))) / 2**nions
        
    nums = 20
    w0 = 1/nums
    w1 = (nums-1)/nums
    
    infid2 = (1-Fcost/numslice)*1e4
    print(infid2)
    
    infid = (w0 * infid1 + w1 * infid2)
    return infid

def calc_fidelity_t(u, nsegments, rfi, ssx, ssy, Ut, nions, amp_max):
    """ for am/pm individual slices and time variation"""
    ux = u[0:nsegments] * amp_max / np.sqrt(2)
    uy = u[nsegments:2*nsegments] * amp_max / np.sqrt(2)
    delT = u[2*nsegments:] * 5e-6
    Uf = (np.zeros((rfi.shape[0],nsegments,2**nions,2**nions), dtype=complex))
    
    for rfindex in range(rfi.shape[0]):
        for segment in range(nsegments):
            if segment == 0:
                Uf[rfindex][segment] = expm(-1.0j * delT[segment] * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                         +  uy[segment] * rfi[rfindex,0] * (ssy / 2)))
            else:
                Uf[rfindex][segment] = np.matmul( expm(-1.0j * delT[segment] * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                                    +  uy[segment] * rfi[rfindex,0] * (ssy / 2))),
                                                                       Uf[rfindex][segment-1])

    Fid = 0
    for rfindex in range(rfi.shape[0]):
        Fid += rfi[rfindex, 1] * abs(np.trace(np.matmul(dagger(Ut),Uf[rfindex][-1]))) / 2**nions
    
    infid1 = (1-Fid)*1e4
    
    numslice = 5
    Fcost = 0
    coef_slice = np.linspace(0,1/300,numslice)
    for j in range(numslice):
        Uf_temp = (np.zeros((rfi.shape[0],nsegments,2**nions,2**nions), dtype=complex))
        for segment in range(nsegments):
            if segment == 0:
                Uf_temp[rfindex][segment] = expm(-1.0j * delT[segment] * coef_slice[j] * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                         +  uy[segment] * rfi[rfindex,0] * (ssy / 2)))
            else:
                Uf_temp[rfindex][segment] = np.matmul( expm(-1.0j * delT[segment] * coef_slice[j] * (ux[segment] * rfi[rfindex,0] * (ssx / 2) 
                                                                    +  uy[segment] * rfi[rfindex,0] * (ssy / 2))),
                                                                       Uf_temp[rfindex][segment-1])
        
        # print(Uf_temp[rfindex][-1])
        Fcost += abs(np.trace(np.matmul(np.eye(2),Uf_temp[rfindex][-1]))) / 2**nions
    
    nums = 20
    w0 = 1/nums
    w1 = (nums-1)/nums
    
    infid2 = (1-Fcost/numslice)*1e4
    print(u)
    print(infid2)
    
    infid = (w0 * infid1 + w1 * infid2)
    return infid




def eval_grad(x, z, a):
    return  -2 * np.real(np.trace(x) * z) / 2**(2 * a)

def calc_grad(Uf, nions, rfi,nsegments,Ut,gr_penal,penalty_weight,ssx,delT):
    # Backward propagator
    Ub = (np.zeros((rfi.shape[0],nsegments,2**nions,2**nions), dtype=complex))
    
    for rfindex in range(rfi.shape[0]):
        for segment in range(nsegments):
            Ub[rfindex][segment] = np.matmul(Uf[rfindex][-1], dagger(Uf[rfindex][segment])) 
    
    Grad = np.zeros(nsegments)

    for rfindex in range(rfi.shape[0]):
        trxjpj = np.trace(np.matmul(dagger(Uf[rfindex][-1]),Ut))
        for segment in range(nsegments):
            XX = np.matmul(Uf[rfindex][segment],dagger(Ut))
            Grad[segment] += rfi[rfindex, 1] * (eval_grad(prod(Ub[rfindex][segment],ssx/2,XX),
                                                             (1.0j * delT) * trxjpj, nions) - penalty_weight * gr_penal[segment])

    return  Grad

def calc_uf(u, nsegments, rfi, ssx,ssy, delT,nions, amp_max):
    
    ux = u[0:nsegments] * amp_max / np.sqrt(2)
    uy = u[nsegments:] * amp_max / np.sqrt(2)
    Uf = (np.zeros((len(rfi),nsegments,2**nions,2**nions), dtype=complex))
    
    for rfindex in range(len(rfi)):
        for segment in range(nsegments):
            if segment == 0:
                Uf[rfindex][segment] = expm(-1.0j * delT * rfi[rfindex] * (ux[segment] * (ssx / 2) 
                                                                          +  uy[segment] * (ssy / 2)))
            else:
                Uf[rfindex][segment] = np.matmul(expm(-1.0j * delT * rfi[rfindex] * (ux[segment] * (ssx / 2) 
                                                                                    +  uy[segment] * (ssy / 2))),
                                                                                       Uf[rfindex][segment-1])
                
    return Uf

def calc_grad2(u, nsegments, rfi, ssx, ssy, Ut, delT, nions, amp_max):
    # Backward propagator
    Uf = calc_uf(u, nsegments, rfi, ssx, ssy, delT, nions, amp_max)
    Ub = (np.zeros((len(rfi),nsegments,2**nions,2**nions), dtype=complex))
    
    for rfindex in range(len(rfi)):
        for segment in range(nsegments):
            Ub[rfindex][segment] = np.matmul(Uf[rfindex][-1], dagger(Uf[rfindex][segment])) 
    
    w0 = 1
    w1 = 1
    
    grad1 = np.zeros(nsegments)
    grad2 = np.zeros(nsegments)

    for rfindex in range(len(rfi)):
        if rfi[rfindex] < 1/4:
            trxjpj = np.trace(np.matmul(dagger(Uf[rfindex][-1]),np.eye(2)))
            for segment in range(nsegments):
                XX = np.matmul(Uf[rfindex][segment],dagger(np.eye(2)))
                grad1[segment] += w0 * eval_grad(prod(Ub[rfindex][segment],ssx/2,XX),
                                                                1.0j * delT * trxjpj, nions)
                grad2[segment] += w0 * eval_grad(prod(Ub[rfindex][segment],ssy/2,XX),
                                                                1.0j * delT * trxjpj, nions)
        else:
            trxjpj = np.trace(np.matmul(dagger(Uf[rfindex][-1]),Ut))
            for segment in range(nsegments):
                XX = np.matmul(Uf[rfindex][segment],dagger(Ut))
                grad1[segment] += w1 * eval_grad(prod(Ub[rfindex][segment],ssx/2,XX),
                                                                1.0j * delT * trxjpj, nions)
                grad2[segment] += w1 * eval_grad(prod(Ub[rfindex][segment],ssy/2,XX),
                                                                1.0j * delT * trxjpj, nions)

    grad = np.append(grad1,grad2) / len(rfi)
    return (-grad)*1e4

def calc_fidelity6_ct(u, nsegments, rfi, ssx, ssy, Ut, delT, nions, amp_max):
    
    ux = u[0:nsegments] * amp_max / np.sqrt(2)
    uy = u[nsegments:] * amp_max / np.sqrt(2)
    Uf = (np.zeros((len(rfi),nsegments,2**nions,2**nions), dtype=complex))
    
    w0 = 1
    w1 = 1
    
    for rfindex in range(len(rfi)):
        for segment in range(nsegments):
            if segment == 0:
                Uf[rfindex][segment] = expm(-1.0j * delT * rfi[rfindex] * (ux[segment] * (ssx / 2) 
                                                                          +  uy[segment] * (ssy / 2)))
            else:
                Uf[rfindex][segment] = np.matmul(expm(-1.0j * delT * rfi[rfindex] * (ux[segment] * (ssx / 2) 
                                                                                    +  uy[segment] * (ssy / 2))),
                                                                                       Uf[rfindex][segment-1])
    Fid = 0
    for rfindex in range(len(rfi)):
        if rfi[rfindex] < 1/4:
            Fid += w0 * abs(np.trace(np.matmul(np.eye(2),Uf[rfindex][-1]))) / 2**nions 
        else:
            Fid += w1 * abs(np.trace(np.matmul(dagger(Ut),Uf[rfindex][-1]))) / 2**nions
    
    infid = (1-Fid/len(rfi))*1e4
    # infid = (1-Fid)*1e4
    # print(infid)
    return infid

def conjugate_grad(resetflag, iter, Grad, oldGrad, oldDirc, FidCurrent, stepsize, nions, nsegments, rfi, ssz, W1, W2, Ut, u, delT, penalty_weight, PenaltyRange):
    if iter != 1 and not resetflag:
        diffDerivs = Grad - oldGrad
        beta = np.sum(Grad * diffDerivs) / np.sum(oldGrad**2)
    else:
        beta = 0

    # Do a sort of reset. If we have really lost conjugacy then beta will be
    # negative. If we then reset beta to zero, we start with the steepest descent again.
    beta = np.max([beta, 0])

    # Define the good direction as the linear combination
    goodDirc = Grad + beta * oldDirc

    # Finding the best step size
    multRange = np.array([0, 1, 2])
    FidTemp = np.zeros(3)
    FidTemp[0] = FidCurrent

    for j in range(1,3):
        uTemp = u + multRange[j] * stepsize * goodDirc
        _, PenalFid = penalty(uTemp, PenaltyRange)
        fidout, _ = calc_fidelity(uTemp, nsegments, rfi, ssz, W1, W2, Ut, delT, nions) 
        FidTemp[j] = fidout - penalty_weight * PenalFid
    # We have three points to fit a quadratic to. The matrix to obtain
    # the [a b c] coordinates for fitting points 0, 1, 2 is
    h = 1
    fitcoeffs = np.matmul(np.array([[1 / (2 * h**2), -(1 / h)**2, 1 / (2 * h**2)],
                          [-(1 / h), 1 / h, 0],
                          [1, 0, 0]]),FidTemp)

    # If the quadratic is negative, this method did not work, so just
    # go for the maximum value

    if fitcoeffs[0] > 0:
        maxindex = np.argmax(FidTemp)
        maxmult = multRange[maxindex]
    # Otherwise, choose the maximum of the quadratic
    else:
        maxmult = -fitcoeffs[1] / (2 * fitcoeffs[0])

    # Move by at least 0.1 and at most 2
    maxmult = np.min([np.max([maxmult, 0.1]), 2])

    return goodDirc, maxmult


def penalty(u, PenaltyRange):
    
    umUbound = u - PenaltyRange
    HeavSideU = np.heaviside(u - PenaltyRange, 0)
    umLbound = u 
    HeavSideL = np.heaviside(-u, 0 )
    
    
    GrPenal = np.zeros((len(u),))
    PenalFid = 0
    
    GrPenal = 2 * umUbound * HeavSideU + 2 * umLbound * HeavSideL
    PenalFid += np.sum(2 * (umUbound**2) * HeavSideU + 2 * (umLbound**2) * HeavSideL)

    return GrPenal, PenalFid


def rand_u(R, N, maxAmp):
    x = [1] + list(range(R, N, R - 1)) + [N]
    # y = [0] + ( maxAmp * np.random.rand(len(x) - 2)).tolist() + [0]
    y =  ( maxAmp * np.random.rand(len(x))).tolist() 
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

def smooth_pulse(u,nsegments,newnum):
    ux = u[0:nsegments]
    uy = u[nsegments:]
    
    x = list(range(1,newnum*nsegments+1,newnum))
    y1 = ux
    y2 = uy
    xx = np.arange(1, newnum*nsegments)
    cs1 = spline(x, y1)
    cs2 = spline(x, y2)
    
    ux_new = cs1(xx); uxx = np.append(ux_new,ux_new[-1])
    uy_new = cs2(xx); uyy = np.append(uy_new,uy_new[-1])
    
    u_new = np.append(uxx,uyy)
    
    return u_new
    