import time
import numpy as np
from scipy.linalg import lstsq
from Constants_and_functions import Kepler_U, Lagrange_ceff_universal, mu 

def karimi_exact(tnp, pos_obs_nd, losnp, tol=1e-8, it_max=500):
    start_time = time.time()

    l = len(tnp)
    t_ = (tnp - tnp[0])
    t_diff = np.diff(t_)
    tau1,tau3 = np.array([-t_diff[:-1],t_diff[1:]])

    f1 = f3 = 1
    g1,g3 = tau1, tau3 

    # non-zero initial guess
    # fg = f1*g3 - f3*g1
    # c_k, d_k = g3/fg, -g1/fg
    fg_in = f1, g1, f3, g3

    # M0, Xi0 = D_M(c_k, d_k, pos_obs_nd, losnp)
    # rho0,_resi,_rnk,_s = lstsq(M0,Xi0) #Gauss n angles

    # zero initial guess
    rho0 = np.zeros((l,1))

    # refine Gauss n angles
    rho, f1_i, f3_i, g1_i, g3_i = refine_ranges(rho0, pos_obs_nd, losnp, tau1, tau3, fg_in, tol, it_max)   
    r_vec = np.array(pos_obs_nd).reshape((l,3))  + rho.reshape(l,1)*losnp

    # Determine mid-point velocity vectors 
    router1_vec, rmid_vec, router3_vec = r_vec[:-2], r_vec[1:-1], r_vec[2:]
    f1, f3, g1, g3 = f1_i.reshape(l-2,1), f3_i.reshape(l-2,1), g1_i.reshape(l-2,1), g3_i.reshape(l-2,1)
    vmid2_vec = (-f3*router1_vec + f1*router3_vec)/(f1*g3 - f3*g1)

    comp_time = time.time() - start_time

    return r_vec, comp_time, vmid2_vec


def refine_ranges(rho, pos_obs_nd, los, tau1, tau3, fg_in, gtol, it_max):
    l = len(rho)
    f1, g1, f3, g3 = fg_in

    # Initial (un-refined) solution
    r_vec = pos_obs_nd + rho.reshape((l,1))*los 

    # Determine initial mid-point velocity vectors. {r1, r2, r3} -> v2 etc
    router1_vec, rmid_vec, router3_vec = r_vec[:-2], r_vec[1:-1], r_vec[2:]

    # Set remaining intial values for refinement
    k = 0    
    diff = 10
    rho_prev = rho
    r1_i, r2_i, r3_i = router1_vec, rmid_vec, router3_vec
    f1_i, g1_i, f3_i, g3_i = np.array([f1]*(l-2)), g1, np.array([f3]*(l-2)), g3

    while np.all(diff > gtol) and (k < it_max): 
        k += 1

        # Reshape to allow matrix operation for variable tracklet length, l
        rq_shape = r1_i.shape
        g1_i = np.broadcast_to(g1_i[:, np.newaxis], rq_shape)
        f1_i = np.broadcast_to(f1_i[:, np.newaxis], rq_shape)
        g3_i = np.broadcast_to(g3_i[:, np.newaxis], rq_shape)
        f3_i = np.broadcast_to(f3_i[:, np.newaxis], rq_shape)
        
        # Update middle vectors with new lagrange coefficients
        vmid2_vec = (-f3_i*r1_i + f1_i*r3_i)/(f1_i*g3_i - f3_i*g1_i)
        rn = np.linalg.norm(r2_i, axis=1)
        v_rad = np.sum(vmid2_vec*r2_i, axis=1) / rn
        vn = np.linalg.norm(vmid2_vec, axis=1) # mid
        a = 2/rn - vn**2 / mu

        # Determine universal anomalies for each triplet
        x1 = [Kepler_U(tau1_, rn_, vrad_, a_, mu) for tau1_, rn_, vrad_, a_ in zip(tau1, rn, v_rad, a)]
        x3 = [Kepler_U(tau3_, rn_, vrad_, a_, mu) for tau3_, rn_, vrad_, a_ in zip(tau3, rn, v_rad, a)]

        # Determine Lagrange coefficients for each triplet
        fg1 = np.array([Lagrange_ceff_universal(a_, x1_, rn_, tau1_, mu) for a_, x1_, rn_, tau1_ in zip(a, x1, rn, tau1)])
        fg3 = np.array([Lagrange_ceff_universal(a_, x3_, rn_, tau3_, mu) for a_, x3_, rn_, tau3_ in zip(a, x3, rn, tau3)])
        # Update lagrange coefficients (same as in Gauss 3 angles)
        # Uses first elements of fg etc. out of simplicity, all elements are same to allow for matrix mult.:
        # f1_i = [array([0.99903824, 0.99903824, 0.99903824])]
        f1_i = (f1_i[:,0] + fg1[:,0]) / 2
        g1_i = (g1_i[:,0] + fg1[:,1]) / 2
        f3_i = (f3_i[:,0] + fg3[:,0]) / 2
        g3_i = (g3_i[:,0] + fg3[:,1]) / 2

        fg = f1_i*g3_i - f3_i*g1_i
        c_k, d_k = g3_i/fg, -g1_i/fg

        # Set up design matrix and solve for updated ranges
        M, Xi = D_M(c_k, d_k, pos_obs_nd, los)
        rho, _resi, _rnk, _s = lstsq(M, Xi)

        # Quantify convergence
        diff = abs(rho - rho_prev)

        # Compute updated vectors, extract outer and middle
        r_vec = np.array(pos_obs_nd).reshape((l,3))  + rho.reshape(l,1)*los
        r1_i, r2_i, r3_i = r_vec[:-2], r_vec[1:-1], r_vec[2:] 

        rho_prev = rho

    if k > it_max:
        print('Gauss (n>=3) refinement executed with max iterations (k={k}).')

    return rho, f1_i, f3_i, g1_i, g3_i

# Design matrix 
def D_M(c1,c3,pos_obs,los):
    n = len(los)

    A_e,B_e = [],[]
    for i in range(n-2):
        A = np.zeros((3,n))
        los1,los2,los3 = los[i:i+3]
        A[0,i],A[1,i],A[2,i] = c1[i]*los1
        A[0,i+1],A[1,i+1],A[2,i+1] = -los2
        A[0,i+2],A[1,i+2],A[2,i+2] = c3[i]*los3

        B = pos_obs[i+1] - c1[i]*pos_obs[i] - c3[i]*pos_obs[i+2]

        A_e.append(A)
        B_e.append(B)
        
    A_e = np.vstack(A_e) 
    B_e = np.hstack(B_e) 

    return A_e,B_e 