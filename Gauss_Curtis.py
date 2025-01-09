import time
import numpy as np
from Constants_and_functions import Kepler_U, Lagrange_ceff_universal

def Gauss_Curtis(t1, t2, t3, R1, R2, R3, rhohat1, rhohat2, rhohat3, GM, all=False):
    '''
    From (Curtis, 2013), 3rd edition, algorithm 5.5 & 5.6.

    \n \n
    (verified: \\02 Code\Testing_verification_val\IOD_algorithms_verif.py)
    '''
    start_time = time.time()

    tau1 = t1 - t2
    tau3 = t3 - t2

    tau = tau3 - tau1

    rhohat1 = np.array(rhohat1).flatten()
    rhohat2 = np.array(rhohat2).flatten()
    rhohat3 = np.array(rhohat3).flatten()

    p1 = np.cross(rhohat2, rhohat3)
    p2 = np.cross(rhohat1, rhohat3)
    p3 = np.cross(rhohat1, rhohat2)

    D0 = np.dot(rhohat1, p1)

    D11 = np.dot(R1, p1)
    D21 = np.dot(R2, p1)
    D31 = np.dot(R3, p1) 
    
    D12 = np.dot(R1, p2)
    D22 = np.dot(R2, p2)
    D32 = np.dot(R3, p2)
    
    D13 = np.dot(R1, p3)
    D23 = np.dot(R2, p3)
    D33 = np.dot(R3, p3)
    
    E = np.dot(R2, rhohat2) #v

    A = 1/D0*(-D12*tau3/tau + D22 + D32*tau1/tau)
    B = 1/6/D0*(D12*(tau3**2 - tau**2)*tau3/tau + D32*(tau**2 - tau1**2)*tau1/tau)

    a = -(A**2 + 2*A*E + np.linalg.norm(R2)**2)
    b = -2*GM*B*(A + E)
    c = -(GM* B)**2

    roots = np.roots([1, 0, a, 0, 0, b, 0, 0, c])
    realpos_roots =  np.real(roots[np.isreal(roots)&(roots > 0)])
    r2_root = realpos_roots[0]

    f1 = 1 - 1/2*GM*tau1**2/r2_root**3
    f3 = 1 - 1/2*GM*tau3**2/r2_root**3

    g1 = tau1 - 1/6*GM*(tau1/r2_root)**3
    g3 = tau3 - 1/6*GM*(tau3/r2_root)**3

    rho1 = 1/D0*((6*(D31*tau1/tau3 + D21*tau/tau3)*r2_root**3 + GM*D31*(tau**2 - tau1**2)*tau1/tau3) /(6*r2_root**3 + GM*(tau**2 - tau3**2)) - D11)
    rho2 = A + GM*B / r2_root**3
    rho3 = 1/D0*((6*(D13*tau3/tau1 - D23*tau/tau1)*r2_root**3 + GM*D13*(tau**2 - tau3**2)*tau3/tau1) /(6*r2_root**3 + GM*(tau**2 - tau1**2)) - D33)

    r1 = R1 + rho1 * rhohat1
    r2 = R2 + rho2 * rhohat2
    r3 = R3 + rho3 * rhohat3

    v2 = (-f3*r1 + f1*r3)/(f1*g3 - f3*g1) 
    
    #initial Gauss results
    r2_ini = r2 
    v2_ini = v2
    r1_ini, r3_ini = r1, r3

    ## Refinement
    rho1_old, rho2_old, rho3_old = rho1, rho2, rho3
    diff1, diff2, diff3 = 1, 1, 1
    k = 0
    k_max = 500
    tol = 1e-8
    while (diff1 > tol) and (diff2 > tol) and (diff3 > tol) and (k < k_max):
        k += 1

        # determine magnitudes and radial velocity component
        rn = np.linalg.norm(r2)
        # print(rn)
        vn = np.linalg.norm(v2)
        v_rad = np.dot(v2, r2) / rn
        a = 2/rn - vn**2 / GM

        # determine universal anomaly
        x1 = Kepler_U(tau1, rn, v_rad, a, GM)
        x3 = Kepler_U(tau3, rn, v_rad, a, GM)

        # determine Lagrange coefficients at tau1 tau3
        ff1, gg1 = Lagrange_ceff_universal(a, x1, rn, tau1, GM)
        ff3, gg3 = Lagrange_ceff_universal(a, x3, rn, tau3, GM)

        # update lagrange coefficients by average of old and new
        f1 = (f1 + ff1) / 2
        f3 = (f3 + ff3) / 2
        g1 = (g1 + gg1) / 2
        g3 = (g3 + gg3) / 2

        # determine coeffcients c1, c3
        den = f1*g3 - f3*g1
        c1 = g3 / den
        c3 = -g1 / den

        # determine new ranges
        rho1 = 1/D0*( -D11 + 1 / c1 * D21 - c3 / c1 * D31)
        rho2 = 1/D0*( -c1 * D12 + D22 - c3 * D32) 
        rho3 = 1/D0*(-c1 / c3 * D13 + 1 / c3 * D23 - D33)

        # determine new position vectors and mid-point velocity vector
        r1 = R1 + rho1*rhohat1
        r2 = R2 + rho2*rhohat2
        r3 = R3 + rho3*rhohat3
        v2 = (-f3 * r1 + f1 * r3) / den

        # quantify convergence
        diff1 = abs(rho1 - rho1_old)
        diff2 = abs(rho2 - rho2_old)
        diff3 = abs(rho3 - rho3_old)

        # update ranges for next iteration
        rho1_old, rho2_old, rho3_old = rho1, rho2, rho3

    if k > k_max:
        print('Gauss refinement executed with max iterations (k={k}).')

    comp_time = time.time() - start_time
    # print(f"Gauss, Time taken: {round(comp_time, 4)} s")
    if all == True:
        return r2_ini, v2_ini, r2, v2, r1, r3, r1_ini, r3_ini, comp_time
    else:
        return r2_ini, v2_ini, r2, v2, comp_time