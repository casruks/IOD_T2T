import numpy as np

mu = 398600441800000.0  # [m3/s2], WGS84

def findc2c3(phi):
    
    if phi > 0:
        sqrt = np.sqrt(phi)
        c2 = (1 - np.cos(sqrt)) / phi
        c3 = (sqrt - np.sin(sqrt)) / sqrt**3

    elif phi < 0:
        sqrt = np.sqrt(-phi)
        c2 = (np.cosh(sqrt) - 1) / (-phi)
        c3 = (np.sinh(sqrt) - sqrt) / sqrt**3
        
    else:
        c2 = 1/2
        c3 = 1/6

    return c2, c3

def Lagrange_ceff_universal(a, x_i, rn, tau_i, GM):
    '''
    From (Curtis, 2013), 3rd edition, algorithm D.15 Equation 3.69.
    '''
    phi = a*x_i**2
    c2, c3 = findc2c3(phi)

    fi = 1 - x_i**2/rn*c2
    gi = tau_i - 1/np.sqrt(GM)*x_i**3*c3
    return fi, gi

def Kepler_U(dt, rn, v_rad, a, GM):
    err = 1e-8
    k_max = 500
    
    x = np.sqrt(GM)*abs(a)*dt

    k = 0
    ratio = 1

    while (abs(ratio) > err) and k < k_max:
        k += 1
        c2, c3 = findc2c3(a*x**2)

        F = rn*v_rad/np.sqrt(GM)*x**2*c2 + (1 - a*rn)*x**3*c3 + rn*x - np.sqrt(GM)*dt
        dFdx = rn*v_rad/np.sqrt(GM)*x*(1 - a*x**2*c3) + (1 - a*rn)*x**2*c2 + rn
        
        ratio = F/dFdx
        x -= ratio

    if k > k_max:
        print('Max iterations used for universal anomaly (k={k}).')
    return x