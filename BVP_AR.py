import gc
import time
import numpy as np
from lamberthub import izzo2015
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative
import matplotlib.pyplot as plt

from Constants_and_functions import mu

ACCEPTABLE_ERRORS = {"No feasible solution, try lower M!", "No feasible solution, try lower M", 
                     "Failed to converge", "Number of revolutions must be equal or greater than zero!"} 

def unitvec(nrads):
    ra, dec = nrads
    rhoh = np.array([[np.cos(dec)*np.cos(ra)],
                     [np.cos(dec)*np.sin(ra)],
                     [np.sin(dec)]])
    return rhoh

def rmse(est, true):
    est = np.array(est)
    true = np.array(true)
    N = len(est)
    diff = est - true
    rss_sum = 0.
    for ii in range(N):
        rss = np.dot(diff[ii], diff[ii])
        rss_sum += rss
    rmse = np.sqrt(rss_sum/N)
    return rmse 

def BVP(p0:list, args:tuple, method:str='BFGS', plot:tuple=(False,), Sol:tuple=None):
    '''
    Implementation of Siminski's BVP approach for tracklet-to-tracklet correlation.

    Parameters:
    p0 (list): Initial guess for the ranges.
    args (tuple): Tuple of arguments containing the following:
        k_range (np.ndarray): Array of feasible orbital revolutions.
        Attr (np.ndarray): Attributable vector [a1, d1, a2, d2].
        R (np.ndarray): Observer vector.
        R_dot (np.ndarray): Observer velocity vector.
        uvec (np.ndarray): Unit vectors of the line of sights.
        cov_z (np.ndarray): Covariance matrix of the measurements.
        cov_z_dot (np.ndarray): Covariance matrix of the rates.
        tof (float): Time of flight [s].
        epoch1 (astropy.Time): Middle epoch of the first tracklet.
        epoch2 (astropy.Time): Middle epoch of the second tracklet.
    method (str): Optimization method to use in scipy.minimize. Default is 'BFGS'.
    plot (tuple): Tuple of boolean and bounds (True, a_min, a_max, e_max) to plot topography and jacobian.
    Sol (tuple): Tuple of true solution vectors (np.hstack([r1, v1]), np.hstack([v2, v2])) to acquire error.
    '''
    start_time = time.time()
    (k_range, Attr, R, R_dot, uvec, cov_z, cov_z_dot, tof, epoch1, epoch2) = args 

    result = {'fun':  np.inf, 'x': [0, 0], 'nit': np.nan}
    k_res = np.nan
    res_path = []
    LOWP = None
    pr = None
    jac = None
    found_solution = False

    for prograde in [True, False]:
        for path in [True, False]:

            for k in k_range:
                evaluation_path_ = []
                def callback(xk):
                    if plot[0]:
                        evaluation_path_.append(np.copy(xk))
                    
                if method.lower() in ('bfgs', 'l-bfgs-b'):
                    options = {'ftol':1e-4, 'gtol':1e-7, 'disp': False, 'c1': 1e-5, 'c2': 1e-4, 'maxiter': 2000}
                    jac = '2-point'
                
                    if method.lower() == 'l-bfgs-b':
                        options = {'ftol': 1e-4, 'maxiter': 2000, 'gtol':1e-7, 'disp': False}
                                    
                else:
                    if method.lower() == 'nelder-mead':
                        options = {'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
                    else:
                        options = {'disp': False}

                opt_args = (k, tof, Attr, R, R_dot, uvec, cov_z, cov_z_dot, prograde, path)
                try:
                    result_ = minimize(Cost_fnc, p0, args=opt_args, method=method, jac=jac, options=options, callback=callback)

                    if (result_['fun'] < result['fun']) and np.isfinite(result_['fun']):
                        k_res = k
                        result = result_.copy()
                        LOWP = path
                        pr = prograde
                        found_solution = True
                        if plot[0]:
                            res_path = evaluation_path_

                        del result_
                        gc.collect()

                except Exception as e:
                    if str(e) in ACCEPTABLE_ERRORS:
                        continue
                    else:
                        print(f"Optimization error for k={k} (kmax:{k_range[-1]}), prograde={prograde}, path={path}: {e}")
                        continue

    comp_time = time.time() - start_time
    
    if not found_solution:
        print(f"No solution ({k % k_range[-1]}) ..")
                                                
    if plot[0]:
        a_min, a_max, e_max = plot[1:]
        args += (Sol,a_min,a_max,e_max, LOWP, comp_time, pr)
        plot_topography(p0, k_res, result, res_path, args)
        plot_jacobian(result, k_res, args, p0)
        plt.show()

    del k_range, Attr, R, R_dot, uvec, cov_z, cov_z_dot, tof, epoch1, epoch2
    gc.collect()
    
    return result, comp_time, k_res, LOWP, pr

def Cost_fnc(p:np.ndarray, k:float, tof:float, Attr:np.ndarray, R:np.ndarray, R_dot:np.ndarray, uvec:np.ndarray, cov_z:np.ndarray, 
             cov_zdot:np.ndarray, prograde:bool, low_path:bool): 

    # Extract attributable
    z = Attr[:2] + Attr[4:6]        #[a1, d1, a2, d2]
    zdot = Attr[2:4] + Attr[6:]     #[ad1, dd1, ad2, dd2]
    u1, u2 = uvec[:3], uvec[3:]

    los_vec = np.hstack((p[0] * u1, p[1] * u2))
    r = R + los_vec

    try: # Handle allowed errors lambert solver
        rdot_1, rdot_2 = izzo2015(mu, r[:3].flatten(), r[3:].flatten(), tof, M=k, prograde=prograde, low_path=low_path, 
                                  maxiter=35, atol=1e-5, rtol=1e-7)
    except Exception as e:
        exc_str = str(e) 
        if exc_str in ACCEPTABLE_ERRORS:  # Skip non-converging solution for k (or M) orbital revolutions.
            return np.nan #return nan to stop optimizing, but must be a better way to break optimizing process 
        else:
            print('Lambert solver raised a ValueError:', exc_str)    # Print all other value errors

    # Model attributable from orbit
    los_rate = np.hstack((rdot_1, rdot_2)) - R_dot
    zdot_hat = rates(los_vec, los_rate)

    # Transform covariance to modelled rates
    J = Jacobian(p, k, z, R, R_dot, tof, prograde, low_path) 
    cov_zdot_hat = np.dot(J, np.dot(cov_z, J.T))

    # Evaluate terms of cost function, and return
    delta_zdot = zdot - zdot_hat                                            #[4 x 1]
    sum_cov = cov_zdot + cov_zdot_hat                                       #[4 x 4]    
    L = np.dot(delta_zdot.T, np.dot(np.linalg.inv(sum_cov), delta_zdot))    #[1 x 1]
    return L   

def Jacobian(p, k, z, R, R_dot, tof, prograde, low_path):
    '''
    Jacobian function to map the least-squares fit angle rates with C_zdot toward C_zdot_hat.
    '''

    def z_hat(z):
        u = np.hstack((unitvec(z[:2]).flatten(), unitvec(z[2:]).flatten()))
        los = np.hstack((p[0] * u[:3], p[1] * u[3:]))
        r = R + los
        
        rdot1, rdot2 = izzo2015(mu, r[:3], r[3:], tof, M=k, prograde=prograde, low_path=low_path, maxiter=35, atol=1e-5, rtol=1e-7)
        losrate = np.hstack((rdot1, rdot2)) - R_dot
        
        return rates(los, losrate)

    J = approx_derivative(z_hat, z, method='3-point')
    return J

def rates(xyz, xyz_dot):
        radec_dot = np.zeros(4)

        for i in range(2):
            lw = 3 * i
            up = 3 * i + 3
            x, y, z = xyz[lw:up]
            xdot, ydot, zdot = xyz_dot[lw:up]

            xy_sq = x**2 + y**2
            xyz_sq = xy_sq + z**2

            ra_dot = (x * ydot - y * xdot) / xy_sq
            dec_dot = (-z * (x * xdot + y * ydot) + zdot * xy_sq) / (np.sqrt(xy_sq) * xyz_sq)
            radec_dot[i*2] = ra_dot
            radec_dot[i*2 + 1] = dec_dot

        return radec_dot #[alphadot1, deltadot1, alphadot2, deltadot2]

def range_bounds(a_min, a_max, e_max, R, R_dot, uvec):
    ''' Schumacher, P., Roscoe, C., and Wilkins, M., “Parallel Track Initiation for Optical Space Surveillance 
        Using Range and Range Rate Bounds”, in <i>Advanced Maui Optical and Space Surveillance Technologies 
        Conference</i>, 2013.
    '''
    R_dot_u1 = np.dot(R_dot[:3], uvec[:3])
    R_dot_u2 = np.dot(R_dot[3:], uvec[3:])
    Rsq1 = np.dot(R[:3], R[:3])
    Rsq2 = np.dot(R[3:], R[3:])

    with np.errstate(invalid='ignore'):
        rho_min1 = - R_dot_u1 + np.sqrt(R_dot_u1**2 + a_min**2 * (1 - e_max)**2 - Rsq1) 
        rho_max1 = - R_dot_u1 + np.sqrt(R_dot_u1**2 + a_max**2 * (1 + e_max)**2 - Rsq1) 

        rho_min2 = - R_dot_u2 + np.sqrt(R_dot_u2**2 + a_min**2 * (1 - e_max)**2 - Rsq2) 
        rho_max2 = - R_dot_u2 + np.sqrt(R_dot_u2**2 + a_max**2 * (1 + e_max)**2 - Rsq2)

    rho_bounds = [(rho_min1, rho_max1), (rho_min2, rho_max2)]
    return rho_bounds

def BVP_initial_val(Attr, tof, R, a_min, a_max):

    k_range = k_int(a_min, a_max, tof)

    # Unit vectors for line of sights
    u1 = unitvec(Attr[:2]).flatten()
    u2 = unitvec(Attr[4:6]).flatten()
    uvec = np.hstack((u1, u2))

    # Calculate R_dot_u1, R_dot_u2, Rsq1, and Rsq2
    R1, R2 = R[:3], R[3:]
    R_dot_u1 = np.dot(R1, u1)
    Rsq1 = np.dot(R1, R1)
    R_dot_u2 = np.dot(R2, u2)
    Rsq2 = np.dot(R2, R2)

    # Assume circular orbit and determine initial guess
    k0 = k_range[int(len(k_range)/4)]
    
    if k0 != 0:
        a0 = ((tof**2 * mu) / (4*np.pi**2 * k0**2))**(1/3)
    else:
        # a0 = (a_min + a_max) / 2
        a0 = a_min  # More stable for LEO and low alt. MEO

    with np.errstate(invalid='ignore'): # TODO: fix
        rho01 = - R_dot_u1 + np.sqrt(R_dot_u1**2 + a0**2 - Rsq1) 
        rho02 = - R_dot_u2 + np.sqrt(R_dot_u2**2 + a0**2 - Rsq2)        

    return [rho01, rho02], k_range, uvec

def plot_jacobian(result, k_res, args, p0_int, num_points=100):
    (k_range, Attr, R, R_dot, uvec, cov_z, cov_z_dot, tof, epoch1, epoch2, Sol, a_min, 
     a_max, e_max, low_path, comp_time, prograde) = args 

    # Extract true solution values
    x1_true, x2_true = Sol
    r1_true = x1_true[:3]
    r2_true = x2_true[:3]

    rho1_true = np.linalg.norm(r1_true - R[:3])
    rho2_true = np.linalg.norm(r2_true - R[3:])
    p0 = [rho1_true, rho2_true]
    p0_true = p0

    fig, axs = plt.subplots(len(p0), 1 , figsize=(6,5), sharex=True)
    labels = [r'($\rho_1$, $\rho_{2 sol}$) [m]', r'($\rho_{1 sol}$, $\rho_2$) [m]']
    for res in range(len(p0)):
        variable_range = np.linspace(p0[res]*0.1, p0[res]*2, num_points)
        jacobian_values = np.zeros(num_points)
        jacobian_values2p = np.zeros(num_points)
        Lvals = np.zeros(num_points)
        original_value = p0[res]
        for i, value in enumerate(variable_range):
            p0[res] = value
            jac = approx_derivative(Cost_fnc, p0, method='2-point', args=(k_res, tof, Attr, R, R_dot, uvec, cov_z, 
                                                                          cov_z_dot, prograde, low_path))
            jac_2p = approx_derivative(Cost_fnc, p0, method='3-point', args=(k_res, tof, Attr, R, R_dot, uvec, cov_z, 
                                                                             cov_z_dot, prograde, low_path))
            jacobian_values[i] = jac[res]
            jacobian_values2p[i] = jac_2p[res]
            Lvals[i] = Cost_fnc(p0, k_res, tof, Attr, R, R_dot, uvec, cov_z, cov_z_dot, prograde, low_path)
        p0[res] = original_value
        ax_left = axs[res]
        line1, = ax_left.plot(variable_range, (jacobian_values), color='r', label='fwd.')
        line2, = ax_left.plot(variable_range, (jacobian_values2p), color='b', linestyle='dashed', label='cent.')
        ax_left.set_ylabel(r'$\frac{\partial}{\partial \rho} \  L(\mathbf{p},k)$')
        ax_left.tick_params(axis='y')

        ax_right = ax_left.twinx()
        line3, = ax_right.plot(variable_range, Lvals, color='k', label=r'$L(\mathbf{p},k)$')
        ax_right.set_ylabel(r'$\log_{10} \ L(\mathbf{p}_s,k)$')
        ax_right.tick_params(axis='y')
        ax_right.set_yscale('log')

        ax_left.set_xlabel(labels[res])
        ax_left.set_ylim(-1, 1)

        vline1 = ax_left.axvline(x=p0_true[res], color='k', linestyle='dashed', label=r'Solution $\mathbf{p}_s$')
        vline2 = ax_left.axvline(x=p0_int[res], color='k', linestyle='dotted', label=r'$\mathbf{p}_0$')
        lines_left, labels_left = ax_left.get_legend_handles_labels()
        lines_right, labels_right = ax_right.get_legend_handles_labels()

    fig.legend(lines_left +lines_right , labels_left +labels_right +['Solution', 'Initial guess'], ncol=2, loc='upper left', 
               bbox_to_anchor=(0.12, 0.89))
    axs[0].set_title(r'Jacobian and Cost Function along $\rho_1$ and $\rho_2$')
    plt.subplots_adjust(hspace=0.35)

def plot_topography(p0, k_res, result, eval_points, args, save=False):
    import pandas as pd
    (k_range, Attr, R, R_dot, uvec, cov_z, cov_z_dot, tof, epoch1, epoch2, Sol, a_min, a_max, e_max, low_path, comp_time, 
     prograde) = args 

    if len(eval_points) != 0:
        rho1 = [rho[0] for rho in eval_points]
        rho2 = [rho[1] for rho in eval_points]
    else:
        rho1 = [p0[0]]
        rho2 = [p0[1]]

    R1, R2 = R[:3], R[3:]
    u1, u2 = uvec[:3], uvec[3:]   

    # Calculate R_dot_u1, R_dot_u2, Rsq1, and Rsq2
    R_dot_u1 = np.dot(R1, u1)
    Rsq1 = np.dot(R1, R1)
    R_dot_u2 = np.dot(R2, u2)
    Rsq2 = np.dot(R2, R2)

    with np.errstate(invalid='ignore'): 
        rho_min1 = - R_dot_u1 + np.sqrt(R_dot_u1**2 + a_min**2 * (1 - e_max)**2 - Rsq1) 
        rho_max1 = - R_dot_u1 + np.sqrt(R_dot_u1**2 + a_max**2 * (1 + e_max)**2 - Rsq1) 
        rho_min2 = - R_dot_u2 + np.sqrt(R_dot_u2**2 + a_min**2 * (1 - e_max)**2 - Rsq2) 
        rho_max2 = - R_dot_u2 + np.sqrt(R_dot_u2**2 + a_max**2 * (1 + e_max)**2 - Rsq2)

    if Sol is None:
        A1, B1 = rho_min1*0.9, rho_max1*1.05
        A2, B2 = rho_min2*0.9, rho_max2*1.05
    else:
        x1_true, x2_true = Sol
        r1_true = x1_true[:3]
        r2_true = x2_true[:3]

        rho1_true = np.linalg.norm(r1_true - R1)
        rho2_true = np.linalg.norm(r2_true - R2)

        margin_factor = 0.5
        A1 = rho1_true * (1 - margin_factor)
        B1 = rho1_true * (1 + margin_factor)
        A2 = rho2_true * (1 - margin_factor)
        B2 = rho2_true * (1 + margin_factor)

    x_range = B1 - A1
    y_range = B2 - A2
    max_range = max(x_range, y_range)
    x_mid = (A1 + B1) / 2
    y_mid = (A2 + B2) / 2
    A1 = x_mid - max_range / 2
    B1 = x_mid + max_range / 2
    A2 = y_mid - max_range / 2
    B2 = y_mid + max_range / 2

    X = np.linspace(A1, B1, 100)
    Y = np.linspace(A2, B2, 100)

    Z = np.zeros((len(X), len(Y)))
    for i in range(len(X)): 
        for j in range(len(Y)):
            Z[i, j] = (np.log10(Cost_fnc([X[i], Y[j]], k_res, tof, Attr, R, R_dot, uvec, cov_z, cov_z_dot, prograde, low_path)))

    fig, ax = plt.subplots()
    contour_filled = ax.contourf(X, Y, Z, levels=100, cmap="gray")

    cbar = fig.colorbar(contour_filled)
    cbar.ax.set_ylabel(r"$\log_{10}L(\mathbf{z}, k)$", rotation=-90, va="bottom")
    ax.plot(([p0[0]] + rho1), ([p0[1]] + rho2), 'r', linestyle='--', marker='o', markerfacecolor='none', markersize=4, 
            label='Evaluation Path', zorder=90)
    ax.scatter(p0[0], p0[1], color='r', marker='x', s=16, label='Start Point', zorder=91)

    if Sol != None:       
        r1_true = x1_true[:3]
        r2_true = x2_true[:3]
        rdot1_true = x1_true[3:]
        rdot2_true = x2_true[3:]
        rho1_true = np.linalg.norm(r1_true - R1)
        rho2_true = np.linalg.norm(r2_true - R2)

        ax.scatter(rho1_true, rho2_true, s=16, color='gold', marker=(6, 2, 0), label='Solution', zorder=100)
        los_vec = np.hstack((rho1[-1] * u1, rho2[-1] * u2))
        r = R + los_vec
        rdot_1, rdot_2 = izzo2015(mu, r[:3].flatten(), r[3:].flatten(), tof, M=k_res, prograde=prograde, low_path=low_path, 
                                  maxiter=35, atol=1e-5, rtol=1e-7)
        
        output = {
            "Metric": ["Epoch", "Obtained RHO [km]", "TRUE RHO [km]", "RMSE P [km]", "RMSE V [m/s]", "REL_ERR RHO [%]", 
                       "Time taken [s]:", "K (REV):", "PROGRADE:", "LOW PATH:", "LOG10 (obtained)", "LOG10 (true)"],
            "EPOCH 1": [epoch1.iso, f"{result['x'][0]*1e-3:.4f}", f"{rho1_true*1e-3:.4f}", f"{rmse(r[:3], r1_true)*1e-3:.4f}", 
                        f"{rmse(rdot_1, rdot1_true):.4f}", f"{100*(rho1[-1] - rho1_true)/rho1_true:.4f}", f"{comp_time:.4f}", 
                        f"{k_res:.4f}", f"{str(prograde)}", f"{str(low_path)}", f"{np.log10(result['fun']):.4f}", 
                        f"{np.log10(Cost_fnc([rho1_true, rho2_true], k_res, tof, Attr, R, R_dot, uvec, cov_z, cov_z_dot, prograde, low_path)):.4f}"],
            "EPOCH 2": [epoch2.iso, f"{result['x'][1]*1e-3:.4f}", f"{rho2_true*1e-3:.4f}", f"{rmse(r[3:], r2_true)*1e-3:.4f}", 
                        f"{rmse(rdot_2, rdot2_true):.4f}", f"{100*(rho2[-1] - rho2_true)/rho2_true:.4f}", "", "", "", "", "", ""]
        }

        output_df = pd.DataFrame(output)
        print(output_df)
        output_df.to_clipboard(index=False, sep='\t')

    ax.scatter(rho1[-1], rho2[-1], color='r', marker='d', facecolors='none', s=25, label='Obtained solution', zorder=99)
    plt.xlabel(r'$\rho_1$ [m]')
    plt.ylabel(r'$\rho_2$ [m]')
    A1, B1, A2, B2 = [None if np.isnan(x) or np.isinf(x) else x for x in [A1, B1, A2, B2] ]
    plt.xlim(A1, B1)
    plt.ylim(A2, B2)

    plt.legend(loc='upper left')
    plt.show()

def k_int(a_min, a_max, tof):
    '''
    Determine interval of orbital half-revolutions.
    '''
    twopi = 2*np.pi
    P_max = twopi * np.sqrt(a_max**3 / mu) 
    P_min = twopi * np.sqrt(a_min**3 / mu)

    k_min, k_max = (tof / P_max), (tof / P_min)
    step = 0.5
    k_range = np.arange((np.floor(k_min*2)/2), (np.floor(k_max*2)/2) + step, step)

    if len(k_range) == 0:
        k_range = np.array([0])

    return k_range