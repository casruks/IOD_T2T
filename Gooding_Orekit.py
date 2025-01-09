import orekit
vm = orekit.initVM()

from orekit.pyhelpers import setup_orekit_curdir, download_orekit_data_curdir

setup_orekit_curdir()

from org.orekit.estimation.iod import IodGooding                # type: ignore (ignores pylance's incorrect warning message)
from org.hipparchus.geometry.euclidean.threed import Vector3D   # type: ignore
from org.orekit.frames import FramesFactory                     # type: ignore
from org.orekit.time import TimeScalesFactory, AbsoluteDate     # type: ignore

import time
import numpy as np
from astropy.time import Time

from Constants_and_functions import mu

def Gooding(los_vectors:np.ndarray, position_vectors:np.ndarray, times:list, rho_int:tuple, k:int, prograde:bool):
    """
    Estimate the orbit using Orekit's Gooding method.
    https://www.orekit.org/static/apidocs/org/orekit/estimation/iod/IodGooding.html 

    Parameters:
    los_vectors (np.ndarray)        : Line-of-sight vectors (3x3 array).
    position_vectors (np.ndarray)   : Position vectors (3x3 array).
    times (list)                    : List of three astropy Time objects.
    rho_int (tuple)                 : Initial guesses for rho1 and rho3.
    k (int)                         : Number of orbital revolutions.
    prograde (bool)                 : Prograde (True) or retrograde (False) orbit. 

    Returns:
    tuple: Estimated position and velocity vectors as numpy arrays.
    """
    start_time = time.time()
    assert isinstance(times, Time)

    gcrs_frame = FramesFactory.getGCRF()
    utc = TimeScalesFactory.getUTC()

    los1, los2, los3 = [Vector3D(float(los[0]), float(los[1]), float(los[2])) for los in los_vectors]
    R1, R2, R3 = [Vector3D(float(pos[0]), float(pos[1]), float(pos[2])) for pos in position_vectors]
    
    # Convert astropy Time objects to Orekit AbsoluteDate objects
    t1, t2, t3 = [AbsoluteDate(time.datetime.year, time.datetime.month, time.datetime.day,
                            time.datetime.hour, time.datetime.minute, time.datetime.second + time.datetime.microsecond * 1e-6, utc)
                for time in times]
    
    rho1_init, rho3_init = rho_int
    est_orbit = IodGooding(mu).estimate(gcrs_frame, R1, R2, R3, los1, t1, los2, t2, los3, t3, float(rho1_init), float(rho3_init), k, prograde)

    r_est = est_orbit.getPVCoordinates().getPosition()
    v_est = est_orbit.getPVCoordinates().getVelocity()

    r2_est_np = np.array([r_est.getX(), r_est.getY(), r_est.getZ()])
    v2_est_np = np.array([v_est.getX(), v_est.getY(), v_est.getZ()])

    comp_time = time.time() - start_time
    return r2_est_np, v2_est_np, comp_time

def Gooding_test_unit():
    gcrs_frame = FramesFactory.getGCRF()
    utc = TimeScalesFactory.getUTC()

    # Inputs
    mu = 1.0
    k = 0

    t1 = AbsoluteDate(2000, 1, 1, 0, 0, 0.0, utc)
    t2 = AbsoluteDate(t1, 0.325593, utc)  
    t3 = AbsoluteDate(t1, 0.701944, utc)   

    R1 = Vector3D(0.7000687, 0.6429399, 0.2789211)
    R2 = Vector3D(0.4306907, 0.8143496, 0.3532745)
    R3 = Vector3D(0.0628371, 0.9007098, 0.3907417)

    los1 = Vector3D(0.9028975, 0.0606048, 0.4255621)
    los2 = Vector3D(0.9224764, 0.0518570, 0.3825549)
    los3 = Vector3D(0.9347684, 0.0802269, 0.3460802)

    rho1_init = 2.399197226489 
    rho3_init = 2.824544883197
    # rho1_init = 5.9
    # rho3_init = 5.9

    # Expected results
    expected = {
        'Sol' : {
            'rho1': 2.399197226489,
            'rho2': 2.563703947213,
            'rho3': 2.824544883197,
            'eps' : 3.1e-15,
            'ecc' : 0.049,
            'N'   : 4
        }
    }

    # Run Gooding
    IODGooding = IodGooding(mu)
    est_orbit = IODGooding.estimate(gcrs_frame, R1, R2, R3, los1, t1, los2, t2, los3, t3, rho1_init, rho3_init, k, True)

    rho1 = IODGooding.getRange1()
    rho2 = IODGooding.getRange2()
    rho3 = IODGooding.getRange3()

    print(rho1, rho2, rho3)

    # Check results
    d1 = abs(rho1 - expected['Sol']['rho1'])
    d2 = abs(rho2 - expected['Sol']['rho2'])
    d3 = abs(rho3 - expected['Sol']['rho3'])

    print(d1)
    print(d2)
    print(d3)