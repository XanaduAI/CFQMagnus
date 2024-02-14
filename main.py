# This is an example code of how to analyze the cost of Magnus expansion
import math
from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt
import json
import os

import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

from magnus_errors import *

# Get current directory
dir_path = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(dir_path, 'coefficients')

# First, we import the Magnus expansion coefficients
with open(os.path.join(save_path, 'xs.json'), 'r') as f:
    xs = json.load(f, object_hook=convert_keys_to_float)

with open(os.path.join(save_path,'ys.json'), 'r') as f:
    ys = json.load(f, object_hook=convert_keys_to_float)

with open(os.path.join(save_path,'zs.json'), 'r') as f:
    zs = json.load(f, object_hook=convert_keys_to_float)

with open(os.path.join(save_path,'overline_xs.json'), 'r') as f:
    overline_xs = json.load(f, object_hook=convert_keys_to_float)

with open(os.path.join(save_path,'xs_split.json'), 'r') as f:
    xs_split = json.load(f, object_hook=convert_keys_to_float)

with open(os.path.join(save_path, 'ys_split.json'), 'r') as f:
    ys_split = json.load(f, object_hook=convert_keys_to_float)

with open(os.path.join(save_path, 'zs_split.json'), 'r') as f:
    zs_split = json.load(f, object_hook=convert_keys_to_float)

with open(os.path.join(save_path, 'overline_xs_split.json'), 'r') as f:
    overline_xs_split = json.load(f, object_hook=convert_keys_to_float)

def convert_sci_to_readable(number_sci):
    # Convert to readable format
    number_readable = number_sci

    # Split the number and exponent parts
    x, y = map(float, number_readable.split('e'))

    # Format as x * 10^y
    number_formatted = f"{x} \\cdot 10^{{{int(y)}}}"

    return number_formatted

# Compute a dictionary with the value of the factorial
factorial = {}
for i in range(0, 75):
    factorial[i] = np.longdouble(math.factorial(i))

hs = [1/2**(i/5+3) for i in range(1,250)]
total_error_list = [1e-3, 1e-7, 1e-11, 1e-15]
total_time_list = [int(2**(i/2)) for i in range(5, 41)]

########### Commutator Free Magnus ###########
range_s = [1, 2, 2, 3, 3]#, 4]
range_m = [1, 2, 3, 5, 6]#, 11]

def error_sum_CF_wout_trotter(h, s, m, overline_xs, ys, maxc = 1, maxp = 40, use_max = True):
    r"""
    h: step size
    n: number of spins
    s: 2s is the order of the Magnus expansion
    m: number of exponentials in the Commutator Free Magnus operator
    overline_xs: list of the coefficients of the Magnus expansion in the basis of the Lie algebra
    ys: list of the coefficients of the Magnus expansion in the basis of univariate integrals
    maxc: maximum value of the norm of the Hamiltonian and its derivatives
    maxp: maximum order of the Magnus expansion
    use_max: if True, the maximum value of the norm of the Hamiltonian and its derivatives is used
    """
    error = np.longdouble(0)

    # Error from the Taylor expansion of the exponential of the Magnus expansion

    p = 2*s+1
    exp_omega_error = exp_Omega_bound(h, p, s, maxc, factorial)
    last_correction = exp_omega_error
    while last_correction/exp_omega_error > 1e-5:
        p += 2
        if p > maxp:
            raise ValueError('The error is not converging')
        last_correction = exp_Omega_bound(h, p, s, maxc, factorial)
        exp_omega_error += last_correction
    error += exp_omega_error

    if s>1:

        # Error from the Taylor expansion of the product of exponentials in the commutator-free operator
        error += Psi_m_Taylor_error(h, maxp, s, m, overline_xs[s][m] * maxc, factorial, use_max = use_max or s > 4)

        # Error from the quadrature rule
        qr = quadrature_residual(h, s, maxc = maxc)
        error += quadrature_error(h, s, m, ys, maxc = maxc, qr = qr)


    # Error from the Trotter product formula
    # error += trotter_error_spu_formula(n, h/Z, s, u = maxc) * m
    
    return error

def compute_step_error_cf(hs, range_s, range_m, maxp, total_time_list, use_max = True, overline_xs = overline_xs, ys = ys, zs = zs):
    r"""
    Computes a dictionary of errors for different values of the step size, the order of the Magnus expansion, and the number of exponentials in the Commutator Free Magnus operator.

    Parameters
    ----------
    hs: list of step sizes
    range_s: list of values of s to consider
    range_m: list of values of m to consider
    maxp: maximum order of the Magnus expansion
    total_error_list: list of total errors to consider
    total_time_list: list of total times to consider
    use_max: if True, the maximum value of the norm of the Hamiltonian and its derivatives is used
    """
    step_error = {}
    step_error_wout_trotter = {}
    for (s, m) in tqdm(zip(range_s, range_m), desc = 's, m'):
        if s not in step_error_wout_trotter.keys():
            step_error_wout_trotter[s] = {}
        step_error_wout_trotter[s][m] = {}
        for h in hs:
            step_error_wout_trotter[s][m][h] = float(error_sum_CF_wout_trotter(h, s, m, overline_xs, ys, maxc = 1, maxp = maxp, use_max = use_max))
    for n in tqdm(total_time_list, desc = 'time'):  
        step_error[n] = {}
        for (s, m) in tqdm(zip(range_s, range_m), desc = 's, m'):
            Z = np.max(np.sum(np.abs(zs[s][m]), axis = 1)) * 4*n if s > 1 else 4*n
            if s not in step_error[n].keys():
                step_error[n][s] = {}
            step_error[n][s][m] = {}
            for h in hs:
                step_error[n][s][m][h] = float(step_error_wout_trotter[s][m][h] + 
                                        trotter_error_spu_formula(n, h/Z, s, u = 1) * m) # m exponentials to be Trotterized
    return step_error

# We first compute the error of a single step
step_error_cf = compute_step_error_cf(hs, range_s, range_m, maxp = 50, total_time_list=total_time_list, use_max = True)

with open('results/step_error_CFMagnus.json', 'w') as f:
    json.dump(step_error_cf, f)

with open('results/step_error_CFMagnus.json', 'r') as f:
    step_error_cf = json.load(f, object_hook=convert_keys_to_float)

# Then we will first create a function to find the minimum cost
def minimize_cost_CFMagnus(hs, s, m, total_time, total_error, step_error, trotter_exponentials = True, splits = 2):
    r"""
    Finds the step size that minimizes the cost of a Magnus expansion.

    Parameters
    ----------
    hs: list of step sizes
    s: 2s is the order of the Magnus expansion
    m: number of exponentials in the Commutator Free Magnus operator
    total_time: total time of the simulation
    total_error: total error of the simulation
    step_error: dictionary of errors for different values of the step size, the order of the Magnus expansion, and the number of exponentials in the Commutator Free Magnus operator.
    """

    cost_exponentials = {}
    errors = {}
    for h in hs:
        cost_exponentials[h] = total_time*m/h
        if trotter_exponentials: 
            cost_exponentials[h] *= 2 * 5**(s-1) * splits
        errors[h] = total_time*step_error[total_time][s][m][h]/h

    min_cost = np.inf
    for h in hs:
        if errors[h] < total_error and cost_exponentials[h] < min_cost:
            min_cost_h = h
            min_cost = cost_exponentials[h]

    return min_cost_h, min_cost

########### Trotter product formula ###########

def error_sum_trotter(h, s, maxc = 1, maxp = 40, n = None):
    r"""
    Error from using Trotter product formulas of arbitrary order, without the corresponding Magnus expansion.

    Parameters
    ----------
    h: step size
    s: 2s is the order of the Trotter product formula
    maxc: maximum value of the norm of the Hamiltonian
    n: number of spins
    """
    error = np.longdouble(0)

    # First, we add the error from the Taylor truncation of Omega
    s0 = 1
    p = 2*s0+1 # We are truncating the Magnus expansion at order 2s+1
    omega_error = exp_Omega_bound(h, p, s, maxc, factorial)
    last_correction = omega_error
    while last_correction/omega_error > 1e-5:
        p += 2
        if p > maxp:
            raise ValueError('The error is not converging')
        last_correction = exp_Omega_bound(h, p, s, maxc, factorial)
        omega_error += last_correction
    error += omega_error

    # Error from the Trotter product formula
    error += trotter_error_spu_formula(n, h/(4*n), s, u = maxc)

    return error

def compute_trotter_step_error(hs, range_s, maxp, total_time_list, use_max = True):
    r"""
    Computes a dictionary of errors for different values of the step size, the order of the Magnus expansion, and the number of exponentials in the Commutator Free Magnus operator.

    Parameters
    ----------
    hs: list of step sizes
    range_s: list of values of s to consider
    range_m: list of values of m to consider
    maxp: maximum order of the Magnus expansion
    total_error_list: list of total errors to consider
    total_time_list: list of total times to consider
    use_max: if True, the maximum value of the norm of the Hamiltonian and its derivatives is used
    """
    trotter_error = {}
    for n in tqdm(total_time_list, desc = 'time'):  
        trotter_error[n] = {}
        for s in tqdm(range_s, desc = 's'):
            if s not in trotter_error[n].keys():
                trotter_error[n][s] = {}
            for h in hs:
                trotter_error[n][s][h] = float(error_sum_trotter(h, s, maxc = 1, maxp = maxp, n = n))
    return trotter_error

range_s_trotter = [1,2,3,4]
step_error_trotter = compute_trotter_step_error(hs, range_s_trotter, maxp = 50, total_time_list = total_time_list, use_max = True)

# json save step_error
with open('results/step_error_trotter.json', 'w') as f:
    json.dump(step_error_trotter, f)

with open('results/step_error_trotter.json', 'r') as f:
    step_error_trotter = json.load(f, object_hook=convert_keys_to_float)

# Then we will first create a function to find the minimum cost
def minimize_cost_trotter(hs, s, total_time, total_error, step_error, trotter_exponentials = True, splits = 2):
    r"""
    Finds the step size that minimizes the cost of a Magnus expansion.

    Parameters
    ----------
    hs: list of step sizes
    s: 2s is the order of the Magnus expansion
    m: number of exponentials in the Commutator Free Magnus operator
    total_time: total time of the simulation
    total_error: total error of the simulation
    step_error: dictionary of errors for different values of the step size, the order of the Magnus expansion, and the number of exponentials in the Commutator Free Magnus operator.
    """

    cost_exponentials = {}
    errors = {}
    for h in hs:
        m = 1
        cost_exponentials[h] = total_time*m/h
        if trotter_exponentials: 
            cost_exponentials[h] *= 2 * 5**(s-1) * splits
        errors[h] = total_time*step_error[total_error][total_time][s][h]/h

    min_cost = np.inf
    for h in hs:
        if errors[h] < total_error and cost_exponentials[h] < min_cost:
            min_cost_h = h
            min_cost = cost_exponentials[h]

    return min_cost_h, min_cost

########### Commutation-free Magnus split-operator ###########

def error_sum_CFsplit(h, s, m, overline_xs_split, ys_split, maxc = 1, maxp = 40, use_max = True):
    r"""
    Computes the step error for a split-operator commutator-free Magnus expansion.

    Parameters
    ----------
    h: step size
    n: number of spins
    s: 2s is the order of the Magnus expansion
    m: number of exponentials in the Commutator Free Magnus operator
    overline_xs_split: list of the coefficients of the Magnus expansion in the basis of the Lie algebra
    ys_split: list of the coefficients of the Magnus expansion in the basis of univariate integrals
    maxc: maximum value of the norm of the Hamiltonian and its derivatives
    maxp: maximum order of the Magnus expansion
    use_max: if True, the maximum value of the norm of the Hamiltonian and its derivatives is used
    """
    error = np.longdouble(0)

    # First, we add the error from the Taylor truncation of Omega
    p = 2*s+1
    exp_omega_error = exp_Omega_bound(h, p, s, maxc, factorial)
    last_correction = exp_omega_error
    while last_correction/exp_omega_error > 1e-5:
        p += 2
        if p > maxp:
            raise ValueError('The error is not converging')
        last_correction = exp_Omega_bound(h, p, s, maxc, factorial)
        exp_omega_error += last_correction
    error += exp_omega_error

    # Error from the Taylor expansion of the product of exponentials in the commutator-free operator
    error += Psi_m_Taylor_error(h, maxp, s, m, overline_xs_split[s][m], factorial, use_max = use_max or s > 4)

    # Error from the quadrature rule
    qr = quadrature_residual(h, s, maxc = maxc)
    error += quadrature_error(h, s, m, ys_split, maxc = maxc, qr = qr)
    
    return error

def compute_step_error_split(hs, range_s, range_m, maxp, use_max = True, overline_xs_split = overline_xs_split, ys_split = ys_split):
    r"""
    Computes a dictionary of errors for different values of the step size, the order of the Magnus expansion, and the number of exponentials in the Commutator Free Magnus operator.

    Parameters
    ----------
    hs: list of step sizes
    range_s: list of values of s to consider
    range_m: list of values of m to consider
    maxp: maximum order of the Magnus expansion
    total_error_list: list of total errors to consider
    total_time_list: list of total times to consider
    use_max: if True, the maximum value of the norm of the Hamiltonian and its derivatives is used
    """
    step_error = {}
    for (s, m) in tqdm(zip(range_s, range_m), desc = 's, m'):
        if s not in step_error.keys():
            step_error[s] = {}
        step_error[s][m] = {}
        for h in hs:
            step_error[s][m][h] = float(error_sum_CFsplit(h, s, m, overline_xs_split, ys_split, maxc = 1, maxp = maxp, use_max = use_max))
    return step_error

# We first compute the error of a single step
range_ss = [2, 3]
range_ms = [12, 20]

step_error_split = compute_step_error_split(hs, range_ss, range_ms, maxp = 50, use_max = True)
# json save step_error
with open('results/step_error_CFMagnus_split.json', 'w') as f:
    json.dump(step_error_split, f)

with open('results/step_error_CFMagnus_split.json', 'r') as f:
    step_error_split = json.load(f, object_hook=convert_keys_to_float)

# Then we will first create a function to find the minimum cost
def minimize_cost_CFMagnus_split(hs, s, m, total_time, total_error, step_error):
    r"""
    Finds the step size that minimizes the cost of a Magnus expansion.

    Parameters
    ----------
    hs: list of step sizes
    s: 2s is the order of the Magnus expansion
    m: number of exponentials in the Commutator Free Magnus operator
    total_time: total time of the simulation
    total_error: total error of the simulation
    step_error: dictionary of errors for different values of the step size, the order of the Magnus expansion, and the number of exponentials in the Commutator Free Magnus operator.
    """

    cost_exponentials = {}
    errors = {}
    for h in step_error[s][m].keys():
        cost_exponentials[h] = total_time*m/h
        #if trotter_exponentials: # No need to Trotterize the exponentials in the split-operator case
        #    cost_exponentials[h] *= 2 * 5**(s-1)
        errors[h] = total_time*step_error[s][m][h]/h

    min_cost = np.inf
    for h in step_error[s][m].keys():
        if errors[h] < total_error and cost_exponentials[h] < min_cost:
            min_cost_h = h
            min_cost = cost_exponentials[h]

    return min_cost_h, min_cost

