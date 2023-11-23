# This is an example code of how to analyze the cost of Magnus expansion
import math
from matplotlib import lines
import numpy as np
import matplotlib.pyplot as plt
import json

from magnus_errors import *

# First, we import the Magnus expansion coefficients
with open('cs.json', 'r') as f:
    cs = json.load(f, object_hook=convert_keys_to_float)

with open('cs_y.json', 'r') as f:
    cs_y = json.load(f, object_hook=convert_keys_to_float)

with open('cs_split.json', 'r') as f:
    cs_split = json.load(f, object_hook=convert_keys_to_float)

with open('cs_y_split.json', 'r') as f:
    cs_y_split = json.load(f, object_hook=convert_keys_to_float)

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

hs = [1/2**(i/3+3) for i in range(1,200)]
total_error_list = [1e-3, 1e-7, 1e-11, 1e-15]
total_time_list = [2**i for i in range(3, 15)]

########### Commutator Free Magnus ###########

def error_sum_CF(h, s, m, cs, cs_y, maxc = 1, maxp = 40, use_max = True, n = None):
    r"""
    h: step size
    n: number of spins
    s: 2s is the order of the Magnus expansion
    m: number of exponentials in the Commutator Free Magnus operator
    cs: list of the coefficients of the Magnus expansion in the basis of the Lie algebra
    cs_y: list of the coefficients of the Magnus expansion in the basis of univariate integrals
    maxc: maximum value of the norm of the Hamiltonian and its derivatives
    maxp: maximum order of the Magnus expansion
    use_max: if True, the maximum value of the norm of the Hamiltonian and its derivatives is used
    """
    error = np.longdouble(0)

    # First, we add the error from the Taylor truncation of Omega
    p = 2*s+1
    omega_error = Omega_bound(h, p, s, maxc)
    last_correction = omega_error
    while last_correction/omega_error > 1e-5:
        p += 1
        if p > maxp:
            raise ValueError('The error is not converging')
        last_correction = Omega_bound(h, p, s, maxc)
        omega_error += last_correction
    error += omega_error


    if s>1:
    # Error from the Taylor expansion of the exponential of the Magnus expansion

        p = 2*s+1
        exp_omega_error = exp_Omega_bound(h, p, s, maxc, factorial)
        last_correction = exp_omega_error
        while last_correction/exp_omega_error > 1e-5: #todo: change 1e-5
            p += 1
            if p > maxp:
                raise ValueError('The error is not converging')
            last_correction = exp_Omega_bound(h, p, s, maxc, factorial)
            exp_omega_error += last_correction
        error += exp_omega_error


        # Error from the Taylor expansion of the product of exponentials in the commutator-free operator
        error += Psi_m_Taylor_error(h, maxp, s, m, cs[s][m], factorial, use_max = use_max or s > 4)

        # Error from the quadrature rule
        qr = quadrature_residual(h, s, m, maxc = maxc)
        error += quadrature_error(h, s, m, cs_y, maxc = maxc, qr = qr)


        # Error from the basis change
        error += basis_change_error(h, s, m, cs_y, maxc)


    # Error from the Trotter product formula
    error += trotter_error_spu_formula(n, h, s, u = maxc)
    
    return error

def compute_step_error(hs, range_s, range_m, maxp, total_error_list, total_time_list, use_max = True, cs = cs, cs_y = cs_y):
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
    for total_error in tqdm(total_error_list, desc='total_error'):
        step_error[total_error] = {}
        for n in tqdm(total_time_list, desc = 'time'):  
            step_error[total_error][n] = {}
            for (s, m) in tqdm(zip(range_s, range_m), desc = 's, m'):
                if s not in step_error[total_error][n].keys():
                    step_error[total_error][n][s] = {}
                step_error[total_error][n][s][m] = {}
                for h in hs:
                    step_error[total_error][n][s][m][h] = float(error_sum_CF(h, s, m, cs, cs_y, maxc = 1, maxp = maxp, use_max = use_max, n = n))
    return step_error

# We first compute the error of a single step
#step_error = compute_step_error(hs, range_s, range_m, maxp = 50, total_error_list = total_error_list, total_time_list = total_time_list, use_max = True)

with open('results/step_error_CFMagnus.json', 'r') as f:
    step_error_cf = json.load(f, object_hook=convert_keys_to_float)

# Then we will first create a function to find the minimum cost
def minimize_cost_CFMagnus(hs, s, m, total_time, total_error, step_error, trotter_exponentials = True):
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
            cost_exponentials[h] *= 5**(s-1)
        #if split_operator and m == 3 and s== 2: 
        #    cost_exponentials[h] = 5**(s-1) * total_time/h # We can concatenate the first and last exponentials
        errors[h] = total_time*step_error[total_error][total_time][s][m][h]/h

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
    omega_error = Omega_bound(h, p, s, maxc)
    last_correction = omega_error
    while last_correction/omega_error > 1e-5:
        p += 1
        if p > maxp:
            raise ValueError('The error is not converging')
        last_correction = Omega_bound(h, p, s, maxc)
        omega_error += last_correction
    error += omega_error

    # Error from the Trotter product formula
    error += trotter_error_spu_formula(n, h, s, u = maxc)

    return error

def compute_trotter_step_error(hs, range_s, maxp, total_error_list, total_time_list, use_max = True):
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
    for total_error in tqdm(total_error_list, desc='total_error'):
        trotter_error[total_error] = {}
        for n in tqdm(total_time_list, desc = 'time'):  
            trotter_error[total_error][n] = {}
            for s in tqdm(range_s, desc = 's'):
                if s not in trotter_error[total_error][n].keys():
                    trotter_error[total_error][n][s] = {}
                for h in hs:
                    trotter_error[total_error][n][s][h] = float(error_sum_trotter(h, s, maxc = 1, maxp = maxp, n = n))
    return trotter_error

#range_s = [1,2,3,4]
#step_error_trotter = compute_trotter_step_error(hs, range_s, maxp = 50, total_error_list = total_error_list, total_time_list = total_time_list, use_max = True)

# json save step_error
#with open('results/step_error_trotter.json', 'w') as f:
#    json.dump(step_error_trotter, f)

with open('results/step_error_trotter.json', 'r') as f:
    step_error_trotter = json.load(f, object_hook=convert_keys_to_float)

# Then we will first create a function to find the minimum cost
def minimize_cost_trotter(hs, s, total_time, total_error, step_error, trotter_exponentials = True):
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
            cost_exponentials[h] *= 5**(s-1)
        errors[h] = total_time*step_error[total_error][total_time][s][h]/h

    min_cost = np.inf
    for h in hs:
        if errors[h] < total_error and cost_exponentials[h] < min_cost:
            min_cost_h = h
            min_cost = cost_exponentials[h]

    return min_cost_h, min_cost

########### Commutation-free Magnus split-operator ###########

def error_sum_CFsplit(h, s, m, cs_split, cs_y_split, maxc = 1, maxp = 40, use_max = True, n = None):
    r"""
    Computes the step error for a split-operator commutator-free Magnus expansion.

    Parameters
    ----------
    h: step size
    n: number of spins
    s: 2s is the order of the Magnus expansion
    m: number of exponentials in the Commutator Free Magnus operator
    cs_split: list of the coefficients of the Magnus expansion in the basis of the Lie algebra
    cs_y_split: list of the coefficients of the Magnus expansion in the basis of univariate integrals
    maxc: maximum value of the norm of the Hamiltonian and its derivatives
    maxp: maximum order of the Magnus expansion
    use_max: if True, the maximum value of the norm of the Hamiltonian and its derivatives is used
    """
    error = np.longdouble(0)

    # First, we add the error from the Taylor truncation of Omega
    p = 2*s+1
    omega_error = Omega_bound(h, p, s, maxc)
    last_correction = omega_error
    while last_correction/omega_error > 1e-5:
        p += 1
        if p > maxp:
            raise ValueError('The error is not converging')
        last_correction = Omega_bound(h, p, s, maxc)
        omega_error += last_correction
    error += omega_error

    p = 2*s+1
    exp_omega_error = exp_Omega_bound(h, p, s, maxc, factorial)
    last_correction = exp_omega_error
    while last_correction/exp_omega_error > 1e-5:
        p += 1
        if p > maxp:
            raise ValueError('The error is not converging')
        last_correction = exp_Omega_bound(h, p, s, maxc, factorial)
        exp_omega_error += last_correction
    error += exp_omega_error


    # Error from the Taylor expansion of the product of exponentials in the commutator-free operator
    error += Psi_m_Taylor_error(h, maxp, s, m, cs_split[s][m], factorial, use_max = use_max or s > 4)

    # Error from the quadrature rule
    qr = quadrature_residual(h, s, m, maxc = maxc)
    error += quadrature_error(h, s, m, cs_y_split, maxc = maxc, qr = qr)

    # Error from the basis change
    error += basis_change_error(h, s, m, cs_y_split, maxc)
    
    return error

def compute_step_error_split(hs, range_s, range_m, maxp, total_error_list, total_time_list, use_max = True, cs_split = cs_split, cs_y_split = cs_y_split):
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
    for total_error in tqdm(total_error_list, desc='total_error'):
        step_error[total_error] = {}
        for n in tqdm(total_time_list, desc = 'time'):  
            step_error[total_error][n] = {}
            for (s, m) in tqdm(zip(range_s, range_m), desc = 's, m'):
                if s not in step_error[total_error][n].keys():
                    step_error[total_error][n][s] = {}
                step_error[total_error][n][s][m] = {}
                for h in hs:
                    step_error[total_error][n][s][m][h] = float(error_sum_CF(h, s, m, cs_split, cs_y_split, maxc = 1, maxp = maxp, use_max = use_max, n = n))
    return step_error

# We first compute the error of a single step
#range_s = [2, 3]
#range_m = [12, 20]
#step_error_split = compute_step_error_split(hs, range_s, range_m, maxp = 50, total_error_list = total_error_list, total_time_list = total_time_list, use_max = True)
# json save step_error
#with open('results/step_error_CFMagnus_split.json', 'w') as f:
#    json.dump(step_error_split, f)

with open('results/step_error_CFMagnus_split.json', 'r') as f:
    step_error_split = json.load(f, object_hook=convert_keys_to_float)

# Then we will first create a function to find the minimum cost
def minimize_cost_CFMagnus_split(hs, s, m, total_time, total_error, step_error, trotter_exponentials = True):
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
            cost_exponentials[h] *= 5**(s-1)
        errors[h] = total_time*step_error[total_error][total_time][s][m][h]/h

    min_cost = np.inf
    for h in hs:
        if errors[h] < total_error and cost_exponentials[h] < min_cost:
            min_cost_h = h
            min_cost = cost_exponentials[h]

    return min_cost_h, min_cost

########### Plot ###########
# Generate 4 plots, for different total errors
fig, ax = plt.subplots(2, 2, figsize = (10,10))

total_error_list = [1e-3, 1e-7, 1e-11, 1e-15]
colors = ['r', 'g', 'b', 'black']

# Now we select is the total error
for total_error, ax in zip(total_error_list, ax.flatten()):
    range_s = [1,2,3,4]
    range_m = [1,2,5,11]
    for (s, m, c) in zip(range_s, range_m, colors):
        min_costs = []
        min_costs_h = []
        for total_time in total_time_list: #todo: Change the step error and minimization function here.
            min_cost_h, min_cost = minimize_cost_CFMagnus(hs, s, m, total_time, total_error, step_error = step_error_cf, trotter_exponentials = True)
            #min_cost_h, min_cost = minimize_cost_trotter(hs, s, total_time, total_error, step_error = step_error_trotter, trotter_exponentials = True)
            #min_cost_h, min_cost = minimize_cost_CFMagnus_split(hs, s, m, total_time, total_error, step_error = step_error_split, trotter_exponentials = True)
            min_costs.append(min_cost)
            min_costs_h.append(min_cost_h)

        # Implement a log log fit of min_cost vs total_time
        log_min_costs = np.log(np.array(min_costs))
        log_total_time_list = np.log(np.array(total_time_list))
        
        # Fit a line
        fit = np.polyfit(log_total_time_list, log_min_costs, 1)
        f1 = fit[1]
        f0 = fit[0]

        f1_formatted = convert_sci_to_readable('{:.2e}'.format(np.exp(f1)))
        label = f's={s}, m={m}, ${f1_formatted}\cdot T^{{{f0:.2f}}}$'
        ax.plot(total_time_list, min_costs, label = label, color = c)
        #ax.plot(total_time_list, min_costs, label = f's={s} m={m}', color = c)

    # set x label
    ax.set_xlabel(r'Total time $T$')
    ax.set_ylabel(r'Number of fast-forwardable exponentials')

    ax.set_title(f'Total error = {total_error}')

    # set logscale
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.legend()

    range_ss = [2, 3]
    range_ms = [12, 20]
    colors_s = ['g', 'b']
    # Now we select is the total error
    for (s, m, c) in zip(range_ss, range_ms, colors_s):
        min_costs = []
        min_costs_h = []
        for total_time in total_time_list: #todo: Change the step error and minimization function here.
            min_cost_h, min_cost = minimize_cost_CFMagnus_split(hs, s, m, total_time, total_error, step_error = step_error_split, trotter_exponentials = True)
            #min_cost_h, min_cost = minimize_cost_trotter(hs, s, total_time, total_error, step_error = step_error_trotter, trotter_exponentials = True)
            #min_cost_h, min_cost = minimize_cost_CFMagnus_split(hs, s, m, total_time, total_error, step_error = step_error_split, trotter_exponentials = True)
            min_costs.append(min_cost)
            min_costs_h.append(min_cost_h)

        # Implement a log log fit of min_cost vs total_time
        log_min_costs = np.log(np.array(min_costs))
        log_total_time_list = np.log(np.array(total_time_list))
        
        # Fit a line
        fit = np.polyfit(log_total_time_list, log_min_costs, 1)
        f1 = fit[1]
        f0 = fit[0]

        f1_formatted = convert_sci_to_readable('{:.2e}'.format(np.exp(f1)))
        label = f's={s}, $m_s={m}$, ${f1_formatted}\cdot T^{{{f0:.2f}}}$'
        ax.plot(total_time_list, min_costs, label = label, color = c, linestyle = '--')
        #ax.plot(total_time_list, min_costs, label = f's={s} m={m}', color = c)

    # set x label
    ax.set_xlabel(r'Total time $T$')
    ax.set_ylabel(r'Number of fast-forwardable exponentials')

    ax.set_title(f'Total error = ${convert_sci_to_readable('{:.1e}'.format(total_error))}$')

    # set logscale
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.legend()

#handles, labels = plt.gca().get_legend_handles_labels()
#by_label = dict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys())

#plt.show()
# save figure #todo: change name here
fig.savefig('commutator_free_magnus_error.pdf', bbox_inches='tight')