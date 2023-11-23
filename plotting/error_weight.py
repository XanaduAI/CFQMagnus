# This is an example code of how to analyze the cost of Magnus expansion
import math
import numpy as np
import matplotlib.pyplot as plt
import json

from plotting.magnus_errors import *

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
    x, y = map(float, number_readable.split('e+'))

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

# We first compute the error of a single step
range_s = [1,2,3,4]
range_m = [1,2,5,11]

def error_list(h, s, m, cs, cs_y, maxc = 1, maxp = 40, use_max = True, n = None):
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
    error = {}
    error_list = []

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
    error['Magnus_truncation'] = omega_error
    error_list.append(omega_error)


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
        error['exp_Magnus_Taylor'] = omega_error
        error_list.append(exp_omega_error)


        # Error from the Taylor expansion of the product of exponentials in the commutator-free operator
        error['Psi_m_Taylor'] = Psi_m_Taylor_error(h, maxp, s, m, cs[s][m], factorial, use_max = use_max or s > 4)
        error_list.append(error['Psi_m_Taylor'])

        # Error from the quadrature rule
        qr = quadrature_residual(h, s, m, maxc = maxc)
        error['Quadrature'] = quadrature_error(h, s, m, cs_y, maxc = maxc, qr = qr)
        error_list.append(error['Quadrature'])


        # Error from the basis change
        error['Basis'] = basis_change_error(h, s, m, cs_y, maxc)
        error_list.append(error['Basis'])

    else:
        error['exp_Magnus_Taylor'] = 0
        error['Psi_m_Taylor'] = 0
        error['Quadrature'] = 0
        error['Basis'] = 0
        error_list += [0,0,0,0]


    # Error from the Trotter product formula
    error['Trotter'] = trotter_error_spu_formula(n, h, s, u = maxc)
    error_list.append(error['Trotter'])

    suma = np.sum(error_list)

    result = np.array(error_list) / suma
    assert(np.abs(np.sum(result) - 1) < 1e-10)
    
    return result

labels = ['Magnus truncation', 'exp(Magnus) Taylor', '$\Psi_m$ Taylor', 'Quadrature', 'Basis change', 'Trotter']

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

########### Plot ###########
# Generate 4 plots, for different total errors
fig, ax = plt.subplots(2, 2, figsize = (10,10))

total_error_list = [1e-3, 1e-7, 1e-11, 1e-15]
colors = ['purple', 'r', 'yellow', 'g', 'b', 'black']
s = 1
m = 1
# Now we select is the total error
error_array = np.zeros((len(total_time_list), len(labels)))
for total_error, ax in zip(total_error_list, ax.flatten()):
    min_costs = []
    min_costs_h = []
    for count, total_time in enumerate(total_time_list):
        min_cost_h, min_cost = minimize_cost_CFMagnus(hs, s, m, total_time, total_error, step_error = step_error_cf, trotter_exponentials = True)
        error_array[count] = error_list(min_cost_h, s, m, cs, cs_y, n = total_time)
        #for j in range(len(error_lista)):
        #    ax.bar(total_time, error_lista[j], bottom=np.sum(error_lista[:j]), color=colors[j], label = labels[j])
    cummulative = np.zeros(len(error_array))
    for j in range(len(error_array[0])):
        y1 = error_array.transpose()[j] + cummulative
        ax.fill_between(total_time_list, y1, cummulative, color=colors[j], label = labels[j])
        cummulative = y1

    # set x label
    ax.set_xlabel(r'Total time $T$')
    ax.set_ylabel(r'Percentage error contribution')

    ax.set_title(f'Total error = {total_error}')

    ax.set_xscale('log')

    ax.legend()

fig.savefig(f'figures/error_contributions_s={s}_m={m}.pdf', bbox_inches='tight')