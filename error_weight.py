# This is an example code of how to analyze the cost of Magnus expansion
import math
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import scienceplots

from magnus_errors import *

plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathrsfs}'


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
    x, y = map(float, number_readable.split('e+'))

    # Format as x * 10^y
    number_formatted = f"{x} \\cdot 10^{{{int(y)}}}"

    return number_formatted

# Compute a dictionary with the value of the factorial
factorial = {}
for i in range(0, 75):
    factorial[i] = np.longdouble(math.factorial(i))

hs = [1/2**(i/5+3) for i in range(1,125)]
total_error_list = [1e-3, 1e-7, 1e-11, 1e-15]
total_time_list = [int(2**(i/2)) for i in range(5, 41)]

# We first compute the error of a single step
range_s = [2,2,3,3]#,4]
range_m = [2,3,5,6]#,11]

def error_list(h, s, m, overline_xs, ys, step_error, maxc = 1, maxp = 40, use_max = True, n = None):
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
    error_l = []

    # First, we add the error from the Taylor truncation of Omega
    p = 2*s+1
    exp_omega_error = exp_Omega_bound(h, p, s, maxc, factorial)
    last_correction = exp_omega_error
    while last_correction/exp_omega_error > 1e-5: #todo: change 1e-5
        p += 1
        if p > maxp:
            raise ValueError('The error is not converging')
        last_correction = exp_Omega_bound(h, p, s, maxc, factorial)
        exp_omega_error += last_correction
    error['exp_Magnus_Taylor'] = exp_omega_error
    error_l.append(exp_omega_error)

    if s>1:
    # Error from the Taylor expansion of the exponential of the Magnus expansion


        # Error from the Taylor expansion of the product of exponentials in the commutator-free operator
        error['Psi_m_Taylor'] = Psi_m_Taylor_error(h, maxp, s, m, overline_xs[s][m], factorial, use_max = use_max or s > 4)
        error_l.append(error['Psi_m_Taylor'])

        # Error from the quadrature rule
        qr = quadrature_residual(h, s, maxc = maxc)
        error['Quadrature'] = quadrature_error(h, s, m, ys, maxc = maxc, qr = qr)
        error_l.append(error['Quadrature'])


    else:
        error['Psi_m_Taylor'] = 0
        error['Quadrature'] = 0
        error_l += [0,0]


    # Error from the Trotter product formula
    Z = np.sum(np.abs(zs[s][m]), axis = 1) / (4*n) if m > 1 else np.array([1/(4*n)])
    error['Trotter'] = np.sum([trotter_error_spu_formula(n, Z_*h, s, u = 1) for Z_ in Z])
    error_l.append(error['Trotter'])

    suma = np.sum(error_l)
    true_sum = step_error[n][s][m][h]
    assert(np.isclose(suma, true_sum))

    result = np.array(error_l) / suma
    assert(np.abs(np.sum(result) - 1) < 1e-10)
    
    return result

labels = ['$\exp(\Omega)$ Taylor', 'CFQM Taylor', 'Quadrature', 'Trotter']

with open('results/step_error_CFMagnus.json', 'r') as f:
    step_error_cf = json.load(f, object_hook=convert_keys_to_float)

# Then we will first create a function to find the minimum cost
########### Plot ###########
# Generate 4 plots, for different total errors
with plt.style.context('science'):

    total_error = 1e-3

    colors = ['b', 'g', 'yellow', 'r']
    s_list = [2,2,3,3]
    m_list = [2,3,5,6]
    ls = ['a', 'b', 'c', 'd']
    # Now we select is the total error
    for s, m, l in zip(s_list, m_list, ls):
        fig, ax = plt.subplots(1, 1, figsize = (3,3))

        error_array = np.zeros((len(total_time_list), len(labels)))
        min_costs = []
        min_costs_h = []
        for count, total_time in enumerate(total_time_list):
            #todo: change depending on whether we want n = fixed or n = total_time
            n = total_time
            min_cost_h, min_cost = minimize_cost_CFMagnus(hs, s, m, total_time, total_error, step_error = step_error_cf, n = n, trotter_exponentials = True)
            error_array[count] = error_list(min_cost_h, s, m, overline_xs, ys, step_error = step_error_cf, n = n)
            #for j in range(len(error_lista)):
            #    ax.bar(total_time, error_lista[j], bottom=np.sum(error_lista[:j]), color=colors[j], label = labels[j])
        cummulative = np.zeros(len(error_array))
        for j in range(len(error_array[0])):
            y1 = error_array.transpose()[j] + cummulative
            ax.fill_between(total_time_list, y1, cummulative, color=colors[j], label = labels[j], alpha =.2)
            cummulative = y1

        # set x label
        ax.set_xlabel(r'Total time $T$')
        if l == 'a':
            ax.set_ylabel(r'Error contribution')

        ax.text(0.27, 0.9, f'({l}) $s$ = {s}, $m$ = {m}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

        #ax.set_title(f'Total error = {total_error}')

        ax.set_xscale('log')

        if l == 'd':
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=True)

        fig.savefig(f'figures/error_contributions_s={s}_m={m}.pdf', bbox_inches='tight')