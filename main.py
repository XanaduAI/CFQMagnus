# This is an example code of how to analyze the cost of Magnus expansion
import math
import numpy as np
import matplotlib.pyplot as plt
import json

from magnus_errors import *

# First, we import the Magnus expansion coefficients
with open('cs.json', 'r') as f:
    cs = json.load(f, object_hook=convert_keys_to_float)

with open('cs_y.json', 'r') as f:
    cs_y = json.load(f, object_hook=convert_keys_to_float)

# Compute a dictionary with the value of the factorial
factorial = {}
for i in range(0, 50):
    factorial[i] = np.longdouble(math.factorial(i))

# We will import the functions from the file "Magnus_error.py"
# and create a single error evaluation function suitable for our purposes
def error_sum(h, s, m, cs, cs_y, maxc = 1, maxp = 40, use_max = True, n = None):
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
    bound_taylor_omega = []
    for p in range(0, maxp+1):
        bound_taylor_omega.append(Omega_bound(h, p, maxc, s))
    acc_bound_taylor_omega = accumulate_from(2*s+1,bound_taylor_omega, maxp)
    error += acc_bound_taylor_omega[maxp]

    if s>1:
    # Error from the Taylor expansion of the exponential of the Magnus expansion
        bound_taylor = []
        for p in range(0, maxp+1):
            bound_taylor.append(exp_Omega_bound(h, p, s, maxc, factorial))
        acc_bound_taylor = accumulate_from(2*s+1, bound_taylor, maxp)
        error += acc_bound_taylor[maxp]


        # Error from the Taylor expansion of the product of exponentials in the commutator-free operator
        psi_m_taylor_order_error = Psi_m_Taylor_error(h, maxp, s, m, cs[s][m], factorial, use_max = use_max or s > 4)
        acc_sum_compositions = accumulate_from(2*s+1, psi_m_taylor_order_error, maxp)
        error += acc_sum_compositions[maxp]


        # Error from the quadrature rule
        qr = quadrature_residual(h, s, m, maxc = maxc)
        error += quadrature_error(h, s, m, cs_y, maxc = maxc, qr = qr)


        # Error from the basis change
        error += basis_change_error(h, s, m, cs_y, maxc)


    # Error from the Trotter product formula
    error += trotter_error_spu_formula(n, h, s, u = maxc)
    
    return error


# Now we want to use this function to compare the cost of three Magnus operators.
# We first compute the error of a single step
hs = [1/2**(i/4+3) for i in range(1,400)]
total_error_list = [1e-3, 1e-7, 1e-11, 1e-15]
total_time_list = [2**i for i in range(3, 15)]
range_s = [1,2,3,4]
range_m = [1,2,5,11]

def compute_step_error(hs, range_s, range_m, maxp, total_error_list, total_time_list, use_max = True):
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
                    step_error[total_error][n][s][m][h] = float(error_sum(h, s, m, cs, cs_y, maxc = 1, maxp = maxp, use_max = use_max, n = n))
    return step_error

step_error = compute_step_error(hs, range_s, range_m, maxp = 30, total_error_list = total_error_list, total_time_list = total_time_list, use_max = True)

# json save step_error
with open('step_error_new.json', 'w') as f:
    json.dump(step_error, f)
with open('step_error_new.json', 'r') as f:
    step_error = json.load(f, object_hook=convert_keys_to_float)


# Then we will first create a function to find the minimum cost
def minimize_cost(hs, s, m, total_time, total_error, step_error, trotter_exponentials = True):
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


# Generate 4 plots, for different total errors
fig, ax = plt.subplots(2, 2, figsize = (10,10))

total_error_list = [1e-3, 1e-7, 1e-11, 1e-15]
colors = ['r', 'g', 'b', 'k']
# Now we select is the total error
for total_error, ax in zip(total_error_list, ax.flatten()):
    for (s, m, c) in zip(range_s, range_m, colors):
        min_costs = []
        min_costs_h = []
        for total_time in total_time_list:
            min_cost_h, min_cost = minimize_cost(hs, s, m, total_time, total_error, step_error, trotter_exponentials = True)
            min_costs.append(min_cost)
            min_costs_h.append(min_cost_h)
        ax.plot(total_time_list, min_costs, label = f's={s} m={m}', color = c)

    # set x label
    ax.set_xlabel(r'Total time $T$')
    ax.set_ylabel(r'Number of single exponentials')

    # set logscale
    ax.set_yscale('log')
    ax.set_xscale('log')

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.show()
fig.savefig('costs.pdf', bbox_inches='tight')