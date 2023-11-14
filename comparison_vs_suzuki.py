import math
from os import error
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
#step_error = compute_step_error(hs, range_s, range_m, maxp = 50, total_error_list = total_error_list, total_time_list = total_time_list, use_max = True)
# json save step_error
#with open('results/step_error_CFMagnus.json', 'w') as f:
#    json.dump(step_error, f)

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


########### Suzuki ###########

########### Plot ###########
# Generate 4 plots, for different total errors
fig, ax = plt.subplots(2, 2, figsize = (10,10))

total_error_list = [1e-3, 1e-7, 1e-11, 1e-15]
colors = ['r', 'g', 'b', 'k']
for total_error, ax in zip(total_error_list, ax.flatten()):

    for s, c in zip(range_s, colors):
        min_costs = []
        min_costs_h = []
        for total_time in total_time_list:
            min_costs_suzuki = np.inf
            min_cost_h_suzuki = None
            for h in hs:
                error_suzuki = suzuki_wiebe_error(h, s, total_time)
                cost_suzuki = suzuki_wiebe_cost(h, s, total_time, error_suzuki)
                if error_suzuki < total_error and cost_suzuki < min_costs_suzuki:
                    min_costs_suzuki = cost_suzuki
                    min_cost_h_suzuki = h

            min_costs.append(min_costs_suzuki)
            min_costs_h.append(min_cost_h_suzuki)

        ax.plot(total_time_list, min_costs, label = f's={s}', color = c, linestyle = '--')



    for (s, m, c) in zip(range_s, range_m, colors):
        min_costs = []
        min_costs_h = []
        for total_time in total_time_list:
            min_cost_h, min_cost = minimize_cost_CFMagnus(hs, s, m, total_time, total_error, step_error = step_error_cf, trotter_exponentials = True)
            min_costs.append(min_cost)
            min_costs_h.append(min_cost_h)
        ax.plot(total_time_list, min_costs, label = f's={s} m={m}', color = c)

    # set x label
    ax.set_xlabel(r'Total time $T$')
    ax.set_ylabel(r'Number of fast-forwardable exponentials')

    ax.set_title(f'Total error = {total_error}')

    # set logscale
    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.legend()

#handles, labels = plt.gca().get_legend_handles_labels()
#by_label = dict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys())

#plt.show()
# save figure #todo: change name here
fig.savefig('Magnus_vs_Suzuki.pdf', bbox_inches='tight')