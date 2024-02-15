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

# First, we import the Magnus expansion coefficients
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
    number_formatted = f"{x} \cdot 10^{{{int(y)}}}"

    return number_formatted

# Example usage
#result = convert_sci_to_readable(1.23e+05)
#print(result)



# Compute a dictionary with the value of the factorial
factorial = {}
for i in range(0, 75):
    factorial[i] = np.longdouble(math.factorial(i))

hs = [1/2**(i/5+3) for i in range(1,250)]
total_error_list = [1e-3, 1e-7, 1e-11, 1e-15]
total_time_list = [int(2**(i/2)) for i in range(5, 41)]


# We first compute the error of a single step
range_s = [1,2,3]#,4]
range_m = [1,2,5]#,11]

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

########### Plot ###########
# Generate 4 plots, for different total errors
with plt.style.context('science'):

    total_error = 1e-3

    plot_letters = ['(a)', '(b)', '(c)', '(d)']

    colors = ['r', 'g', 'b']#, 'k']

    for total_error, plot_letter in zip(total_error_list, plot_letters):
        fig, ax = plt.subplots(1, 1, figsize = (4,4))


        ########### Suzuki ###########

        lines_suzuki = []  # List to store line instances for Suzuki
        #linestyles = [(0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), 
        #            (0, (3, 5, 1, 5, 1, 5)), (0, (3, 1, 1, 1, 1, 1))]
        linestyles = ['--']*3
        for s, c, style in zip(range_s, colors, linestyles):
            min_costs = []
            min_costs_h = []
            for total_time in total_time_list:
                min_costs_suzuki = np.inf
                min_cost_h_suzuki = None
                for h in hs:
                    r = total_time/h
                    error_suzuki = total_error / r
                    cost_suzuki = suzuki_wiebe_cost(h, s, total_time, error_suzuki)
                    if error_suzuki < total_error and cost_suzuki < min_costs_suzuki:
                        min_costs_suzuki = cost_suzuki
                        min_cost_h_suzuki = h

                min_costs.append(min_costs_suzuki)
                min_costs_h.append(min_cost_h_suzuki)

            # Implement a log log fit of min_cost vs total_time
            log_min_costs = np.log(np.array(min_costs))
            log_total_time_list = np.log(np.array(total_time_list))
            
            # Fit a line
            fit = np.polyfit(log_total_time_list, log_min_costs, 1)
            f1 = fit[1]
            f0 = fit[0]

            f1_formatted = convert_sci_to_readable('{:.2e}'.format(np.exp(f1)))
            label = f's={s}'#, ${f1_formatted}\cdot T^{{{f0:.2f}}}$'
            print(f'Suzuki, s={s}, ${f1_formatted}\cdot T^{{{f0:.2f}}}$')
            line, = ax.plot(total_time_list, min_costs, label = label, color = c, linestyle = style)
            lines_suzuki.append(line)

        legend1 = ax.legend(handles=lines_suzuki, loc = 'upper left', title = 'Suzuki',frameon=True)
        ax.add_artist(legend1)

        ########### CF Magnus ###########

        lines_magnus = []  # List to store line instances for CF Magnus
        linestyles = ['-']*3 #['-', (0, (5, 1)), '--', (0, (5, 10))]
        for (s, m, c, style) in zip(range_s, range_m, colors, linestyles):
            min_costs = []
            min_costs_h = []
            for total_time in total_time_list:
                min_cost_h, min_cost = minimize_cost_CFMagnus(hs, s, m, total_time, total_error, step_error = step_error_cf, trotter_exponentials = True)
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
            label = f's={s} m={m}'#, ${f1_formatted}\cdot T^{{{f0:.2f}}}$' #todo: take
            print(f'CFQM, s={s}, $m={m}$, ${f1_formatted}\cdot T^{{{f0:.2f}}}$')
            line, = ax.plot(total_time_list, min_costs, label = label, color = c, linestyle = style)
            #for i, txt in enumerate(min_costs_h):
            #    ax.annotate("{:.0e}".format(txt), (total_time_list[i], min_costs[i]))
            lines_magnus.append(line)

        legend2 = ax.legend(handles=lines_magnus, loc = 'lower right', title = 'CF quasi-Magnus',frameon=True)
        ax.add_artist(legend2)


        # set x label
        ax.set_xlabel(r'Total time $T$', fontsize = 14)
        if plot_letter == '(a)':
            ax.set_ylabel(r'Exponentials', fontsize = 14)

        total_error_scientific = "{:e}".format(total_error)
        coef, exp = total_error_scientific.split("e")

        if float(coef) == 1:
            number_str = f'$10^{{{int(exp)}}}$'
        else:
            number_str = f'${coef} \cdot 10^{{{int(exp)}}}$'

        ax.text(0.3, 0.9, f'{plot_letter} $\epsilon = $' + number_str , horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize=14, weight='bold')

        # set logscale
        ax.set_yscale('log')
        ax.set_xscale('log')

        # Cambiar la fuente de los xticks
        ax.tick_params(axis='x', labelsize = 14)

        # Cambiar la fuente de los yticks
        ax.tick_params(axis='y', labelsize = 14)

        fig.savefig(f'figures/Magnus_vs_Suzuki_{exp}.pdf', bbox_inches='tight')