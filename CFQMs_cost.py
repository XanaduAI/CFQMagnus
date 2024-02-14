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
from main import minimize_cost_CFMagnus_split, minimize_cost_CFMagnus, minimize_cost_trotter

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

range_s = [1, 2, 2, 3, 3]#, 4]
range_m = [1, 2, 3, 5, 6]#, 11]

with open('results/step_error_CFMagnus.json', 'r') as f:
    step_error_cf = json.load(f, object_hook=convert_keys_to_float)

range_ss = [2, 3]
range_ms = [12, 20]

with open('results/step_error_CFMagnus_split.json', 'r') as f:
    step_error_split = json.load(f, object_hook=convert_keys_to_float)

########### Plot ###########
# Generate 4 plots, for different total errors
with plt.style.context('science'):

    total_error_list = [1e-3, 1e-7, 1e-11, 1e-15]
    plot_letters = ['(a)', '(b)', '(c)', '(d)']

    colors = ['r', 'g', 'orange', 'b', 'purple', 'black']
    colors_s = ['g', 'b']

    # Now we select is the total error
    for total_error, plot_letter in zip(total_error_list, plot_letters):

        fig, ax = plt.subplots(1, 1, figsize = (4,4))

        lines_cfmagnus = []
        for (s, m, c) in zip(range_s, range_m, colors):
            min_costs = []
            min_costs_h = []
            for total_time in total_time_list: #todo: Change the step error and minimization function here.
                min_cost_h, min_cost = minimize_cost_CFMagnus(hs, s, m, total_time, total_error, step_error = step_error_cf, trotter_exponentials = True, splits = 2)
                #min_cost_h, min_cost = minimize_cost_trotter(hs, s, total_time, total_error, step_error = step_error_trotter, trotter_exponentials = True)
                #min_cost_h, min_cost = minimize_cost_CFMagnus_split(hs, s, m, total_time, total_error, step_error = step_error_split)
                min_costs.append(min_cost)
                min_costs_h.append(min_cost_h)

            # Implement a log log fit of min_cost vs total_time
            log_min_costs = np.log(np.array(min_costs))
            log_total_time_list = np.log(np.array(total_time_list))
            
            # Fit a line
            fit = np.polyfit(log_total_time_list[-3:], log_min_costs[-3:], 1)
            f1 = fit[1]
            f0 = fit[0]

            f1_formatted = convert_sci_to_readable('{:.2e}'.format(np.exp(f1)))
            label = f's={s}, m={m}'#, ${f1_formatted}\cdot T^{{{f0:.2f}}}$'
            line, = ax.plot(total_time_list, min_costs, label = label, color = c)
            #for i, txt in enumerate(min_costs_h):
            #    ax.annotate("{:.1e}".format(total_time_list[i]), (total_time_list[i], min_costs[i]))
            print(f'CFQM right, s={s}, $m={m}$, ${f1_formatted}\cdot T^{{{f0:.2f}}}$')

            fit = np.polyfit(log_total_time_list[:2], log_min_costs[:2], 1)
            f1 = fit[1]
            f0 = fit[0]

            f1_formatted = convert_sci_to_readable('{:.2e}'.format(np.exp(f1)))
            label = f's={s}, m={m}'#, ${f1_formatted}\cdot T^{{{f0:.2f}}}$'
            line, = ax.plot(total_time_list, min_costs, label = label, color = c)
            #for i, txt in enumerate(min_costs_h):
            #    ax.annotate("{:.1e}".format(total_time_list[i]), (total_time_list[i], min_costs[i]))
            #print(f'CFQM left, s={s}, $m={m}$, ${f1_formatted}\cdot T^{{{f0:.2f}}}$')

            #ax.plot(total_time_list, min_costs, label = f's={s} m={m}', color = c)
            lines_cfmagnus.append(line)

        legend = ax.legend(handles=lines_cfmagnus, loc = 'upper left', title = 'CF quasi-Magnus',frameon=True)
        ax.add_artist(legend)

        # set x label
        #ax.set_xlabel(r'Total time $T$')
        #ax.set_ylabel(r'Number of fast-forwardable exponentials')

        total_error_scientific = "{:e}".format(total_error)
        coef, exp = total_error_scientific.split("e")

        if coef == "1":
            title = f'Total error = 10^{{{int(exp)}}}'
        else:
            title = f'Total error = {coef} * 10^{{{int(exp)}}}'

        #ax.set_title(title)

        # set logscale
        ax.set_yscale('log')
        ax.set_xscale('log')

        lines_split = []


        # Now we select is the total error
        for (s, m, c) in zip(range_ss, range_ms, colors_s):
            min_costs = []
            min_costs_h = []
            for total_time in total_time_list: #todo: Change the step error and minimization function here.
                #min_cost_h, min_cost = minimize_cost_CFMagnus(hs, s, m, total_time, total_error, step_error = step_error_cf, trotter_exponentials = True)
                #min_cost_h, min_cost = minimize_cost_trotter(hs, s, total_time, total_error, step_error = step_error_trotter, trotter_exponentials = True)
                min_cost_h, min_cost = minimize_cost_CFMagnus_split(hs, s, m, total_time, total_error, step_error = step_error_split)
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
            label = f'$s={s}$, $m={m}$'#, ${f1_formatted}\cdot T^{{{f0:.2f}}}$'
            print(f'CFQMsplit s={s}, $m_s={m}$, ${f1_formatted}\cdot T^{{{f0:.2f}}}$')
            line, = ax.plot(total_time_list, min_costs, label = label, color = c, linestyle = '--')
            #ax.plot(total_time_list, min_costs, label = f's={s} m={m}', color = c)
            lines_split.append(line)

        legend2 = ax.legend(handles=lines_split, loc = 'lower right', title = 'CFQM split', frameon=True)
        ax.add_artist(legend2)

        # set x label
        ax.set_xlabel(r'Total time $T$', size=14)
        if plot_letter == '(a)':
            ax.set_ylabel(r'Exponentials', size=14)
        else:
            ax.set_ylabel(r'', size=14)

        total_error_scientific = "{:e}".format(total_error)
        coef, exp = total_error_scientific.split("e")

        if float(coef) == 1:
            number_str = f'$10^{{{int(exp)}}}$'
        else:
            number_str = f'${coef} \cdot 10^{{{int(exp)}}}$'

        ax.text(0.15, 0.03, f'{plot_letter} $\epsilon = $' + number_str , horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize=14, weight='bold')

        # set logscale
        ax.set_yscale('log')
        ax.set_xscale('log')

        # Cambiar la fuente de los xticks
        ax.tick_params(axis='x', labelsize = 14)

        # Cambiar la fuente de los yticks
        ax.tick_params(axis='y', labelsize = 14)


        fig.savefig(f'figures/commutator_free_magnus_{exp}.pdf', bbox_inches='tight')