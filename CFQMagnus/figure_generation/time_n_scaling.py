# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



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

from CFQMagnus.magnus_errors import *

# Get current directory

dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
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
total_time_list = [int(2**(i/2)) for i in range(5, 29)]

########### Commutator Free Magnus ###########
range_s = [1, 2, 2, 3, 3]#, 4]
range_m = [1, 2, 3, 5, 6]#, 11]

with open('results/step_error_CFMagnus.json', 'r') as f:
    step_error_cf = json.load(f, object_hook=convert_keys_to_float)

# Then we will first create a function to find the minimum cost

########### Commutation-free Magnus split-operator ###########

# We first compute the error of a single step
range_ss = [2, 3]
range_ms = [12, 20]

with open('results/step_error_CFMagnus_split.json', 'r') as f:
    step_error_split = json.load(f, object_hook=convert_keys_to_float)

# Then we will first create a function to find the minimum cost
########### Plot ###########
# Generate 4 plots, for different total errors

if not os.path.exists("CFQMagnus/figures/"):
    os.mkdir("CFQMagnus/figures/")

with plt.style.context('science'):

    total_error = 1e-3
    plot_letters = ['(a)', '(b)']

    colors_s = ['g', 'b']

    range_s = [2, 3]

    range_m2 = [2, 3]
    range_ms2 = [12]

    range_m3 = [5, 6]
    range_ms3 = [20]

    colors = ['b', 'y', 'g', 'r']

    for s, range_m, range_ms, letter in zip(range_s, 
                                            [range_m2, range_m3],
                                            [range_ms2, range_ms3],
                                            plot_letters):

        fig, ax = plt.subplots(1, 1, figsize = (4,4))

        # CFQM
        for (m, c) in zip(range_m, colors):
            min_costs = []
            min_costs_h = []
            for total_time in total_time_list:
                min_cost_h, min_cost = minimize_cost_CFMagnus(hs, s, m, total_time, total_error, 
                                                            step_error = step_error_cf, 
                                                            trotter_exponentials = True, 
                                                            splits = 2, n = total_time)
                min_costs.append(min_cost)
                min_costs_h.append(min_cost_h)

            # Implement a log log fit of min_cost vs total_time
            log_min_costs = np.log(np.array(min_costs))
            log_total_time_list = np.log(np.array(total_time_list))
            
            # Fit a line
            fit = np.polyfit(log_total_time_list, log_min_costs, 1)
            f1 = fit[1]
            f0 = fit[0]

            f1_formatted = convert_sci_to_readable('{:.4e}'.format(np.exp(f1)))
            label = f'CFQM $m={m}$'#, ${f1_formatted}\cdot T^{{{f0:.5f}}}$'
            ax.scatter(total_time_list, min_costs, label = label, color = c, s = 5, marker = '^')
            print(f'CFQM right, s={s}, $m={m}$, ${f1_formatted}\cdot T^{{{f0:.5f}}}$')


        # CFQM split
        for (ms, c) in zip(range_ms, [colors[2]]):
            min_costs = []
            min_costs_h = []
            for total_time in total_time_list:
                min_cost_h, min_cost = minimize_cost_CFMagnus_split(hs, s, ms, total_time, total_error, 
                                                                    step_error = step_error_split)
                min_costs.append(min_cost)
                min_costs_h.append(min_cost_h)

            # Implement a log log fit of min_cost vs total_time
            log_min_costs = np.log(np.array(min_costs))
            log_total_time_list = np.log(np.array(total_time_list))
            
            # Fit a line
            fit = np.polyfit(log_total_time_list, log_min_costs, 1)
            f1 = fit[1]
            f0 = fit[0]

            f1_formatted = convert_sci_to_readable('{:.4e}'.format(np.exp(f1)))
            label = f'CFQM split $m_s={ms}$'#, ${f1_formatted}\cdot T^{{{f0:.5f}}}$'
            print(f'CFQM split, $s={s}$ $m_s={ms}$, ${f1_formatted}\cdot T^{{{f0:.5f}}}$')
            ax.scatter(total_time_list, min_costs, label = label, color = c, s = 5, marker = '*')

            fitted_y_values = np.exp(f1) * total_time_list ** f0
            #ax.plot(total_time_list, fitted_y_values, color = c, linestyle = '--')


        # Suzuki
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

        f1_formatted = convert_sci_to_readable('{:.4e}'.format(np.exp(f1)))
        label = f'Suzuki'#, ${f1_formatted}\cdot T^{{{f0:.5f}}}$'
        print(f'Suzuki, $s={s}$, ${f1_formatted}\cdot T^{{{f0:.5f}}}$')
        ax.scatter(total_time_list, min_costs, label = label, color = colors[3], s = 5, marker = 'o')

        # Cambiar la fuente de los xticks
        ax.tick_params(axis='x', labelsize = 14)

        # Cambiar la fuente de los yticks
        ax.tick_params(axis='y', labelsize = 14)

        # set logscale
        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.set_ylim([1e3, 5e9])

        # set x label
        ax.set_xlabel(r'Total time $T = n$', size=14)
        if letter == '(a)':
            ax.set_ylabel(r'Number of exponentials', size=14)
        else:
            ax.set_ylabel(r'', size=14)

        ax.text(0.5, 0.07, f'{letter} $s = {s}$' , horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize=14, weight='bold')

        ax.legend(loc = 'upper left')


        fig.savefig(f'figures/time_n_scaling_{letter}.pdf', bbox_inches='tight')