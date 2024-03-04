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
import numpy as np
import matplotlib.pyplot as plt
import json
import os

from CFQMagnus.magnus_errors import *
import scienceplots

plt.style.use('science')

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
    x, y = map(float, number_readable.split('e+'))

    # Format as x * 10^y
    number_formatted = f"{x} \\cdot 10^{{{int(y)}}}"

    return number_formatted

# Compute a dictionary with the value of the factorial
factorial = {}
for i in range(0, 75):
    factorial[i] = np.longdouble(math.factorial(i))

hs = [1/2**(i/5+3) for i in range(1,250)]


# Plot the value of h vs the error sum for different values of s and m
n = 128
total_time = 1

with open('results/step_error_CFMagnus.json', 'r') as f:
    step_error_cf = json.load(f, object_hook=convert_keys_to_float)
with open('results/step_error_CFMagnus_split.json', 'r') as f:
    step_error_split = json.load(f, object_hook=convert_keys_to_float)

with plt.style.context('science'):

    fig, ax = plt.subplots(1, 1, figsize = (4,4))

    range_s = [1,2,2,3,3]
    range_m = [1,2,3,5,6]
    colors = ['y', 'orange', 'red', 'blue', 'green']

    for s, m, c in zip(range_s, range_m, colors):
        error_list = list(step_error_cf[n][s][m].values())
        ax.plot(hs, error_list, label = f'CFQM, $s = {s}, m = {m}$', linestyle = '-', color = c)

    ############## Split operator ################

    range_s = [2,3]
    range_m = [12,20]
    colors = ['orange', 'b']


    for s, m, c in zip(range_s, range_m, colors):
        error_list = list(step_error_split[s][m].values())
        ax.plot(hs, error_list, label = f'CFQM split, $s = {s}, m_s = {m}$', linestyle = 'dashdot', color = c)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Step size $h$', fontsize = 14)
    ax.set_ylabel('Error $\epsilon$', fontsize = 14)

    ax.set_ylim([1e-10, 1e-2])
    ax.set_xlim([1e-3, 1e-1])

    plt.xticks(fontsize=14)  # Controla el tamaño de las etiquetas del eje x
    plt.yticks(fontsize=14)  # Controla el tamaño de las etiquetas del eje y

    ax.text(1.2e-3, 1e-3, f'(a)' , horizontalalignment='left', verticalalignment='bottom', fontsize=14, weight='bold') # , transform=ax.transAxes

    fig.savefig('figures/error_per_step.pdf')



    #ax.legend()
    ################################# Second plot #################################

    fig, ax2 = plt.subplots(1, 1, figsize = (4,4))

    range_s = [1,2,2,3,3]
    range_m = [1,2,3,5,6]
    colors = ['y', 'orange', 'red', 'blue', 'green']
    for s, m, c in zip(range_s, range_m, colors):
        cost_exponentials = {}
        errors = {}
        for h in hs:
            cost_exponentials[h] = total_time*m/h * 5**(s-1)
            errors[h] = total_time*step_error_cf[n][s][m][h]/h
        ax2.plot(cost_exponentials.values(), errors.values(), label = f'CFQM, $s = {s}, m = {m}$', linestyle = '-', color = c)

    # Let us also plot the error for the split operator
    range_s = [2,3]
    range_m = [12,20]
    colors = ['orange', 'b']
    for s, m, c in zip(range_s, range_m, colors):
        cost_exponentials = {}
        errors = {}
        for h in hs:
            cost_exponentials[h] = total_time*m/h
            errors[h] = total_time*step_error_split[s][m][h]/h
        ax2.plot(cost_exponentials.values(), errors.values(), label = f'CFQM split, $s = {s}, m_s = {m}$', linestyle = 'dashdot', color = c)


    ######## Finally, let us also plot Suzuki cost ########
    range_s = [1,2,3]
    colors = ['r', 'orange', 'b']
    for s, c in zip(range_s, colors):
        costs = []
        errors = []
        for h in hs:
            error_suzuki = suzuki_wiebe_error(h, s, total_time)
            cost_suzuki = suzuki_wiebe_cost(h, s, total_time, error_suzuki)
            costs.append(cost_suzuki)
            errors.append(error_suzuki)
        #ax2.plot(costs, errors, label = f'Suzuki, s = {s}', linestyle = '--', color = c)

    ax2.set_xscale('log')
    ax2.set_yscale('log')

    ax2.set_xlabel('Number of exponentials', fontsize = 14)
    #ax2.set_ylabel('Error $\epsilon$', fontsize = 14)

    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize = 14)

    ax2.set_ylim([1e-10, 1e-2])
    ax2.set_xlim([1e2, 2e5])

    plt.xticks(fontsize=14)  # Controla el tamaño de las etiquetas del eje x
    plt.yticks(fontsize=14)  # Controla el tamaño de las etiquetas del eje y

    ax2.text(7e4, 1e-3, f'(b)' , horizontalalignment='left', verticalalignment='bottom', fontsize=14, weight='bold')


    fig.savefig('figures/cost_per_step.pdf')