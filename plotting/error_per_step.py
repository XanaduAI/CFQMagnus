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

hs = [1/2**(i/3+3) for i in range(1,100)]


# Plot the value of h vs the error sum for different values of s and m
n = 64
total_time = n
total_error = 1e-7

with open('results/step_error_CFMagnus.json', 'r') as f:
    step_error_cf = json.load(f, object_hook=convert_keys_to_float)
with open('results/step_error_CFMagnus_split.json', 'r') as f:
    step_error_split = json.load(f, object_hook=convert_keys_to_float)

fig, (ax, ax2) = plt.subplots(1, 2, figsize = (15,10))

range_s = [1,2,3,4]
range_m = [1,2,5,11]
colors = ['r', 'g', 'b', 'k']

for s, m, c in zip(range_s, range_m, colors):
    error_list = list(step_error_cf[total_error][n][s][m].values())[:99]
    ax.plot(hs, error_list, label = f'CF Magnus, s = {s}, m = {m}', linestyle = '-', color = c)

############## Split operator ################

range_s = [2,3]
range_m = [12,20]
colors = ['g', 'b']


for s, m, c in zip(range_s, range_m, colors):
    error_list = list(step_error_split[total_error][n][s][m].values())[:99]
    ax.plot(hs, error_list, label = f'CF Magnus Split, s = {s}, $m_s$ = {m}', linestyle = 'dashdot', color = c)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Step size $h$')
ax.set_ylabel('Error $\epsilon$')

ax.legend()
################################# Second plot #################################

range_s = [1,2,3,4]
range_m = [1,2,5,11]
colors = ['r', 'g', 'b', 'k']
for s, m, c in zip(range_s, range_m, colors):
    cost_exponentials = {}
    errors = {}
    for h in hs:
        cost_exponentials[h] = total_time*m/h * 5**(s-1)
        errors[h] = total_time*step_error_cf[total_error][total_time][s][m][h]/h
    ax2.plot(cost_exponentials.values(), errors.values(), label = f'CF Magnus, s = {s}, m = {m}', linestyle = '-', color = c)

# Let us also plot the error for the split operator
range_s = [2,3]
range_m = [12,20]
colors = ['g', 'b']
for s, m, c in zip(range_s, range_m, colors):
    cost_exponentials = {}
    errors = {}
    for h in hs:
        cost_exponentials[h] = total_time*m/h
        errors[h] = total_time*step_error_split[total_error][total_time][s][m][h]/h
    ax2.plot(cost_exponentials.values(), errors.values(), label = f'CF Magnus Split, s = {s}, $m_s$ = {m}', linestyle = 'dashdot', color = c)


######## Finally, let us also plot Suzuki cost ########
range_s = [1,2,3,4]
colors = ['r', 'g', 'b', 'k']
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

ax2.set_xlabel('Cost of exponentials')
ax2.set_ylabel('Error $\epsilon$')

ax2.legend()
#ax2.set_ylim([1e-16, 1e-1])





fig.savefig('figures/error_per_step.pdf')