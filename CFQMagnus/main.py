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

from magnus_errors import *

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

########### Commutator Free Magnus ###########

range_s = [1, 2, 2, 3, 3]#, 4]
range_m = [1, 2, 3, 5, 6]#, 11]

step_error_cf = compute_step_error_cf(hs, range_s, range_m, maxp = 50, total_time_list=total_time_list, use_max = True)

with open('results/step_error_CFMagnus.json', 'w') as f:
    json.dump(step_error_cf, f)

with open('results/step_error_CFMagnus.json', 'r') as f:
    step_error_cf = json.load(f, object_hook=convert_keys_to_float)

########### Trotter product formula ###########

range_s_trotter = [1,2,3]

step_error_trotter = compute_trotter_step_error(hs, range_s_trotter, maxp = 50, total_time_list = total_time_list, use_max = True)

with open('results/step_error_trotter.json', 'w') as f:
    json.dump(step_error_trotter, f)

with open('results/step_error_trotter.json', 'r') as f:
    step_error_trotter = json.load(f, object_hook=convert_keys_to_float)

########### Commutation-free Magnus split-operator ###########

range_ss = [2, 3]
range_ms = [12, 20]

step_error_split = compute_step_error_split(hs, range_ss, range_ms, maxp = 50, use_max = True)

with open('results/step_error_CFMagnus_split.json', 'w') as f:
    json.dump(step_error_split, f)

with open('results/step_error_CFMagnus_split.json', 'r') as f:
    step_error_split = json.load(f, object_hook=convert_keys_to_float)