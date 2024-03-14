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


# In this script we compile the coefficients of the CFQM and save them to a json file.
# The data was obtained from https://iopscience.iop.org/article/10.1088/0305-4470/39/19/S05,
# for the split-operator CFQMs, and from https://personales.upv.es/~serblaza/2006APNUM.pdf 
# and https://arxiv.org/abs/1102.5071 for the non-split operator CFQMs.


import numpy as np
import json
import os
from sympy.matrices import Matrix
import sympy as sp

############## Functions to generate matrix T and the quadrature matrix Q ###########################

def generate_T(order):
    T = Matrix(order, order, lambda i,j: (1-(-1)**(i+j+1))/((i+j+1)*2**(i+j+1)))
    return T

def generate_T_large(order, dim=25):
    T = Matrix(order, dim, lambda i,j: (1-(-1)**(i+j+1))/((i+j+1)*2**(i+j+1)))
    return T

def gauss_legendre(n,x):
    Pnx = sp.legendre(n,x)
    Pp = sp.diff(Pnx,x)
    ci = sp.solve( Pnx, x )
    wi = [ sp.simplify(2/(1 - xj**2)/(Pp.subs(x,xj))**2) for xj in ci ]
    return ci, wi

def generate_Q(order):
    x = sp.Symbol('x')
    ci, wi = gauss_legendre(order,x)
    # sort the roots and weights
    indices = [ci.index(c) for c in np.sort(ci)]
    ci = [ci[i] for i in indices]
    wi = [wi[i] for i in indices]
    Q = Matrix(order, order, lambda i,j: wi[j]*(ci[j]**i)/(2**(i+1)))
    return Q


############## Functions to changes between the basis of univariate integrals and Lie algebra generators ###########################
def x_from_y(n: int, y: np.ndarray):
    r'''
    n: size of the vector
    x: vector to be multiplied by T

    Returns y @ T
    '''
    assert(y.shape == (1,n))
    T = generate_T(n)
    return y @ T

def y_from_x(n: int, x: np.ndarray):
    r'''
    n: size of the vector
    y: vector to be multiplied by R

    Returns x @ R
    '''
    assert(x.shape == (1,n))
    R = generate_T(n).inv()
    return x @ R

def overline_x_from_y(n: int, y: np.ndarray):
    r'''
    n: size of the vector
    x: vector to be multiplied by T

    Returns y @ T
    '''
    assert(y.shape == (1,n))
    T = generate_T_large(n)
    return y @ T

def y_from_z(n: int, z: np.ndarray):
    r'''
    n: size of the vector
    z: vector to be multiplied by R

    Returns z @ Q_inv
    '''
    assert(z.shape == (1,n))
    Q_inv = generate_Q(n).inv()
    return z @ Q_inv

def z_from_y(n:int, y:np.ndarray):
    r'''
    n: size of the vector
    y: vector to be multiplied by R

    Returns y @ Q
    '''
    assert(y.shape == (1,n))
    Q = generate_Q(n)
    return y @ Q

################# Helper functions ######################

# Function to convert keys to float
def convert_keys_to_float(data):
    new_data = {}
    for key, value in data.items():
        new_key = float(key)
        new_data[new_key] = value
    return new_data

# Get current directory
dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
save_path = os.path.join(dir_path, 'coefficients')

# Dictionaries indexed by (s, m)

xs = {}
ys = {} 
xs[2] = {}
xs[2][2] = [[1/2, 1/6], [1/2, -1/6]]
xs[2][3] = [[0, 1/12], [1, 0], [0, -1/12]]
xs[3] = {}
xs[3][5] = [[0.2, 0.08734395950888931101, 0.03734395950888931101], 
        [0.34815492558797391479, 0.053438272547684150, 0.00584269157837031012],
        [1-2*(0.2+0.34815492558797391479), 0, 1/12-2*(0.03734395950888931101 + 0.00584269157837031012)],
        [0.34815492558797391479, -0.053438272547684150, 0.00584269157837031012],
        [0.2, -0.08734395950888931101, 0.03734395950888931101]]
xs[3][6] = [[0.208, 0.09023186422416794596, 0.03823186422416794596], 
        [0.312, 0.04467385661651479788, 0.00439421553992544024],
        [1/2-(0.208 + 0.312), 0.01407960659498524468, 1/24-(0.03823186422416794596+0.00439421553992544024)],
        [1/2-(0.208 + 0.312), -0.01407960659498524468, 1/24-(0.03823186422416794596+0.00439421553992544024)],
        [0.312, -0.04467385661651479788, 0.00439421553992544024],
        [0.208, -0.09023186422416794596, 0.03823186422416794596]]

zs = {}
zs[4] = {}
zs[4][11] = [
        [0.169715531043933180094151,    +0.152866146944615909929839, 0.119167378745981369601216,  +0.068619226448029559107538],
        [0.379420807516005431504230,    +0.148839980923180990943008, -0.115880829186628075021088, -0.188555246668412628269760],
        [0.469459306644050573017994,    -0.379844237839363505173921, 0.022898814729462898505141,  +0.571855043580130805495594],
        [-0.448225927391070886302766,   +0.362889857410989942809900, -0.022565582830528472333301, -0.544507517141613383517695],
        [-0.293924473106317605373923,   -0.026255628265819381983204,  0.096761509131620390100068, +0.000018330145571671744069],
        [0.447109510586798614120629,    0,                           -0.200762581179816221704073, 0],
        [-0.293924473106317605373923,   +0.026255628265819381983204,  0.096761509131620390100068, -0.000018330145571671744069],
        [-0.448225927391070886302766,   -0.362889857410989942809900, -0.022565582830528472333301, +0.544507517141613383517695],
        [0.469459306644050573017994,    +0.379844237839363505173921, 0.022898814729462898505141,  -0.571855043580130805495594],
        [0.379420807516005431504230,    -0.148839980923180990943008, -0.115880829186628075021088, +0.188555246668412628269760],
        [0.169715531043933180094151,    -0.152866146944615909929839, 0.119167378745981369601216,  -0.068619226448029559107538]
        ]

# Compute the coefficients ys[s][m]
ys = {}
ys[1] = {}
ys[1][1] = [[1]]
ys[2] = {}
ys[2][2] = [list(y_from_x(2, np.expand_dims(x, axis = 0))) for x in xs[2][2]]
ys[2][3] = [list(y_from_x(2, np.expand_dims(x, axis = 0))) for x in xs[2][3]]
ys[3] = {}
ys[3][5] = [list(y_from_x(3, np.expand_dims(x, axis = 0))) for x in xs[3][5]]
ys[3][6] = [list(y_from_x(3, np.expand_dims(x, axis = 0))) for x in xs[3][6]]

ys[4] = {}
ys[4][11] = [list(y_from_z(4, np.expand_dims(z, axis = 0))) for z in zs[4][11]]


# Compute the coefficients zs[s][m]
zs[2] = {}
zs[2][2] = [list(z_from_y(2, np.expand_dims(y, axis = 0))) for y in ys[2][2]]
zs[2][3] = [list(z_from_y(2, np.expand_dims(y, axis = 0))) for y in ys[2][3]]
zs[3] = {}
zs[3][5] = [list(z_from_y(3, np.expand_dims(y, axis = 0))) for y in ys[3][5]]
zs[3][6] = [list(z_from_y(3, np.expand_dims(y, axis = 0))) for y in ys[3][6]]

# Compute the coefficients xs[s][m]
xs[4] = {}
xs[4][11] = [list(x_from_y(4, np.expand_dims(y, axis = 0))) for y in ys[4][11]]


# overline_xs
overline_xs = {}
overline_xs[2] = {}
overline_xs[2][2] = [list(overline_x_from_y(2, np.expand_dims(y, axis = 0))) for y in ys[2][2]]
overline_xs[2][3] = [list(overline_x_from_y(2, np.expand_dims(y, axis = 0))) for y in ys[2][3]]
overline_xs[3] = {}
overline_xs[3][5] = [list(overline_x_from_y(3, np.expand_dims(y, axis = 0))) for y in ys[3][5]]
overline_xs[3][6] = [list(overline_x_from_y(3, np.expand_dims(y, axis = 0))) for y in ys[3][6]]
overline_xs[4] = {}
overline_xs[4][11] = [list(overline_x_from_y(4, np.expand_dims(y, axis = 0))) for y in ys[4][11]]

def default(o):
    return float(o)

# Dump the coefficients
with open(os.path.join(save_path, 'xs.json'), 'w') as f:
    json.dump(xs, f, default=default)

with open(os.path.join(save_path, 'ys.json'), 'w') as f:
    json.dump(ys, f, default=default)

with open(os.path.join(save_path, 'zs.json'), 'w') as f:
    json.dump(zs, f, default=default)

with open(os.path.join(save_path, 'overline_xs.json'), 'w') as f:
    json.dump(overline_xs, f, default=default)




########################## Split methods ##########################

xs_split = {}
xs_split[2] = {}
b11 = 0.0792036964311957
a11 = 0.209515106613362
b21 = 0.353172906049774
a21 = -0.143851773179818
b31 = -0.0420650803577195
a31 = 1/2 -(a11+a21)
b41 = 1-2*(b11+b21+b31)

c = 12*(a11 + 2*a21 + a31 - 2*b11 + 2*a11*b11 - 2*b21 + 2*a11*b21)
d = 12*(2*a21 - b11 + 2*a11*b11 - 2*a21*b11 - b21 + 2*a11*b21)

b12 = (2*a11+2*a21-2*b11-b21)/d
a12 = (2*a11+2*a21+a31-2*b11-2*b21)/c
b22 = (-2*a11 + b11)/d
a22 = 0
b32 = 0
a32 = -a11/c
b42 = 0

xs_split[2][12] = [
    [0., 0.],
    [b11, b12],
    [a11, a12],
    [b21, b22],
    [a21, a22],
    [b31, b32],
    [a31, a32],
    [b41, b42],
    [a31, -a32],
    [b31, -b32],
    [a21, -a22],
    [b21, -b22],
    [a11, -a12],
    [b11, -b12]
]

xs_split[3] = {}

a11 = 0.0502627644003922 
a12 = 0.022059009674017884 
a13 = -0.000326878764898432
b11 = 0.148816447901042 
b12 = 0.06325193140810957 
b13 = 0.03156029484304291
a21 = 0.413514300428344 
a22 = 0.03639087263834154 
a23 = 0.05639771119273678
b21 = -0.132385865767784 
b22 = -0.0564220584435047 
b23 = 0.00004713758165544868
a31 = 0.0450798897943977 
a32 = -0.029722051174027396 
a33 = 0.0032603041391350658
b31 = 0.067307604692185 
b32 = 0.030997085102486225 
b33 = 0.001271609241968303
a41 = -0.188054853819569 
a42 = 0.07316095552711696 
a43 = -0.008
b41 = 0.432666402578175 
b42 = 0.086709890573243 
b43 = 0.012967625
a51 = 0.541960678450780 
a52 = -0.10825317547305482 
a53 = 0
b51 = 1/2 - (b11 + b21 + b31 + b41) 
b52 = 0 
b53 = -0.00418
a61 = 1 - 2*(a11 + a21 + a31 + a41 + a51) 
a62 = 0 
a63 = -0.019328939800613495

xs_split[3][20] = [
    [a11, a12, a13],
    [b11, b12, b13],
    [a21, a22, a23],
    [b21, b22, b23],
    [a31, a32, a33],
    [b31, b32, b33],
    [a41, a42, a43],
    [b41, b42, b43],
    [a51, a52, a53],
    [b51, b52, b53],
    [a61, a62, a63],
    [b51, -b52, b53],
    [a51, -a52, a53],
    [b41, -b42, b43],
    [a41, -a42, a43],
    [b31, -b32, b33],
    [a31, -a32, a33],
    [b21, -b22, b23],
    [a21, -a22, a23],
    [b11, -b12, b13],
    [a11, -a12, a13],
    [0, 0, 0]
]


# Compute the coefficients ys[s][m]
ys_split = {}
ys_split[2] = {}
ys_split[2][12] = [list(y_from_x(2, np.expand_dims(x, axis = 0))) for x in xs_split[2][12]]
ys_split[3] = {}
ys_split[3][20] = [list(y_from_x(3, np.expand_dims(x, axis = 0))) for x in xs_split[3][20]]


# Compute the coefficients zs[s][m]
zs_split = {}
zs_split[2] = {}
zs_split[2][12] = [list(z_from_y(2, np.expand_dims(y, axis = 0))) for y in ys_split[2][12]]
zs_split[3] = {}
zs_split[3][20] = [list(z_from_y(3, np.expand_dims(y, axis = 0))) for y in ys_split[3][20]]

# Compute the coefficients overline_xs[s][m]
overline_xs_split = {}
overline_xs_split[2] = {}
overline_xs_split[2][12] = [list(overline_x_from_y(2, np.expand_dims(y, axis = 0))) for y in ys_split[2][12]]
overline_xs_split[3] = {}
overline_xs_split[3][20] = [list(overline_x_from_y(3, np.expand_dims(y, axis = 0))) for y in ys_split[3][20]]


# Dump the coefficients
with open(os.path.join(save_path, 'xs_split.json'), 'w') as f:
    json.dump(xs_split, f, default=default)

with open(os.path.join(save_path, 'ys_split.json'), 'w') as f:
    json.dump(ys_split, f, default=default)

with open(os.path.join(save_path, 'zs_split.json'), 'w') as f:
    json.dump(zs_split, f, default=default)

with open(os.path.join(save_path, 'overline_xs_split.json'), 'w') as f:
    json.dump(overline_xs_split, f, default=default)