#### Functions to compute the error terms in the Magnus expansion

import numpy as np
from sympy.utilities.iterables import partitions
from itertools import permutations
from tqdm import tqdm
import math
from tqdm import tqdm
from sympy.matrices import Matrix
import sympy as sp
import json
import os



################ Bounding \tilde{\Psi}_m^{[2s]} ########################

def efficient_number_compositions_old(p: int, s: int, factorial: dict) -> dict:
    r'''
    Given a number p, computes a dictionary analyzing the 
    integer compositions of p. The dictionary key is the length of the composition of p,
    and the value is the number of compositions of that length.

    Parameters
    ----------
    p: integers up to which compositions add up
    s: maximum value of each term in the composition
    factorial: a dictionary with the factorial of all the numbers up to p or above
        (to avoid recomputing them every time you use it.)
    '''
    raise Warning('This function is deprecated. Use efficient_number_compositions instead.')
    pts = partitions(p, k=s) #todo: remove k=s
    
    comps = {}
    for part in pts:
        len_ = np.sum(list(part.values()))

        num_comps = factorial[len_]
        for n in part.values():
            num_comps /= factorial[n]

        if len_ in comps.keys():
            comps[len_] += num_comps
        else:
            comps[len_] = num_comps

    return comps

def efficient_number_compositions(p: int, factorial: dict) -> dict:
    r'''
    Given a number p, computes a dictionary analyzing the 
    integer compositions of p. The dictionary key is the dimension of the composition of p,
    and the value is the number of compositions of that dimension.

    Parameters
    ----------
    p: integers up to which compositions add up
    factorial: a dictionary with the factorial of all the numbers up to p or above
        (to avoid recomputing them every time you use it.)
    '''
    pts = partitions(p)
    
    comps = {}
    for part in pts:
        len_ = np.sum(list(part.values()))

        num_comps = factorial[len_]
        for n in part.values():
            num_comps /= factorial[n]

        if len_ in comps.keys():
            comps[len_] += num_comps
        else:
            comps[len_] = num_comps

    return comps


def partition_to_list(partition: dict) -> list:
    r'''Converts a partition dictionary to a list.'''
    l = []
    for k, v in partition.items():
        l += [k]*v
    return l


def add_term(m: int, mean_cs: list, partition: dict, factorial: dict) -> float:
    r'''
    For a given partition $k_1,\ldots,k_m$, compute

    .. math::
        \sum_{\vec{k}\in P(i_1,\ldots,i_m)}\frac{\bar{c}^{k_1}_1\cdots \bar{c}^{k_m}_m}{k_1!\cdots k_m!}

    where $c_1,\ldots,c_m$ are the mean values of the coefficients of the Magnus expansion, for exponential m
    and $P(i_1,\ldots,i_m)$ is the set of all permutations of the list $(i_1,\ldots,i_m)$.

    Parameters
    ----------
    m: number of terms in the Magnus expansion
    mean_cs: list of the mean values of the coefficients of the Magnus expansion
    partition: dictionary with the partition of the number of terms in the Magnus expansion
    factorial: a dictionary with the factorial of all the numbers up to \max(i_1,\ldots,i_m) or above.
    '''
    part_ = partition_to_list(partition)
    part_ = part_ + [0]*(m-len(part_))
    perms = permutations(part_)
                
    add_term = np.longdouble(0)
    for per in perms:
        add_term += np.prod([mean_cs[i]**per[i]/factorial[per[i]] for i in range(len(per))])
    return add_term


def efficient_weak_compositions(max_dim_w: int, m: int, cs: list, factorial: dict, use_max: bool = True):
    r'''
    Computes sum{k_1...k_m\in weak_compositions(max_dim_w)} c^{k_1}/k_1!...c^{k_m}/k_m!) for all weak compositions of length m.
    Depending on the value of `use_max`, it uses either the maximum value of c for all at once (faster) or the specific values of c,
    (more accurate).

    Parameters
    ----------
    max_dim_w: normw is the sum value of the weak composition.
        max_dim_w is the maximum value of normw. The result with be computed for range(max_dim_w+1)
    list_m: list of the possible values of m.
    list_cs: correspondingly, list of the values of c.
    use_max: 
        if True, use the maximum value of cs instead of the specific values of cs. 
        Faster but less tight bound will be returned.
    factorial: a dictionary with the factorial of all the numbers up to max_dim_w or above.

    Returns
    -------
    weak_comps: a dictionary with the sum for each m, for each normw.
    '''

    weak_comps = {}

    # In case we want to be more precise
    vector_add_term = np.vectorize(lambda part: add_term(m, mean_cs, part))

    max_cs = np.max(cs)
    mean_cs = np.mean(cs, axis = 1)

    weak_comps[m] = {}
    for dim_w in range(max_dim_w+1):
        
        suma = np.longdouble(0)

        partitions_k = partitions(dim_w, m=m)

        if use_max:
            for k in partitions_k:
                len_ = np.sum(list(k.values()))

                # factorial[m]/ prod factorial[n] is the number of compositions 
                # corresponding to this partition (eg permutations)
                num_comps = factorial[m]/factorial[m-len_]
                for n in k.values():
                    num_comps /= factorial[n] 

                denominator = np.longdouble(1)
                for ki, num_ki in k.items():
                    denominator = denominator * (factorial[ki] ** num_ki) #todo: added "** num_ki"

                suma = suma + num_comps / denominator * max_cs**dim_w

        else:
            add = vector_add_term(list(partitions_k))
            suma += np.sum(add)

        weak_comps[m][dim_w] = suma

    return weak_comps


def Psi_m_Taylor_error(h: float, maxp: int, s: int, m: int, bar_xs: list, factorial: list, use_max: bool = True):
    r'''
    Computes
    .. math::
        \sum_{p = 1}^{maxp} h^p \sum_{
            \substack{
                w\in\mathcal{C}_{s}(p)
                }
        }
        \sum_{k_i : \sum_{i=1}^m k_i = |w|}
        \frac{\bar{c}^{k_1}\cdots \bar{c}^{k_m}}{k_1!\cdots k_m!}


    Parameters
    ----------
    h: step size
    maxp: maximum order p of the summatory (It is assumed that this is sufficient for convergence)
    s: 2s is the order of the commutator-free Magnus operator
    m: number of exponentials in the commutator-free Magnus expansion
    bar_xs: list of values $|\overline{x}_{i,j}a_j|$ for each i,j
    factorial: a dictionary with the factorial of all the numbers up to maxp or above.
    use_max: whether to use a faster but less tight bound (True) or a slower but tighter bound (False)

    Returns
    -------
    suma: a dictionary with the sum for each p.
    '''

    # We first precompute all the weak compositions
    weak_comps = efficient_weak_compositions(maxp, m, bar_xs, factorial, use_max)

    p = 2*s+1
    compositions_p = efficient_number_compositions(p, factorial)
    
    last_contribution = np.longdouble(0)
    for dim_Cp, num_Cp in compositions_p.items():
        last_contribution += num_Cp*weak_comps[m][dim_Cp]

    last_contribution *= h**p
    error = last_contribution

    while last_contribution /error > 1e-5:
        p += 1
        if p > maxp:
            raise ValueError('The error is not converging')
        compositions_p = efficient_number_compositions(p, factorial)

        last_contribution = np.longdouble(0)
        for dim_Cp, num_Cp in compositions_p.items():
            last_contribution += num_Cp*weak_comps[m][dim_Cp] 
        
        last_contribution *= h**p

        error += last_contribution

    return error


############## Bounding \|\exp(\Omega(h))\| ###########################

def exp_Omega_bound(h: float, p: int, s: int, maxc: float, factorial: dict):
    r'''
    Computes a bound for the error of the Magnus expansion of order p.
    in 
    .. math::
        \sum_{p=2s+1}^\infty \|\exp(\Omega_{2s,2s})_{p}\|

    .. math::
        \sum_{p=0}^\infty\left(\frac{h}{2}\right)^p \sum_{\bm{k}\in\mathcal{C}(p)}\frac{1}{|\bm{k}|!} \prod_{l=1}^z \sum_{\bm{j}_l\in\mathcal{C}(k_l)} \frac{c^{|\bm{j}_l|}}{|\bm{j}_l|}\frac{1}{{j_1}_l\cdots {j_n}_l}

    Parameters
    ----------
    h: step size
    p: order of the Magnus expansion
    s: 2s is the order of the Commutator Free Magnus operator
    maxc: maximum value of the norm of |a_j|
    factorial: a dictionary with the factorial of all the numbers up to p or above.

    Returns
    -------
    bound: the bound for the error of the Magnus expansion of order p.
    '''
    bound = np.longdouble(0)

    # We first generate all partitions of p into z parts
    parts_p = list(partitions(p))

    # For each possible partition,
    for k in parts_p:
        # We first compute the size and number of combinations of the external partition
        dim_k = np.sum(list(k.values()))
        permutations_part_k = factorial[dim_k]/np.prod([factorial[v] for v in k.values()])

        # We further generate more partitions of each part into up to 2s parts
        product = np.longdouble(1)
        for kl, kl_repetitions in k.items(): # Iterating over the dictionary of a big partition
            suma = np.longdouble(0)
            j_parts = partitions(kl)
            for jl in j_parts: # Here we get a dictionary of small partitions
                dim_jl = np.sum(list(jl.values()))
                permutations_part_jl = factorial[dim_jl]/np.prod([factorial[jlv] for jlv in jl.values()])
                term = (2*maxc)**dim_jl / dim_jl
                term = term / np.prod([jlk**jlv for jlk, jlv in jl.items()])
                suma = suma + term * permutations_part_jl

            # For we have to multiply product by suma as many times as k appears
            product = product * (suma ** kl_repetitions)

        bound += permutations_part_k/factorial[dim_k] * product

    return bound * (h/2)**p


############## Quadrature error ###########################

def quadrature_residual(h, s, maxc = 1):
    r'''
    Computes the error of the quadrature rule for the Magnus expansion of order 2s+1.

    Parameters
    ----------
    h: step size
    s: 2s is the order of the Magnus expansion
    maxc: maximum value of the norm |a_j|

    Variables used
    --------------
    n: order of the quadrature rule (2s)
    i: index of the univariate integral
    j: missing terms in the univariate integral, that constitute the error term
    '''
    quadrature_res = {}
    for i in range(s):
        quadrature_res[i] = {}
        n = 2*s
        partial_sums = []
        for j in range(0, 350):
            term = math.factorial(2*n+j)/math.factorial(j) * maxc * (h/2)**(j)
            partial_sums.append(term + partial_sums[-1] if len(partial_sums) > 0 else term)

        partial_sums = np.array(partial_sums)
        partial_sums *= (math.factorial(n)/math.factorial(2*n))**3 * math.factorial(n)/(2*n+1) * h**(2*n+1-i)

        quadrature_res[i] = partial_sums[-1]

    return quadrature_res

def quadrature_error(h, s, m, ys, maxc = 1, qr = None):
    r'''
    Computes the error of the quadrature rule error of order s,
    for the commutator free Magnus expansion of order 2s+1.

    Parameters
    ----------
    h: step size
    s: 2s is the order of the Magnus expansion
    m: number of exponentials in the Commutator Free Magnus operator
    ys: list of the coefficients y_{i,j} of the Magnus expansion
    maxc: maximum value of the norm |a_j|
    '''

    if qr is None:
        qr = quadrature_residual(h, s, m, maxc = maxc)

    error = 0
    for i in range(m):
        for j in range(s):
            error += qr[j] * abs(ys[s][m][i][j])

    return error

############## Trotter error ###########################

def trotter_error_spu_formula(n, h, s, u = 1):
    r"""
    n: number of spins.
    h: step size
    s: 2s is the order of the Magnus expansion and of the Trotter product formula
    u: maximum value of the norm of the Hamiltonian and its derivatives.
    """

    # Following notation from the paper
    stages = 2*5**(s-1)
    p = 2*s
    error = np.longdouble(0)

    for k in range(1, stages+1):
        error += n * (2*k-1)**p * (k*u) * (2*u)**p * (2*k-2)**p * h**(p+1) / math.factorial(p+1)
        error += n * (2*k+1)**p * (k*u) * (2*u)**p * (2*k)**p * h**(p+1) / math.factorial(p+1)

    return error

def trotter_error_k_local(h: float, s: int, k: int, maxc: float):
    r"""
    Error bound for the Trotter product formula for a k-local Hamiltonian.

    Parameters
    ----------
    h: step size
    s: 2s is the order of the Magnus expansion and of the Trotter product formula
    k: locality of the Hamiltonian
    maxc: norm of the Hamiltonian
    """
    stages = 2*5**(s-1)
    norm = np.longdouble(1)
    for p in range(2*s+1):
        norm *=  2*k*(k+(p-1)*(k-1))*maxc
    return 2*(stages*h)**(p+1)/(p+1) * norm

def trotter_error_order2AB(h: float, normA: float = 1/2, normB: float = 1/2):
    r"""
    Uses the bound for Trotter formula of order 2 in Proposition 10 of the paper
    "Theory of Trotter Error with Commutator Scaling". See also eq L5 of the paper.

    Parameters
    ----------
    h: step size
    normA: norm of the odd term of the Hamiltonian A
    normB: norm of the even term of the Hamiltonian B
    """

    return h**3/12 * 2**3 * normB * normB * normA + \
            h**3/24 * 2**3 * normA * normA * normB

def trotter_error_order4AB(h: float, normA: float = 1/2, normB: float = 1/2):
    r"""
    Uses the bound for Trotter formula of order 4 in Proposition M.1 of the paper
    "Theory of Trotter Error with Commutator Scaling".

    Parameters
    ----------
    h: step size
    normA: norm of the odd term of the Hamiltonian A
    normB: norm of the even term of the Hamiltonian B
    """

    return h**5 * 2**5 * (
        0.0047 * normA**4 * normB + 0.0057 * normA**3 * normB**2 +
        0.0046 * normA**3 * normB**2 + 0.0074 * normA**2 * normB**3 +
        0.0097 * normA**3 * normB**2 + 0.0097 * normA**2 * normB**3 +
        0.0173 * normA**2 * normB**3 + 0.0284 * normA * normB**4
    )

def suzuki_wiebe_error(h: float, s: int, total_time: float, maxc: float = 1):
    # total error over total time, so we have multiplied by total_time/h
    return 9/10 * (5/3)**s * h * maxc * total_time/h

def suzuki_wiebe_cost(h: float, s: int, total_time: float, epsilon: float, maxc: float = 1, splits: int = 2):
    # Lambda is taken to be 1
    # Cost over total time, so we have multiplied by total_time/h

    return 3 * splits * h * maxc * s *  (25/3)**s * (h/epsilon)**(1/(2*s)) * total_time / h


############## Helper functions ###########################

def accumulate_from(n, accumulated, maxp):
    r'''
    Computes the accumulated sum of sum_compositions from n to maxp+1.

    Parameters
    ----------
    n: starting point of the accumulated sum
    accumulated: accumulated sum from 1 to n-1
    maxp: maximum value of p for the accumulated sum

    Returns
    -------
    acc_sum_compositions: accumulated sum from n to maxp+1
    '''
    
    acc = {}
    acc[n-1] = 0
    for i in range(n, maxp+1):
        acc[i] = acc[i-1] + accumulated[i]

    return acc


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


# Deprecated
############## Bounding \|\Omega(h)\| ###########################

def Omega_bound(h: float, p: int, s: int = None, maxc: float = 1):
    r'''
    Computes a tight bound for the error of the Magnus expansion of order p,
    accounting for 2s terms in the Magnus operator.

    Parameters
    ----------
    h: step size
    p: order of the Magnus expansion
    s: 2s is the order of the Commutator Free Magnus operator
    maxc: maximum value of the norm of |a_j|

    Returns
    '''

    suma = np.longdouble(0)
    for part in partitions(p):
        prod = np.longdouble(1)
        dim = np.sum(list(part.values()))
        for k, v in part.items():
            prod = prod * maxc**v / k**v
        prod = prod / dim if dim > 0 else np.longdouble(0)
        suma = suma + prod

    return suma * (h/2)**p


############## Basis change error ###########################

def basis_change_error(h, s, m, ys, maxc = 1):
    r"""
    Computes the error of the basis change of the Magnus expansion of order 2s+1.

    Parameters
    ----------
    h: step size
    s: n = 2s is order of the Magnus expansion
    m: number of exponentials in the Commutator Free Magnus operator
    maxc: maximum value of the norm of the Hamiltonian and its derivatives
    ys: list of the values of the quadrature rule
    """

    basis_change_error = 0

    for k in range(1, m+1):
        for j in range(s):
            basis_change_error += abs(ys[s][m][k-1][j])/2**j * (2*maxc)/(2*s+j+1)

    return basis_change_error * (h/2)**(2*s+1) / (1-h/2)

# Compute a dictionary with the value of the factorial
factorial = {}
for i in range(0, 75):
    factorial[i] = np.longdouble(math.factorial(i))

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

############### CFQM error ####################

def error_sum_CF_wout_trotter(h, s, m, overline_xs, ys, maxc = 1, maxp = 40, use_max = True):
    r"""
    h: step size
    n: number of spins
    s: 2s is the order of the Magnus expansion
    m: number of exponentials in the Commutator Free Magnus operator
    overline_xs: list of the coefficients of the Magnus expansion in the basis of the Lie algebra
    ys: list of the coefficients of the Magnus expansion in the basis of univariate integrals
    maxc: maximum value of the norm of the Hamiltonian and its derivatives
    maxp: maximum order of the Magnus expansion
    use_max: if True, the maximum value of the norm of the Hamiltonian and its derivatives is used
    """
    error = np.longdouble(0)

    # Error from the Taylor expansion of the exponential of the Magnus expansion

    p = 2*s+1
    exp_omega_error = exp_Omega_bound(h, p, s, maxc, factorial)
    last_correction = exp_omega_error
    while last_correction/exp_omega_error > 1e-5:
        p += 1
        if p > maxp:
            raise ValueError('The error is not converging')
        last_correction = exp_Omega_bound(h, p, s, maxc, factorial)
        exp_omega_error += last_correction
    error += exp_omega_error

    if s>1:

        # Error from the Taylor expansion of the product of exponentials in the commutator-free operator
        error += Psi_m_Taylor_error(h, maxp, s, m, overline_xs[s][m] * maxc, factorial, use_max = use_max or s > 4)

        # Error from the quadrature rule
        qr = quadrature_residual(h, s, maxc = maxc)
        error += quadrature_error(h, s, m, ys, maxc = maxc, qr = qr)


    # Error from the Trotter product formula
    # error += trotter_error_spu_formula(n, h/Z, s, u = maxc) * m
    
    return error

def compute_step_error_cf(hs, range_s, range_m, maxp, total_time_list, use_max = True, overline_xs = overline_xs, ys = ys, zs = zs):
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
    step_error_wout_trotter = {}
    for (s, m) in tqdm(zip(range_s, range_m), desc = 's, m'):
        if s not in step_error_wout_trotter.keys():
            step_error_wout_trotter[s] = {}
        step_error_wout_trotter[s][m] = {}
        for h in hs:
            step_error_wout_trotter[s][m][h] = float(error_sum_CF_wout_trotter(h, s, m, overline_xs, ys, maxc = 1, maxp = maxp, use_max = use_max))
    for n in tqdm(total_time_list, desc = 'time'):  
        step_error[n] = {}
        for (s, m) in tqdm(zip(range_s, range_m), desc = 's, m'):
            Z = np.sum(np.abs(zs[s][m]), axis = 1) / (4*n) if m > 1 else np.array([1/(4*n)])
            assert len(Z) == m
            if s not in step_error[n].keys():
                step_error[n][s] = {}
            step_error[n][s][m] = {}
            for h in hs:
                step_error[n][s][m][h] = float(step_error_wout_trotter[s][m][h] + 
                                        np.sum([trotter_error_spu_formula(n, Z_*h, s, u = 1) for Z_ in Z])) # m exponentials to be Trotterized
    return step_error

def minimize_cost_CFMagnus(hs, s, m, total_time, total_error, step_error, trotter_exponentials = True, splits = 2, n = None):
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

    if n is None:
        n = total_time

    cost_exponentials = {}
    errors = {}
    for h in hs:
        cost_exponentials[h] = total_time*m/h
        if trotter_exponentials: 
            cost_exponentials[h] *= 2 * 5**(s-1) * splits
        errors[h] = total_time*step_error[n][s][m][h]/h

    min_cost = np.inf
    for h in hs:
        if errors[h] < total_error and cost_exponentials[h] < min_cost:
            min_cost_h = h
            min_cost = cost_exponentials[h]

    return min_cost_h, min_cost

############## CFQM split error ####################

def error_sum_CFsplit(h, s, m, overline_xs_split, ys_split, maxc = 1, maxp = 40, use_max = True):
    r"""
    Computes the step error for a split-operator commutator-free Magnus expansion.

    Parameters
    ----------
    h: step size
    n: number of spins
    s: 2s is the order of the Magnus expansion
    m: number of exponentials in the Commutator Free Magnus operator
    overline_xs_split: list of the coefficients of the Magnus expansion in the basis of the Lie algebra
    ys_split: list of the coefficients of the Magnus expansion in the basis of univariate integrals
    maxc: maximum value of the norm of the Hamiltonian and its derivatives
    maxp: maximum order of the Magnus expansion
    use_max: if True, the maximum value of the norm of the Hamiltonian and its derivatives is used
    """
    error = np.longdouble(0)

    # First, we add the error from the Taylor truncation of Omega
    p = 2*s+1
    exp_omega_error = exp_Omega_bound(h, p, s, maxc, factorial)
    last_correction = exp_omega_error
    while last_correction/exp_omega_error > 1e-5:
        p += 1
        if p > maxp:
            raise ValueError('The error is not converging')
        last_correction = exp_Omega_bound(h, p, s, maxc, factorial)
        exp_omega_error += last_correction
    error += exp_omega_error

    # Error from the Taylor expansion of the product of exponentials in the commutator-free operator
    error += Psi_m_Taylor_error(h, maxp, s, m, overline_xs_split[s][m], factorial, use_max = use_max or s > 4)

    # Error from the quadrature rule
    qr = quadrature_residual(h, s, maxc = maxc)
    error += quadrature_error(h, s, m, ys_split, maxc = maxc, qr = qr)
    
    return error

def compute_step_error_split(hs, range_s, range_m, maxp, use_max = True, overline_xs_split = overline_xs_split, ys_split = ys_split):
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
    for (s, m) in tqdm(zip(range_s, range_m), desc = 's, m'):
        if s not in step_error.keys():
            step_error[s] = {}
        step_error[s][m] = {}
        for h in hs:
            step_error[s][m][h] = float(error_sum_CFsplit(h, s, m, overline_xs_split, ys_split, maxc = 1, maxp = maxp, use_max = use_max))
    return step_error

def minimize_cost_CFMagnus_split(hs, s, m, total_time, total_error, step_error):
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
    for h in step_error[s][m].keys():
        cost_exponentials[h] = total_time*m/h
        #if trotter_exponentials: # No need to Trotterize the exponentials in the split-operator case
        #    cost_exponentials[h] *= 2 * 5**(s-1)
        errors[h] = total_time*step_error[s][m][h]/h

    min_cost = np.inf
    for h in step_error[s][m].keys():
        if errors[h] < total_error and cost_exponentials[h] < min_cost:
            min_cost_h = h
            min_cost = cost_exponentials[h]

    return min_cost_h, min_cost


############## Trotter error ####################

def error_sum_trotter(h, s, maxc = 1, maxp = 40, n = None):
    r"""
    Error from using Trotter product formulas of arbitrary order, without the corresponding Magnus expansion.

    Parameters
    ----------
    h: step size
    s: 2s is the order of the Trotter product formula
    maxc: maximum value of the norm of the Hamiltonian
    n: number of spins
    """
    error = np.longdouble(0)

    # First, we add the error from the Taylor truncation of Omega
    s0 = 1
    p = 2*s0+1 # We are truncating the Magnus expansion at order 2s+1
    omega_error = exp_Omega_bound(h, p, s, maxc, factorial)
    last_correction = omega_error
    while last_correction/omega_error > 1e-5:
        p += 1
        if p > maxp:
            raise ValueError('The error is not converging')
        last_correction = exp_Omega_bound(h, p, s, maxc, factorial)
        omega_error += last_correction
    error += omega_error

    # Error from the Trotter product formula
    error += trotter_error_spu_formula(n, h/(4*n), s, u = maxc)

    return error

def compute_trotter_step_error(hs, range_s, maxp, total_time_list, use_max = True):
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
    trotter_error = {}
    for n in tqdm(total_time_list, desc = 'time'):  
        trotter_error[n] = {}
        for s in tqdm(range_s, desc = 's'):
            if s not in trotter_error[n].keys():
                trotter_error[n][s] = {}
            for h in hs:
                trotter_error[n][s][h] = float(error_sum_trotter(h, s, maxc = 1, maxp = maxp, n = n))
    return trotter_error

def minimize_cost_trotter(hs, s, total_time, total_error, step_error, trotter_exponentials = True, splits = 2):
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
        m = 1
        cost_exponentials[h] = total_time*m/h
        if trotter_exponentials: 
            cost_exponentials[h] *= 2 * 5**(s-1) * splits
        errors[h] = total_time*step_error[total_error][total_time][s][h]/h

    min_cost = np.inf
    for h in hs:
        if errors[h] < total_error and cost_exponentials[h] < min_cost:
            min_cost_h = h
            min_cost = cost_exponentials[h]

    return min_cost_h, min_cost
