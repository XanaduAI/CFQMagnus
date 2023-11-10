#### Functions to compute the error terms in the Magnus expansion

import scipy
import numpy as np
from sympy.utilities.iterables import multiset_partitions, partitions
from itertools import permutations
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax.numpy as jnp
import jax
import math
from tqdm import tqdm
from sympy.matrices import Matrix
import sympy as sp



################ Bounding \Psi_m^{[2s]} ########################

def efficient_number_compositions(p: int, s: int, factorial: dict) -> dict:
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
    pts = partitions(p, k=s)
    
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


def efficient_weak_compositions(maxw: int, list_m: list, list_cs, factorial: dict, use_max: bool = True):
    r'''
    Computes sum{k_1...k_m\in weak_compositions(maxw)} c^{k_1}/k_1!...c^{k_m}/k_m!) for all weak compositions of length m.
    Depending on the value of `use_max`, it uses either the maximum value of c for all at once (faster) or the specific values of c,
    (more accurate).

    Parameters
    ----------
    maxw: normw is the sum value of the weak composition.
        maxw is the maximum value of normw. The result with be computed for range(maxw+1)
    list_m: list of the possible values of m.
    list_cs: correspondingly, list of the values of c.
    use_max: 
        if True, use the maximum value of cs instead of the specific values of cs. 
        Faster but less tight bound will be returned.
    factorial: a dictionary with the factorial of all the numbers up to maxw or above.

    Returns
    -------
    weak_comps: a dictionary with the sum for each m, for each normw.
    '''

    weak_comps = {}
    for m, cs in zip(list_m, list_cs):

        vector_add_term = np.vectorize(lambda part: add_term(m, mean_cs, part))

        max_cs = np.max(cs)
        mean_cs = np.mean(cs, axis = 1)

        weak_comps[m] = {}
        for normw in range(maxw+1):
            
            suma = np.longdouble(0)

            pts = partitions(normw, m=m)

            if use_max:
                for part in pts:
                    len_ = np.sum(list(part.values()))

                    num_comps = factorial[m]/factorial[m-len_] * max_cs**len_
                    for n in part.values():
                        num_comps /= factorial[n]

                    denominator = np.longdouble(1)
                    for i in part.keys():
                        denominator = denominator*factorial[i]

                    suma = suma + num_comps/denominator

            else:
                add = vector_add_term(list(pts))
                suma += np.sum(add)

            weak_comps[m][normw] = suma

    return weak_comps


def summatory_compositions(h: float, maxp: int, s: int, m: int, cs: list, use_max: bool = True):
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
    cs: list of values |x_{i,j}a_j| for each i,j
    use_max: whether to use a faster but less tight bound (True) or a slower but tighter bound (False)

    Returns
    -------
    suma: a dictionary with the sum for each p.
    '''

    weak_comps = efficient_weak_compositions(maxp, [m], [cs], use_max)

    suma = {}
    for p in range(1, maxp+1):
        num_comps = efficient_number_compositions(p, s)

        suma[p] = np.longdouble(0)
        for len_comp, nc in num_comps.items():
            suma[p] += nc*weak_comps[m][len_comp] * h**p

    return suma


############## Bounding \|\Omega(h)\| ###########################

def Omega_bound(h: float, p: int, maxc: float, s: int = None):
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
    for part in partitions(p, m=2*s):
        prod = np.longdouble(1)
        dim = np.sum(list(part.values()))
        for k, v in part.items():
            prod = prod * maxc**v / k**v
        prod = prod / dim
        suma = suma + prod

    return suma * (h/2)**p


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

    # We first generate all partitions of p into z parts of size up to 2s
    pts = list(partitions(p, k=2*s))

    # For each possible partition,
    for part in pts:
        large_comb_size = np.sum(list(part.values()))
        num_combs_large = factorial[large_comb_size]/np.prod([factorial[v] for v in part.values()])


        # We further generate more partitions of each part into up to 2s parts
        product = np.longdouble(1)
        for k in part.keys(): # Iterating over the dictionary of a big partition
            suma = np.longdouble(0)
            js = partitions(k, m = 2*s)
            for j in js: # Here we get a dictionary of small partitions
                size = np.sum(list(j.values()))
                num_small_combs = factorial[size]/np.prod([factorial[jv] for jv in j.values()])
                term = maxc**size / size 
                term = term / np.prod([jk**jv for jk, jv in j.items()])
                suma = suma + term * num_small_combs

            product = product * suma

        bound += num_combs_large/factorial[large_comb_size] * product

    assert(bound/2**(p) <= 1)

    return bound * (h/2)**p


############## Quadrature error ###########################

def quadrature_residual(h, s, m, maxc = 1):
    r'''
    Computes the error of the quadrature rule for the Magnus expansion of order 2s+1.

    Parameters
    ----------
    h: step size
    s: 2s is the order of the Magnus expansion
    m: number of exponentials in the Commutator Free Magnus operator
    maxc: maximum value of the norm |a_j|

    Variables used
    --------------
    n: order of the quadrature rule (2s)
    i: index of the univariate integral
    j: missing terms in the univariate integral, that constitute the error term
    '''
    quadrature_residual = {}
    for i in range(1,m+1):
        quadrature_residual[i-1] = {}
        n = 2*s
        partial_sums = []
        for j in range(0, 350):
            term = math.factorial(2*n+j)/math.factorial(j)*(1-(-1)**(i+j))/((i+j+2*n)*2**(2*n+j+i))* maxc * (h/2)**(j)
            partial_sums.append(term + partial_sums[-1] if len(partial_sums) > 0 else term)

        partial_sums = np.array(partial_sums)
        partial_sums *= (math.factorial(n)/math.factorial(2*n))**3 * math.factorial(n)/(2*n+1) * h**(2*n+1)

        quadrature_residual[i-1] = partial_sums[-1]

    return quadrature_residual

def quadrature_error(h, s, m, cs_y, maxc = 1, qr = None):
    r'''
    Computes the error of the quadrature rule error of order s,
    for the commutator free Magnus expansion of order 2s+1.

    Parameters
    ----------
    h: step size
    s: 2s is the order of the Magnus expansion
    m: number of exponentials in the Commutator Free Magnus operator
    cs_y: list of the coefficients y_{i,j} of the Magnus expansion
    maxc: maximum value of the norm |a_j|
    '''

    if qr is None:
        qr = quadrature_residual(h, s, m, maxc = maxc)

    error = 0
    for i in range(m):
        for t in range(s):
            error += qr[i] * abs(cs_y[(s,m)][i][t])

    return error

############## Basis change error ###########################

def basis_change_error(h, s, m, cs_y, maxc = 1):
    r"""
    Computes the error of the basis change of the Magnus expansion of order 2s+1.

    Parameters
    ----------
    h: step size
    s: n = 2s is order of the Magnus expansion
    m: number of exponentials in the Commutator Free Magnus operator
    maxc: maximum value of the norm of the Hamiltonian and its derivatives
    cs_y: list of the values of the quadrature rule
    """

    basis_change_error = 0

    for k in range(1, m+1):
        for j in range(s):
            basis_change_error += abs(cs_y[(s,m)][k-1][j])/2**j * (2*maxc)/(2*s+j+1)

    return basis_change_error * (h/2)**(2*s+1) / (1-h/2)


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
        error += n * (2*k-1)**p * (k*u) * (2*u)**p * (2*k-2)**p * h**(2*p+1) / math.factorial(p+1)
        error += n * (2*k+1)**p * (k*u) * (2*u)**p * (2*k)**p * h**(2*p+1) / math.factorial(p+1)

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
def generate_x(n, y):
    r'''
    n: size of the vector
    x: vector to be multiplied by T

    Returns T@x
    '''
    assert(len(y) == n)
    T = generate_T(n)
    return T@y

def generate_y(n, x):
    r'''
    n: size of the vector
    y: vector to be multiplied by R

    Returns R@y
    '''
    assert(len(x) == n)
    R = generate_T(n).inv()
    return R@x