from numpy import linalg as la
import numpy
import sympy
import sys

P_dir = '/Users/gil/Google Drive/repos/potapov/potapov_interpolation/Potapov_Code/'
sys.path.append(P_dir)

from functions import limit

def double_up(P, M, second_row_minus = False):
    """
    Generate a 'doubled-up' version of the positive (P) and negative (M) matrices.
    """
    r1 = M.col_join(P)
    r2 = (P.conjugate()).col_join(M.conjugate())
    if second_row_minus:
        r2 *= -1
    return r1.row_join(r2)

def J(n):
    """
    Generate the signature matrix of half-dimension n.

    I.e. J = [[I_n, 0],
              [0, -I_n]]
    """
    r1 = sympy.eye(n).col_join(sympy.zeros(n))
    r2 = sympy.zeros(n).col_join(-sympy.eye(n))
    return r1.row_join(r2)

def flat(M):
    """
    return J M^\dag J for the appropriate dimension of J.
    """
    m,n = M.shape
    if m%2 != 0 or n%2 != 0:
        raise ValueError("dimensions of matrix should be even.")
    return J(n/2)*M.H*J(m/2)

def make_ABCD(S, Phi, Omega):
    """
    Generate the ABCD matrices from the S, Phi, and Omega
    """
    D = double_up(sympy.zeros(S.shape[0]),S)
    B = -flat(Phi)*D
    A = -0.5*flat(Phi)*Phi - 1j*Omega
    C = Phi
    return A, B, C, D


def make_T(A, B, C, D, z):
    """
    Make the transfer function using the ABCD matrices.
    """
    IA = sympy.eye(A.shape[0])
    T_z = D + C*(z*IA - A).inv()*B
    return T_z


def scale_vector_doubled_up(v):
    """Given a vector v, attempt to scale it so that
    it has doubled-up form, u = Sigma*u^\#.

    Raises: ValueError
    """
    sub_sigma = numpy.diag([1]*(v.shape[0]/2))
    M = v.shape[0]/2
    Sigma = numpy.vstack((
                numpy.hstack((numpy.zeros((M,M)),sub_sigma)),
                numpy.hstack((sub_sigma,numpy.zeros((M,M))))
            ))
    angles = numpy.angle((Sigma*v).conj() / v)
    pos_angles = [a if a > 0 else a+2*numpy.pi for a in angles]
    u = v*numpy.exp(0.5j*pos_angles[0])
    if la.norm(u - Sigma*u.conj()) > 1e-5:
        raise ValueError("Vector cannot be scaled to doubled-up form.")
    return u

def real_scaling(v1, v2):
    """
    Scale the two vectors v1 and v2 appropriately, granted they
    correspond to two real poles.
    """
    M = v1.shape[0]/2
    Jv = numpy.matrix(numpy.diag([1 for i in range(M)]+[-1 for i in range(M)]))
    N = v1.H*Jv*v2
    u1 = v1 / la.norm(numpy.sqrt(2*N))
    u2 = v2 / la.norm(numpy.sqrt(2*N))
    return u1, u2


def make_Sigma(M, conv="doubled_up"):
    """
    Generate the permutation matrix Sigma of half-dimension M.

    I.e. Sigma = [[0, I_M],
                  [I_M, 0]]
    """
    sub_sigma = numpy.diag([1]*M)
    Sigma = numpy.vstack((
                numpy.hstack((numpy.zeros((M,M)),sub_sigma)),
                numpy.hstack((sub_sigma,numpy.zeros((M,M))))
            ))
    if conv == "doubled_up":
        pass
    elif conv == "consecutive":
        P_conv = permutation_conv_matrix(M)
        Sigma = P_conv*Sigma*P_conv.T
    else:
        raise ValueError("Unknown convention %s." %conv)

    return Sigma

def check_J_unitary(T, z, eps = 1e-5, print_norm = False, conv="doubled_up"):
    """
    Check if the transfer function T is J-unitary at z.

    Args:
        T (matrix-valued function): transfer function to check
        z (complex number): location to check
        eps (float): tolerance level
        print_norm (bool): whether or not to print norm
        conv (string): Which J convention to use.
            "doubled_up" means [1,1,...,1,-1,-1,...,-1],
            "consecutive" means [1,-1,1,-1,...,1,-1].
    Returns:
        is_j_unitary (bool): Whether or not T is J-unitary at z.
    """
    T_mat1 = numpy.matrix(T(z))
    T_mat2 = numpy.matrix(T(-numpy.conj(z)))
    d = T_mat1.shape[0]/2
    Jv = make_Jv(d, conv)
    norm = la.norm(T_mat1*Jv*T_mat2.H - Jv)
    if print_norm:
        print(norm)
    is_j_unitary = norm < eps
    return is_j_unitary

def permutation_conv_matrix(dim):
    """Get permutation matrix to convert from doubled-up to consecutive convention.

    Args:
        dim (int): Dimensionality, half of the doubled-up space.
    Returns:
        permutation matrix to go from doubled-up to consecutive convention
        (i.e. from (1,1,...,1,-1,-1,...,-1) to (1,-1,1,-1,...,1,-1) ).
    """
    P_conv = numpy.matrix(numpy.zeros((2*dim, 2*dim)))
    p_conv_list = numpy.concatenate([[i,i+dim] for i in range(dim)])
    for i in range(2*dim):
        P_conv[i, p_conv_list[i]] = 1
    return P_conv


def check_doubled_up_func(T, z, eps = 1e-5, conv="doubled_up"):
    """
    Check if the transfer function T is doubled-up at z.
    """
    T_mat = numpy.matrix(T(z))
    dim = T_mat.shape[0]/2
    Sigma = make_Sigma(dim, conv=conv)
    return la.norm(Sigma * T(numpy.conj(z)).H * Sigma - T(z)) < eps


def check_doubled_up_mat(M, conv="doubled_up", eps = 1e-5):
    """
    Check if the transfer function T is doubled-up at z.
    """
    d1, d2 = M.shape[0]/2, M.shape[1]/2
    Sigma1 = make_Sigma(d1, conv=conv)
    Sigma2 = make_Sigma(d2, conv=conv)

    return la.norm( (Sigma1 * M * Sigma2).conj() - M) < eps


def purge(lst, eps=1e-5, max_norm=None):
    """
    Gets rid of redundant elements up to error eps.
    """
    if len(lst) <= 1:
        return lst

    new_lst = []
    for lst_i in lst:
        if all([abs(el - lst_i) > eps
                for el in new_lst]):
            new_lst.append(lst_i)
    if max_norm is None:
        return new_lst
    else:
        return [el for el in new_lst if la.norm(el) < max_norm]

def make_Jv(M, conv="double_up"):
    if conv == "doubled_up":
        Jv = numpy.matrix(numpy.diag([1]*M+[-1]*M))
    elif conv == "consecutive":
        Jv = numpy.matrix(numpy.diag([1,-1]*M))
    else:
        raise ValueError("Unknown value for conv: %s." %conv)
    return Jv

def complex_prod_deg(z, poles, vecs, dim, eps=1e-3, verbose=False, conv="doubled_up"):
    """
    Generate a complex product given complex poles and vectors,
    and evaluate at $z$.

    In the example, the complex eigenvectors were all degenerate,
    so we perturb the first component by $\eps$. In this case we
    actually need two terms to approximate the degenerate space.
    """
    M = int(dim/2)
    R = numpy.matrix(numpy.eye(dim))
    J1 = numpy.matrix(numpy.diag([1, -1]))
    Jv = make_Jv(M, conv)
    Sigma = make_Sigma(M, conv)

    for p, eig in zip(poles, vecs):
        v1, val = eig
        if verbose:
            print "Unperturbed vector is: \n", v1
        v1[0,0] += eps ## In case of degenerate vectors, set eps != 0.

        v1 = v1 / numpy.sqrt(v1.H*Jv*v1)
        if v1.H*Jv*v1 > 0:
            V1 = numpy.hstack([v1, Sigma*v1.conj()])
        else: ## flipping v and \Sigma *v^# changes the normalization sign
            V1 = numpy.hstack([Sigma*v1.conj(), v1])
        V1_flat = J1*V1.H*Jv

        F1 = numpy.matrix([[(z+p.conj())/(z-p),0],
                            [0,(z+p)/(z-p.conj())]])

        F2 = numpy.matrix([[(z+p)/(z-p.conj()),0],
                           [0,(z+p.conj())/(z-p)]])

        I = numpy.matrix(numpy.eye(dim))
        R = R * (I -V1*V1_flat+ V1*F1*V1_flat)*(I -V1*V1_flat+ V1*F2*V1_flat)
    return R

def factorize_complex_poles(poles, T_tilde, verbose=False, conv="doubled_up"):
    """
    Find the vectors at a given list of complex poles.
    """
    dim = T_tilde(0).shape[0]
    found_vecs = []
    for p in poles:

        R = complex_prod_deg(p, poles, found_vecs, dim, verbose=verbose, conv=conv)
        L = la.inv(R) * limit(lambda z: (z-p)*T_tilde(z),p)
        [eigvals,eigvecs] = la.eig(L)
        if verbose:
            print("Eigenvalues: ")
            print(eigvals)
            print ("eigenvectors")
            print(eigvecs)
        index = numpy.argmax(map(abs,eigvals))
        big_vec = numpy.asmatrix(eigvecs[:,index])
        big_val = eigvals[index]
        found_vecs.append((big_vec,big_val))
    return found_vecs

def factorize_real_poles(p1, p2, T_tilde, conv="doubled_up"):
    """
    Obtain a factor from the two real non-degenerate poles p1 and p2.

    Return a function that can be evaluated at complex values $z$.
    """
    vecs = get_Potapov_vecs(T_tilde, [p1, p2])
    u1, u2 = vecs
    w1 = scale_vector_doubled_up(u1)
    w2 = scale_vector_doubled_up(u2)
    v1, v2 = real_scaling(w1, w2)
    M = v1.shape[0]/2
    Jv = make_Jv(M, conv)
    U = numpy.matrix([[1,1],[-1j,1j]])
    V = numpy.hstack([v1,v2])*U
    V_flat = JA*V.H*Jv

    if (V*V_flat)[0,0] < 0:
        v1, v2 = v2, v1
        p1, p2 = p2, p1
        V = numpy.hstack([v1,v2])*U
        V_flat = JA*V.H*Jv

    F1 = lambda z: (numpy.matrix([[(z+p2)/(z-p1),0],
                                 [0,(z+p1)/(z-p2)]])
                    )


    I =  numpy.matrix(numpy.eye(v1.shape[0]))
    return lambda z: (I - V*V_flat + V*la.inv(U)*F1(z)*(U)*V_flat)

def factorize_deg_real_pole(pole, T):
    """
    Function for factorizing a degenerate real pole.

    It is similar to the non-degenerate real case, but the same pole appears twice.
    """

    L = (limit(lambda z: (z-pole)*T(z),pole))
    _, vecs = la.eig(L)
    u1 = vecs[:,0]
    u2 = vecs[:,1]
    w1 = scale_vector_doubled_up(u1)
    w2 = scale_vector_doubled_up(u2)
    v2, v1 = real_scaling(w1, w2)
    M = v1.shape[0]/2
    Jv = numpy.matrix(numpy.diag([1 for i in range(M)]+[-1 for i in range(M)]))

    U = numpy.matrix([[1,1],[-1j,1j]])
    V = numpy.hstack([v1,v2])*U

    V_flat = JA*V.H*Jv

    if (V*V_flat)[0,0] < 0:
        v1, v2 = v2, v1
        V = numpy.hstack([v1,v2])*U
        V_flat = JA*V.H*Jv


    F1 = lambda z: (numpy.matrix([[(z+pole)/(z-pole),0],
                                 [0,(z+pole)/(z-pole)]])
                    )

    return lambda z: numpy.matrix(numpy.eye(2)) - V*V_flat + V*F1(z)*V_flat
