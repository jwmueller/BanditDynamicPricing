""" Helper functions used in our simulation study """

from __future__ import division
import numpy as np
from scipy.linalg import orth
from scipy import optimize

from BanditPricing import *

def generateDemands(p, U, z, V, noise_std):
    """ Returns vector of observed demands q (1-D array) 
        for given product-pricing p (1-D array), 
        assuming z_t, V_t are fixed at z (1-D array), V (2-D array).
    """
    N,d = U.shape
    p = p.reshape((N,1))
    z = z.reshape((d,1))
    q = np.dot(U, z) - np.dot(np.dot(np.dot(U,V), U.transpose()), p) + np.random.normal(scale=noise_std, size=N).reshape((N,1))
    return q.flatten()

def categoryFeatures(N, num_categories):
    """ Randomly assigns products to categories.
        Returns U which is indicator of membership in category.
    """
    U = np.zeros((N,num_categories))
    return U

def forcePosDef(V, eigen_threshold=0.0, increment=1e-6):
    """ Keeps adding to diagonal of matrix V 
        until it is strongly positive definite, meaning  
        minimum eigenvalue of (V + V^T)/2 exceeds eigen_threshold.
    """
    Vpd = V
    d = V.shape[0]
    eigvals = np.linalg.eigvals((Vpd + Vpd.transpose())/2.0)
    while (sum(~np.isreal(eigvals))) > 0 or (np.min(eigvals) <= eigen_threshold):
        Vpd[range(d),range(d)] = Vpd[range(d),range(d)] + increment
        eigvals = np.linalg.eigvals((Vpd + Vpd.transpose())/2.0)
        increment *= 1.1
    return Vpd 

def optimalPriceFast(z_list, V_list, U, s_radius, max_iter = 1e4):
    """ Returns tuple of optimal prices p^* chosen in hindsight, and revenue achieved
        if we know z_t (list of 1-D arrays) V_t (list of 2-D arrays), 
        and U (2-D array) is Orthogonal!
        Performs optimization over low-dimensional actions and is therefore fast.
    """
    TOL = 1e-10 # numerical error allowed.
    T = len(z_list)
    z = np.zeros(z_list[0].shape)
    V = np.zeros(V_list[0].shape)
    N = U.shape[0]
    for t in range(T):
        z += z_list[t]
        V += V_list[t]
    z = z/T
    z = z.reshape((z_list[0].shape[0],1))
    V = V/T
    c = np.dot(U, z)
    B = np.dot(U, np.dot(V, U.transpose()))
    # ensure B is positive definite:
    eigvals = np.linalg.eigvals((B + B.transpose())/2.0)
    if (sum(~np.isreal(eigvals))) > 0 or (np.min(eigvals) < -TOL):
        print("Warning: B is not positive definite")
    cons = {'type':'ineq', 'fun': lambda x: s_radius - np.linalg.norm(x), 
             'jac': lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) > TOL else np.zeros(x.shape)}
    res = optimize.minimize(fun=hindsightLowDimObj, 
        x0=np.zeros(U.shape[1]), args = (z,V),
        jac = hindsightLowDimGrad, method = 'SLSQP', 
        constraints=cons, options={'disp':True,'maxiter':max_iter})
    # Note: Set options['disp'] = True to print output of optimization.
    x_star = res['x']
    p_star = np.dot(U, x_star)
    p_norm = np.linalg.norm(p_star)
    if p_norm > s_radius:
        print ("Warning: p_star not in constraints")
        p_star = s_radius * p_star/p_norm
    R_star = hindsightObj(p_star, c, B)*T
    for i in range(1000): # compare with random search to see if optimization worked
        p_rand = randomPricing(N,s_radius)
        R_rand = hindsightObj(p_rand, c, B)*T
        if (R_star - R_rand)/np.abs(R_star + TOL) > 0.001:
            raise ValueError("SLSQP optimization failed, R_star="+
                str(R_star)+ ",  R_rand="+str(R_rand))
        if R_rand < R_star:
            p_star = p_rand
            R_star = R_rand
    return (p_star, R_star)

def optimalPriceSlow(z_list, V_list, U, s_radius, max_iter = 1e4):
    """ Returns tuple of optimal prices p^* chosen in hindsight, and revenue achieved
        if we know z_t (list of 1-D arrays) V_t (list of 2-D arrays), and U (2-D array).
        Performs high-dimensional optimization directly over p and is therefore slow.
    """
    TOL = 1e-10 # numerical error allowed
    T = len(z_list)
    z = np.zeros(z_list[0].shape)
    V = np.zeros(V_list[0].shape)
    N = U.shape[0]
    for t in range(T):
        z += z_list[t]
        V += V_list[t]
    z = z/T
    z = z.reshape((z_list[0].shape[0],1))
    V = V/T
    c = np.dot(U, z)
    B = np.dot(U, np.dot(V, U.transpose()))
    # ensure B is positive definite:
    eigvals = np.linalg.eigvals((B + B.transpose())/2.0)
    if (sum(~np.isreal(eigvals))) > 0 or (np.min(eigvals) < -TOL):
        print("Warning: B is not positive definite, eigvals(B)=", eigvals)
    cons = {'type':'ineq', 'fun': lambda p: s_radius - np.linalg.norm(p), 
             'jac': lambda p: p / np.linalg.norm(p) if np.linalg.norm(p) > TOL else np.zeros(p.shape)}
    res = optimize.minimize(fun=hindsightObj, x0=np.zeros(U.shape[0]),
            args = (c,B), jac = hindsightGrad, 
            method = 'SLSQP', constraints=cons, options={'disp':True,'maxiter':max_iter})
    # Set options['disp'] = True to print output of optimization.
    p_star = res['x']
    p_norm = np.linalg.norm(p_star)
    if p_norm > s_radius:
        print ("Warning: p_star not in constraints")
        p_star = s_radius * p_star/p_norm
    R_star = hindsightObj(p_star, c, B)*T
    """ # compare with random search to verify the optimization worked:
    for i in range(100):
        p_rand = randomPricing(N,s_radius)
        R_rand = hindsightObj(p_rand, c, B)*T
        if (R_star - R_rand)/np.abs(R_star + TOL) > 0.01:
            raise ValueError("SLSQP optimization failed, R_star="+
                str(R_star)+ ",  R_rand="+str(R_rand))
    """
    return (p_star, R_star)

def OrthogonalSparse(N,d):
    """ Creates sparse + nonnegative + orthogonal U """
    U = np.zeros((N,d))
    num_prods_percategory = int(np.ceil(N/d))
    low_index = 0
    for i in range(d):
        col_i = np.zeros(N)
        if low_index+num_prods_percategory <= N:
            col_i[low_index:(low_index+num_prods_percategory)] = 1.0
        else:
            col_i[low_index:N] = 1.0
        col_i = col_i/np.linalg.norm(col_i)
        U[:,i] = col_i
        low_index += num_prods_percategory
    return U 


""" Helper functions for optimalPrice functions """

def hindsightObj(p, c, B):
    p = p.reshape((p.shape[0],1))
    R_val = np.dot(p.transpose(),np.dot(B,p)) - np.dot(p.transpose(),c)
    return R_val[0][0]

def hindsightGrad(p, c, B):
    p = p.reshape((p.shape[0],1))
    return np.dot(B+B.transpose(),p) - c

def hindsightLowDimObj(x, z, V):
    return np.dot(x,np.dot(V,x)) - np.dot(x,z)

def hindsightLowDimGrad(x, z, V):
    x = x.reshape((x.shape[0],1))
    return np.dot(V+V.transpose(),x) - z

def OrthogonalGaussian(N,d):
    """ Draws entries of U from Gaussian and then orthogonalizes """
    U = np.random.normal(size=N*d).reshape((N,d))
    return orth(U)


""" Similar functions for log-linear demand model (misspecified setting) """

def generateLogLinDemands(p, U, z, V, noise_std, s_radius):
    N,d = U.shape
    p = p.reshape((N,1))
    z = z.reshape((d,1))
    q = np.exp( np.dot(U, z) - np.dot(np.dot(np.dot(U,V), U.transpose()), np.log(p + 5*s_radius)) + np.random.normal(scale=noise_std, size=N).reshape((N,1)))
    return q.flatten()

def optimalLogLinPrice(z_list, V_list, U, s_radius, p0 = None, max_iter = 1e4):
    TOL = 1e-10 # numerical error allowed.
    T = len(z_list)
    N = U.shape[0]
    if p0 is None:
        p0 = np.zeros(N)
    cons = {'type':'ineq', 'fun': lambda x: s_radius - np.linalg.norm(x), 
             'jac': lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) > TOL else np.zeros(x.shape)}
    res = optimize.minimize(fun=hindsightLogLinObj, 
        x0=np.dot(U.transpose(),p0), args = (z_list,V_list,U,s_radius), method = 'SLSQP', 
        constraints=cons, options={'disp':True,'maxiter':max_iter})
    # Set options['disp'] = True to print output of optimization.
    x_star = res['x']
    p_star = np.dot(U, x_star)
    p_norm = np.linalg.norm(p_star)
    if p_norm > s_radius:
        print ("Warning: p_star not in constraints")
        p_star = s_radius * p_star/p_norm 
    R_star = hindsightLogLinObj(np.dot(U.transpose(),p_star), z_list, V_list,U,s_radius)*T
    for i in range(1000): # compare with random search to see if optimization worked
        p_rand = randomPricing(N,s_radius)
        R_rand = hindsightLogLinObj(np.dot(U.transpose(),p_rand), z_list, V_list, U, s_radius)*T
        if (R_star - R_rand)/np.abs(R_star + TOL) > 0.001:
            raise ValueError("SLSQP optimization failed, R_star="+
                str(R_star)+ ",  R_rand="+str(R_rand))
        if R_rand < R_star:
            p_star = p_rand
            R_star = R_rand
    return (p_star, R_star)

def hindsightLogLinObj(x, z_list, V_list, U, s_radius):
    p = np.dot(U,x)
    p_norm = np.linalg.norm(p)
    if p_norm > s_radius:
        p = s_radius * p/p_norm 
    tot_demands = np.zeros(len(p))
    T = len(z_list)
    for t in range(T):
        c_t = np.dot(U, z_list[t])
        B_t = np.dot(U, np.dot(V_list[t], U.transpose()))
        tot_demands += np.exp(c_t - np.dot(B_t,np.log(p+5*s_radius)))/T
    return -np.dot(p,tot_demands)

def optimalLogLinPriceShock(z_list, V_list, U, s_radius, shock_times, p0 = None, max_iter = 1e4):
    TOL = 1e-10 # numerical error allowed
    T = len(z_list)
    N = U.shape[0]
    if p0 is None:
        p0 = np.zeros(N)
    cons = {'type':'ineq', 'fun': lambda x: s_radius - np.linalg.norm(x), 
             'jac': lambda x: x / np.linalg.norm(x) if np.linalg.norm(x) > TOL else np.zeros(x.shape)}
    res = optimize.minimize(fun=hindsightLogLinShockObj, 
        x0=np.dot(U.transpose(),p0), args = (z_list,V_list,U,s_radius,shock_times), method = 'SLSQP', 
        constraints=cons, options={'disp':True,'maxiter':max_iter})
    # Set options['disp'] = True to print output of optimization.
    x_star = res['x']
    p_star = np.dot(U, x_star)
    p_norm = np.linalg.norm(p_star)
    if p_norm > s_radius:
        print ("Warning: p_star not in constraints")
        p_star = s_radius * p_star/p_norm 
    R_star = hindsightLogLinShockObj(np.dot(U.transpose(),p_star), z_list, V_list,U,s_radius,shock_times)
    for i in range(1000): # compare with random search to see if optimization worked
        p_rand = randomPricing(N,s_radius)
        R_rand = hindsightLogLinShockObj(np.dot(U.transpose(),p_rand), z_list, V_list, U, s_radius,shock_times)
        if (R_star - R_rand)/np.abs(R_star + TOL) > 0.001:
            raise ValueError("SLSQP optimization failed, R_star="+
                str(R_star)+ ",  R_rand="+str(R_rand))
        if R_rand < R_star:
            p_star = p_rand
            R_star = R_rand
    return (p_star, R_star)

def hindsightLogLinShockObj(x, z_list, V_list, U, s_radius, shock_times):
    p = np.dot(U,x)
    p_norm = np.linalg.norm(p)
    if p_norm > s_radius:
        p = s_radius * p/p_norm 
    tot_demands = np.zeros(len(p))
    t = len(z_list)
    for st in shock_times[::-1]:
        if t > st:
            c_t = np.dot(U, z_list[st])
            B_t = np.dot(U, np.dot(V_list[st], U.transpose()))
            cnt = min(t-st, shock_times[1]-1)
            tot_demands += np.exp(c_t - np.dot(B_t,np.log(p+5*s_radius)))*cnt    
    return -np.dot(p,tot_demands)
