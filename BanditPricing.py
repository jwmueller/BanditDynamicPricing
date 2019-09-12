""" Bandit methods for dynamic pricing with low-rank demand variation """

from __future__ import division
import numpy as np


def GDG(R_prev, eta, delta, s_radius, prev_state):
    """ GDG bandit method of Flaxman and Kalai for choosing next price.
        
        Args:
            R_prev (float): Previously observed negative revenue at the prices p_tilde chosen by this method in the last round.  
            eta (float): Positive value. 
            delta (float): Positive value. Recall, if revenue is bounded by B and L-Lipshitz and r = s_radius, 
                           we want to set: eta = r/(B*sqrt(T)), delta = T^(-1/4) * sqrt((BNr^2)/(3*(Lr + B))), 
                           alpha parameter is automatically set = delta/r inside this function.
            s_radius (float): Postive radius of constraint-set, which must be centered Euclidean ball.
            prev_state (tuple): Internal bandit state from previous rounds that is required for this round 
                                (was returned as next_state in previous round).
                                In first round, we want: prev_state = (initial_price, random_unit-vector)
    """
    alpha = delta/s_radius
    if (alpha <= 0) or (alpha > 1) or (eta <= 0) or (delta <= 0):
        raise ValueError("eta, delta, or alpha invalid")
    p_prev_clean, xi_prev = prev_state
    N = p_prev_clean.shape[0]
    p_next_clean = p_prev_clean - eta * R_prev * xi_prev
    next_norm = np.linalg.norm(p_next_clean)
    if next_norm > (1-alpha)*s_radius: # project into interior of S
        p_next_clean = (1-alpha)*s_radius * (p_next_clean/next_norm)
    xi_next = randUnitVector(N)
    p_tilde = p_next_clean + delta*xi_next
    if np.linalg.norm(p_tilde) > s_radius:
        raise ValueError("constraints violated, norm(p_tilde)="+str(np.linalg.norm(p_tilde)))
    next_state = (p_next_clean, xi_next)
    return( (p_tilde, next_state) )

def OPOK(R_prev, eta, delta, s_radius, U, prev_state):
    """ Low-rank dynamic pricing with *Known* product features.
        Here, FindPrice is regularized toward 0, the center of the ball of feasible prices.
        
        Args:
            R_prev (float): Previously observed negative revenue at the prices p_tilde chosen by this method in the last round.  
            eta (float): Positive value. 
            delta (float): Positive value. Recall, if revenue is bounded by B and L-Lipshitz and r = s_radius, 
                           we want to set: eta = r/(B*sqrt(T)), delta = T^(-1/4) * sqrt((BNr^2)/(3*(Lr + B))), 
                           alpha parameter is automatically set = delta/r inside this function.
            s_radius (float): Postive radius of constraint-set, which must be centered Euclidean ball.
            U (2D array): N x d matrix of given product features (must be orthonormal).
            prev_state (tuple): Internal bandit state from previous rounds that is required for this round 
                                (was returned as next_state in previous round).
                                In first round, we want: prev_state = (initial_price, random_unit-vector)
    """
    alpha = delta/s_radius
    if (alpha <= 0) or (alpha > 1) or (eta <= 0) or (delta <= 0):
        raise ValueError("eta, delta, or alpha invalid")
    x_prev_clean, xi_prev, p_prev = prev_state
    d = x_prev_clean.shape[0]
    x_next_clean = x_prev_clean - eta * R_prev * xi_prev
    """ 
    # If U is non-orthonormal, need to use the following U projection instead:
    pinv_U = np.linalg.pinv(U.transpose())
    A = np.dot(pinv_U.transpose(),pinv_U)/((s_radius**2)*(1-alpha)**2)
    x_next_clean = ellipsoidProjection(x_next_clean, A) # project into interior of U^T(S)
    """
    xnorm = np.linalg.norm(x_next_clean)
    if xnorm > (1-alpha)*s_radius:
        x_next_clean = (1-alpha)*s_radius*(x_next_clean/xnorm)
    xi_next = randUnitVector(d)
    # print(np.linalg.norm(p_next_clean) + np.linalg.norm(delta*xi_next))
    # print(np.linalg.norm(p_next_clean + delta*xi_next))
    x_tilde = x_next_clean + delta*xi_next
    # p_tilde = findPriceOptim(x_tilde, U, s_radius, p_prev, max_iter = 1000)
    p_tilde = findPrice(x_tilde, U)
    if np.linalg.norm(p_tilde) > s_radius:
        raise ValueError("constraints violated, norm(p_tilde)="+str(np.linalg.norm(p_tilde)))
    next_state = (x_next_clean, xi_next, p_tilde)
    return( (p_tilde, next_state) )

def OPOKprevprice(R_prev, eta, delta, s_radius, U, prev_state):
    """ Low-rank dynamic pricing with *Known* product features.
        Here, FindPrice is regularized toward previous price rather than 0 for less pricing variation.
        
        Args:
            R_prev (float): Previously observed negative revenue at the prices p_tilde chosen by this method in the last round.  
            eta (float): Positive value. 
            delta (float): Positive value. Recall, if revenue is bounded by B and L-Lipshitz and r = s_radius, 
                           we want to set: eta = r/(B*sqrt(T)), delta = T^(-1/4) * sqrt((BNr^2)/(3*(Lr + B))), 
                           alpha parameter is automatically set = delta/r inside this function.
            s_radius (float): Postive radius of constraint-set, which must be centered Euclidean ball.
            U (2D array): N x d matrix of given product features (must be orthonormal).
            prev_state (tuple): Internal bandit state from previous rounds that is required for this round 
                                (was returned as next_state in previous round).
                                In first round, we want: prev_state = (initial_price, random_unit-vector)
    """
    alpha = delta/s_radius
    if (alpha <= 0) or (alpha > 1) or (eta <= 0) or (delta <= 0):
        raise ValueError("eta, delta, or alpha invalid")
    x_prev_clean, xi_prev, p_prev = prev_state
    d = x_prev_clean.shape[0]
    x_next_clean = x_prev_clean - eta * R_prev * xi_prev  # approximate gradient step.
    """ 
    # If U is non-orthonormal, need to use the following U projection instead:
    pinv_U = np.linalg.pinv(U.transpose())
    A = np.dot(pinv_U.transpose(),pinv_U)/((s_radius**2)*(1-alpha)**2)
    x_next_clean = ellipsoidProjection(x_next_clean, A) # project into interior of U^T(S)
    """
    xnorm = np.linalg.norm(x_next_clean)
    if xnorm > (1-alpha)*s_radius:
        x_next_clean = (1-alpha)*s_radius*(x_next_clean/xnorm)
    xi_next = randUnitVector(d)
    x_tilde = x_next_clean + delta*xi_next
    p_tilde = findPriceOptim(x_tilde, U, s_radius, p_prev, max_iter = 1000)
    if np.linalg.norm(p_tilde) > s_radius:
        raise ValueError("constraints violated, norm(p_tilde)="+str(np.linalg.norm(p_tilde)))
    next_state = (x_next_clean, xi_next, p_tilde)
    return( (p_tilde, next_state) )

def OPOL(demands_prev, eta, delta, s_radius, prev_state):
    """ Low-rank dynamic pricing with *Unknown* product features (latent U is assumed orthonormal).
    
    Args:
            demands_prev (1D array): Observed product demands at the prices p_tilde chosen by this method in the last round.
                                     Is used to calculate: R_prev = -p_tilde * demands_prev
            eta (float): Positive value. 
            delta (float): Positive value. Recall, if revenue is bounded by B and L-Lipshitz and r = s_radius, 
                           we want to set: eta = r/(B*sqrt(T)), delta = T^(-1/4) * sqrt((BNr^2)/(3*(Lr + B))), 
                           alpha parameter is automatically set = delta/r inside this function.
            s_radius (float): Postive radius of constraint-set, which must be centered Euclidean ball.
            prev_state (tuple): Internal bandit state from previous rounds that is required for this round 
                                (was returned as next_state in previous round).
                                In first round, we want: prev_state = (initial_price, random_unit-vector)
    """
    alpha = delta/s_radius
    if (alpha <= 0) or (alpha > 1) or (eta <= 0) or (delta <= 0):
        raise ValueError("eta, delta, or alpha invalid")
    x_prev_clean, Q, t, update_cnts, xi_prev, p_prev = prev_state
    d = x_prev_clean.shape[0]
    N = p_prev.shape[0]
    col_ind = t % d
    cnt_ind = update_cnts[col_ind] + 1 # num times this column has been updated
    update_cnts[col_ind] = cnt_ind
    Q[:,col_ind] = (1.0/cnt_ind)*demands_prev + ((cnt_ind-1)/cnt_ind)*Q[:,col_ind]
    if np.min(np.sum(np.abs(Q),axis=0)) == 0.0: 
        # while Q contains zero column (first d rounds), simply select next price randomly:
        p_rand = p_prev + delta*randUnitVector(N)/10.0
        if np.linalg.norm(p_rand) > s_radius:
            p_rand = s_radius * (p_rand/np.linalg.norm(p_rand))
        Uhat, singval, right_singvec = np.linalg.svd(Q, full_matrices=False) 
        x_prev_clean = np.dot(Uhat.transpose(), p_prev) # first low-dimensional action.
        next_state = (x_prev_clean, Q, t+1, update_cnts, xi_prev, p_prev)
        return( (p_rand, next_state) )
    # Otherwise run our algorithm:
    Uhat, singval, right_singvec = np.linalg.svd(Q, full_matrices=False) # Update product-feature estimates. 
    R_prev = negRevenue(p_prev, demands_prev)
    x_next_clean = x_prev_clean - eta * R_prev * xi_prev  # approximate gradient step.
    # Project into (1-alpha)*U^T(S) when U is orthnormal, S = ball:
    xnorm = np.linalg.norm(x_next_clean)
    if xnorm > (1-alpha)*s_radius:
        x_next_clean = (1-alpha)*s_radius*(x_next_clean/xnorm)
    xi_next = randUnitVector(d)
    x_tilde = x_next_clean + delta*xi_next
    p_tilde = findPrice(x_tilde, Uhat)
    if np.linalg.norm(p_tilde) > s_radius:
        raise ValueError("constraints violated, norm(p_tilde)="+str(np.linalg.norm(p_tilde)))
    next_state = (x_next_clean, Q, t+1, update_cnts, xi_next, p_tilde)
    return( (p_tilde, next_state) )

def exploreExploitPricing(R_prev, T, s_radius, prev_state):
    """ Sets uniformly random price for each of the first T^(2/3) rounds.
        Then fixes prices at the best observed configuration so far for the remaining rounds.
        
        Args:
            R_prev (float): Previously observed negative revenue at the prices p_tilde chosen by this method in the last round.  
            T (int): Total number of pricing rounds.
            s_radius (float): Postive radius of constraint-set, which must be centered Euclidean ball.
            prev_state (tuple): Internal algorithm state from previous rounds that is required for this round 
                                (was returned as next_state in previous round).
                                In first round, define prev_state = (np.zeros(N), np.inf, 0, np.zeros(N))
    """
    p_best, R_best, t, p_prev = prev_state
    if (t > T) or (t < 0):
        raise ValueError("invalid t")
    thres = np.power(T, 3.0/4.0) # np.power(T, 2.0/3.0)
    if t < thres: # explore
        p_next = randomPricing(p_prev.shape[0], s_radius)
        if R_prev < R_best:
            R_best = R_prev
            p_best = p_prev
    else: # exploit
        p_next = p_best
    next_state = (p_best, R_best, t+1, p_next)
    return(p_next, next_state)


def firstOPOLstate(d, p_init):
    """ Produces the initial state for the OPOL algorithm (prev_state for first round).
    
        Args:
            d (int): Rank of the product features to be learned.
            p_init (1D array): Initial chosen price-vector.
    """
    N = p_init.shape[0]
    Q = np.zeros((N,d))
    x0 = np.zeros(d)
    t0 = 0
    update_cnts = np.zeros(d)
    xi0 = randUnitVector(d)
    return( (x0, Q, t0, update_cnts, xi0, p_init) )

def findPrice(x, U):
    p_next = np.dot(U,x)
    if np.linalg.norm(np.dot(U.transpose(),p_next)-x) > 1e-4:
        raise ValueError(('invalid price: ', p_next))
    return( p_next )

def findPricePinv(x, U, s_radius):
    """ When set of feasible prices = ball of radius s_radius and  U is orthonormal, 
        leverage pseudoinverse property of Transpose to find next price (which will be regularized toward 0).
    """
    Ut = U.transpose()
    pinv_U = np.linalg.pinv(Ut)
    p_next = np.dot(pinv_U, x)
    if np.linalg.norm(np.dot(Ut,p_next)-x) > 1e-4 or np.linalg.norm(p_next) > s_radius:
        raise ValueError(('invalid price: ', p_next))
    return( p_next )

def randomPricing(N, s_radius):
    """ Returns Uniformly random prices from constraint-set S, when S = ball.
        
        Args:
            N (int): Number of products.
            s_radius: radius of ball S.
    """
    rand_mag = np.random.uniform(low=0.0, high=s_radius) # random magnitude
    return( rand_mag * randUnitVector(N) )

def randUnitVector(dimension):
    xi = np.random.normal(size=dimension)
    return( xi / np.linalg.norm(xi) )

def negRevenue(p, q):
    return( -np.dot(p,q) )

def ellipsoidProjection(y, A, tol = 1e-8): 
    """ Projects given point y into {x: x^T A x < 1}. 
        Only needed for FindPrice when U is not orthogonal.
        Based on implementation from: https://mathproblems123.wordpress.com/2013/10/17/distance-from-a-point-to-an-ellipsoid/
    """
    Aog = A
    if np.dot(y, np.dot(A,y)) <= 1:
        return( y )
    # first diagonalize A:
    eigvals,V = np.linalg.eigh(A) # A = V*D*V^(-1), where V^(-1) = V^T.
    A = np.diag(eigvals)
    y = np.dot(V.transpose(),y) # Change coordinate-system via V^T to work with the new A
    # Find Lagrange multiplier lambda:
    def f(lam,y,eigvals):
        return( np.sum(np.multiply(eigvals,np.square(np.divide(y,lam*eigvals+1)))) - 1 )
    def deriv(lam,y,eigvals):
        return( -2*np.sum(np.divide(np.multiply(np.square(eigvals),np.square(y)), np.power(lam*eigvals+1,3))) )
    x0=0.0
    x1=0.1
    while np.abs(x1-x0) > tol:
        x0 = x1
        x1 = x0-f(x0,y,eigvals)/deriv(x0,y,eigvals)
    lam = x1
    x = np.dot(np.linalg.pinv(np.identity(len(eigvals))+lam*A),y) # projected point.
    x_star = np.dot(np.linalg.pinv(V.transpose()),x) # undo coordinate-transformation.
    viol_og = np.dot(x_star, np.dot(Aog,x_star)) - 1
    viol = viol_og
    if viol > 1e-3:
        raise ValueError('ellipsoidProjection incorrect, violation=' + str(viol))
    while viol > 0.0:
        x_star /= (1 + 10*viol_og)
        viol = np.dot(x_star, np.dot(Aog,x_star)) - 1
    return( x_star )
