#coding:gb18030
"""
Copyright (c) 2015 Inc. All Rights Reserved
@file: pybursts.py
@author: Fan Tao (fantao@mail.ustc.edu.cn)
@date: 2015/12/08
@desc: kleinberg infinite-state model for bursty;

"""

import numpy as np
import math


def find_bursty_structure(offsets, q, n, N, freq_type=False):
    """return bursty；
    """
    if not freq_type:
        bursts = np.array([np.repeat(np.nan, N), np.repeat(offsets[0], N), 
                       np.repeat(offsets[0], N)], ndmin=2, dtype=object).transpose()
    else:
        bursts = np.array([np.repeat(np.nan, N), np.repeat(0, N), 
                       np.repeat(0, N)], ndmin=2, dtype=object).transpose()
        
    burst_counter = -1
    prev_q = 0
    stack = np.repeat(np.nan, N)
    stack_counter = -1
    for t in range(n):
        if q[t] > prev_q:
            num_levels_opened = q[t] - prev_q
            for i in range(int(num_levels_opened)):
                burst_counter += 1
                bursts[burst_counter, 0] = prev_q + i
                if not freq_type:
                    bursts[burst_counter, 1] = offsets[t]
                else:
                    bursts[burst_counter, 1] = t                    
                stack_counter += 1
                stack[stack_counter] = burst_counter
        elif q[t] < prev_q:
            num_levels_closed = prev_q - q[t]
            for i in range(int(num_levels_closed)):
                if not freq_type:
                    bursts[stack[stack_counter], 2] = offsets[t]
                else:
                    bursts[stack[stack_counter], 2] = t                    
                stack_counter -= 1
        prev_q = q[t] 

    while stack_counter >= 0:
        if not freq_type:
            bursts[stack[stack_counter], 2] = offsets[n]
        else:
            bursts[stack[stack_counter], 2] = n - 1
            
        stack_counter -= 1
            
    return bursts


def kleinberg(offsets, s=2, gamma=1):
    """
	Input:
		offsets: a list of time offsets (numeric)
		s: the base of the exponential distribution that is used for modeling the event frequencies
		gamma: coefficient for the transition costs between states
		
	return:
		An array of intervals in which a burst of activity was detected. 
		The first column denotes the level within the hierarchy; 
		the second column the start value of the interval; 
		the third column the end value. 
		The first row is always the top-level activity (the complete interval from start to finish).
	
	example:
		>>offsets = [4, 17, 23, 27, 33, 35, 37, 76, 77, 82, 84, 88, 90, 92]
		>>bursts = kleinberg(offsets, s=2, gamma=0.1)
		>>print bursts				
    """
    if s <= 1:
        raise ValueError("s must be greater than 1!")
    if gamma <= 0:
        raise ValueError("gamma must be positive!")
    if len(offsets) < 1:
        raise ValueError("offsets must be non-empty!")

    offsets = np.array(offsets, dtype=object)
    if offsets.size == 1:
        bursts = np.array([0, offsets[0], offsets[0]], ndmin=2, dtype=object)
        return bursts

    offsets = np.sort(offsets)
    gaps = np.diff(offsets)

    if not np.all(gaps):
        raise ValueError("Input cannot contain events with zero time between!")

    T = np.sum(gaps)
    n = np.size(gaps)
    g_hat = T / n
    k = int(math.ceil(float(1 + math.log(T, s) + math.log(1 / np.amin(gaps), s))))
    gamma_log_n = gamma * math.log(n)

    def tau(i, j):
        """转移的代价；
        """
        if i >= j:
            return 0
        else:
            return (j - i) * gamma_log_n

    alpha_function = np.vectorize(lambda x: s ** x / g_hat)
    alpha = alpha_function(np.arange(k))

    def f(j, x):
        """ density;
        """
        return alpha[j] * math.exp(-alpha[j] * x)

    C = np.repeat(float("inf"), k)
    C[0] = 0
    q = np.empty((k, 0))
    for t in range(n):
        C_prime = np.repeat(float("inf"), k)
        q_prime = np.empty((k, t + 1))
        q_prime.fill(np.nan)
        for j in range(k):
            cost_function = np.vectorize(lambda x: C[x] + tau(x, j))
            cost = cost_function(np.arange(0, k))
            
            el = np.argmin(cost)
            if f(j, gaps[t]) > 0:
                C_prime[j] = cost[el] - math.log(f(j, gaps[t]))
            if t > 0:
                q_prime[j, 0:t] = q[el]                
            q_prime[j, t] = j + 1
            
        C = C_prime
        q = q_prime

    j = np.argmin(C)
    q = q[j]
    prev_q = 0	
    N = 0
    for t in range(n):
        if q[t] > prev_q:
            N = N + q[t] - prev_q
        prev_q = q[t]
       
    bursts = find_bursty_structure(offsets, q, n, N, freq_type=False)

    return bursts


def kleinberg_freqs_bursts(offsets, s=2, gamma=1, k=2):
    """
	Input:
		offsets: a list of freqs offsets (numeric)
		s: the base of the exponential distribution that is used for modeling the event frequencies
		gamma: coefficient for the transition costs between states
		k: state numbers; 		
	return:
		An array of intervals in which a burst of activity was detected. 
		The first column denotes the level within the hierarchy; 
		the second column the start value of the interval; 
		the third column the end value. 
		The first row is always the top-level activity (the complete interval from start to finish). 			
	""" 
    if s <= 1:
        raise ValueError("s must be greater than 1!")
    if gamma <= 0:
        raise ValueError("gamma must be positive!")
    if len(offsets) < 1:
        raise ValueError("offsets must be non-empty!")
 
    offsets = np.array(offsets, dtype=object)
    if offsets.size == 1:
        bursts = np.array([0, offsets[0], offsets[0]], ndmin=2, dtype=object)
        return bursts
    
    n = len(offsets)
    gamma_log_n = gamma * math.log(n)

    def tau(i, j):
        """转移的代价；
        """
        if i >= j:
            return 0
        else:
            return (j - i) * gamma_log_n

    theta_0 = np.var(offsets)
    alpha_function = np.vectorize(lambda x: (s ** x) * theta_0)
    alpha = alpha_function(np.arange(k))

    def f(j, x):
        """ density;
        """
        return 1.0 / (math.sqrt(6.28 * alpha[j])) * math.exp(-(x ** 2)/(2 * alpha[j]))

    C = np.repeat(float("inf"), k)
    C[0] = 0
    q = np.empty((k, 0))
    for t in range(n):
        C_prime = np.repeat(float("inf"), k)
        q_prime = np.empty((k, t + 1))
        q_prime.fill(np.nan)
        for j in range(k):
            cost_function = np.vectorize(lambda x: C[x] + tau(x, j))
            cost = cost_function(np.arange(0, k))            
            el = np.argmin(cost)
            if f(j, offsets[t]) > 0:
                C_prime[j] = cost[el] - math.log(f(j, offsets[t]))
            if t > 0:
                q_prime[j, 0:t] = q[el]                
            q_prime[j, t] = j + 1
            
        C = C_prime
        q = q_prime

    j = np.argmin(C)
    q = q[j]
    prev_q = 0    
    N = 0
    for t in range(n):
        if q[t] > prev_q:
            N = N + q[t] - prev_q
        prev_q = q[t]
       
    bursts = find_bursty_structure(offsets, q, n, N, freq_type=True) 
    return bursts 
