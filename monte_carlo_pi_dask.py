# -*- coding: utf-8 -*-
import time
from dask.distributed import Client, wait
import numpy as np


def in_circle(n,dummy):
    coords = np.random.rand(n,2)
    count = 0
    for i in range(n):
        if (coords[i][0])**2 + (coords[i][1])**2 < 1:
            count += 1
    return count


def parallel_pi(P,L,N):
    np.random.seed(42)
    client = Client(n_workers=P)
    
    start = time.time()
    
    counts = client.map( in_circle, [L]*N , range(N) )
    total = client.submit(sum,counts)
    wait(total)
    
    pi_estimate = 4*total.result()/L/N
    
    stop = time.time()
    time_ex = stop-start
    
    client.close()
    
    return [pi_estimate, time_ex]


if __name__ == '__main__':
    P = 8
    L = 10000
    N = 10000
    
    [pi_value, run_time] = parallel_pi(P,L,N)
    
    print(pi_value)
    print(run_time)
