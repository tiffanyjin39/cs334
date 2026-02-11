"""
Vectorization Comparison for Computing Sum of Squares
~~~~~~
"""

import timeit
import numpy as np
import pandas as pd

def gen_random_samples(n):
    """
    Generate n random samples using the
    numpy random.randn module.

    Returns
    ----------
    sample : 1d array of size n
        An array of n random samples
    """
    samples = np.random.randn(n)
    return samples


def sum_squares_for(samples):
    """
    Compute the sum of squares using a forloop

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    """
    ss = 0
    for x in samples:
      ss = ss + x*x
    return ss


def sum_squares_np(samples):
    """
    Compute the sum of squares using Numpy's dot module

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    """
    
    ss = np.dot(samples,samples)
    return ss


def time_ss(sample_list):
    """
    Time it takes to compute the sum of squares
    for varying number of samples. The function should
    generate a random sample of length s (where s is an 
    element in sample_list), and then time the same random 
    sample using the for and numpy loops.

    Parameters
    ----------
    samples : list of length n
        A list of integers to .

    Returns
    -------
    ss_dict : Python dictionary with 3 keys: n, ssfor, ssnp.
        The value for each key should be a list, where the 
        ordering of the list follows the sample_list order 
        and the timing in seconds associated with that 
        number of samples.
    """
    ss_dict = {
      'n':[],
      'ssfor':[],
      'ssnp':[]
    }
    
    for s in sample_list:
      # n
      samples = gen_random_samples(s)

      # time for ssfor
      start = timeit.default_timer()
      sum_squares_for(samples)
      elapsed_ssfor = timeit.default_timer() - start

      # time for ssnp
      start = timeit.default_timer()
      sum_squares_np(samples)
      elapsed_ssnp = timeit.default_timer() - start

      # output
      ss_dict['n'].append(s)
      ss_dict['ssfor'].append(elapsed_ssfor)
      ss_dict['ssnp'].append(elapsed_ssnp)

    return ss_dict


def timess_to_df(ss_dict):
    """
    Time the time it takes to compute the sum of squares
    for varying number of samples.

    Parameters
    ----------
    ss_dict : Python dictionary with 3 keys: n, ssfor, ssnp.
        The value for each key should be a list, where the 
        ordering of the list follows the sample_list order 
        and the timing in seconds associated with that 
        number of samples.

    Returns
    -------
    time_df : Pandas dataframe that has n rows and 3 columns.
        The column names must be n, ssfor, ssnp and follow that order.
        ssfor and ssnp should contain the time in seconds.
    """
    time_df = pd.DataFrame({
      'n':ss_dict['n'],
      'ssfor':ss_dict['ssfor'],
      'ssnp':ss_dict['ssnp']
    })
    return time_df

def main():
    # generate 100 samples
    samples = gen_random_samples(100)
    # call the for version
    ss_for = sum_squares_for(samples)
    # call the numpy version
    ss_np = sum_squares_np(samples)
    # make sure they are approximately the same value
    import numpy.testing as npt
    npt.assert_almost_equal(ss_for, ss_np, decimal=5)


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt

n_range = [10,100,1000,10000,100000,1000000,10000000]
ss_dict = time_ss(n_range)
time_df = timess_to_df(ss_dict)

plt.figure()
# Plot for method 1 using for loop
plt.scatter(time_df['n'],time_df['ssfor'],color='black',marker='o',
            label='For-loop method')
# Plot for method 2 using numpy
plt.scatter(time_df['n'],time_df['ssnp'],color='dodgerblue',marker='s',
            label='Numpy-loop method')
plt.legend()
plt.xscale('log')
plt.xlabel('Samples')
plt.ylabel('Time')
plt.show()
