"""
Linear Regression
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from helper import load_data
import time

def generate_polynomial_features(X, M):
    """
    Create a polynomial feature mapping from input examples. Each element x
    in X is mapped to an (M+1)-dimensional polynomial feature vector
    i.e. [1, x, x^2, ...,x^M].

    Args:
        X: np.array, shape (N, 1). Each row is one instance.
        M: a non-negative integer

    Returns:
        Phi: np.array, shape (N, M+1)
    """
    N = X.shape[0]
    Phi = np.zeros((N, M + 1))

    for i in range(M + 1):
        Phi[:, i] = (X[:, 0]) ** i

    return Phi

def calculate_squared_loss(X, y, theta):
    """
    Args:
        X: np.array, shape (N, d)
        y: np.array, shape (N,)
        theta: np.array, shape (d,)

    Returns:
        loss: float. The empirical risk based on squared loss as defined in the assignment.

    """
    N = X.shape[0]
    errors_sq = (y - np.dot(X, theta)) ** 2

    loss = (1 / N) * np.sum(errors_sq / 2)
    return loss

def calculate_RMS_Error(X, y, theta):
    """
    Args:
        X: np.array, shape (N, d)
        y: np.array, shape (N,)
        theta: np.array, shape (d,)

    Returns:
        E_rms: float. The root mean square error as defined in the assignment.
    """
    N = X.shape[0]

    squared_errors = (y - np.dot(X, theta)) ** 2
    E_rms = np.sqrt((1 / N) * np.sum(squared_errors))

    return E_rms


def ls_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Gradient Descent (GD) algorithm for least squares regression.
    Note:
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10
    Args:
        X: np.array, shape (N, d)
        y: np.array, shape (N,)
        learning_rate: float, the learning rate for GD

    Returns:
        theta: np.array, shape (d,)
    """
    N, d = X.shape
    theta = np.zeros(d)

    stopping = int(1e6)
    convergence = 1e-10

    prev_loss = calculate_squared_loss(X, y, theta)

    for iteration in range(stopping):
        gradient = (1 / N) * np.dot(X.T, np.dot(X, theta) - y)
        theta = theta - learning_rate * gradient
        # check for convergence
        new_loss = calculate_squared_loss(X, y, theta)
        if abs(new_loss - prev_loss) <= convergence:
          ls_gradient_descent.iterations = iteration + 1
          break
        prev_loss = new_loss

    else:
        ls_gradient_descent.iterations = stopping

    return theta


def ls_stochastic_gradient_descent(X, y, learning_rate=0):
    """
    Implements the Stochastic Gradient Descent (SGD) algorithm for least squares regression.
    Note:
        - Please do not shuffle your data points.
        - Please use the stopping criteria: number of iterations >= 1e6 or |new_loss - prev_loss| <= 1e-10

    Args:
        X: np.array, shape (N, d)
        y: np.array, shape (N,)
        learning_rate: float or 'adaptive', the learning rate for SGD

    Returns:
        theta: np.array, shape (d,)
    """
    N, d = X.shape
    theta = np.zeros(d)

    stopping = int(1e6)
    convergence = 1e-10
    n = 0

    prev_loss = calculate_squared_loss(X, y, theta)

    while n < int(stopping):
        for i in range(N):
            xi = X[i]
            yi = y[i]

            gradient = (np.dot(X[i], theta) - y[i]) * X[i]

            # Adaptive learning rate based on function n_k = 1/(1+k)
            if learning_rate == 'adaptive':
                n_k = 1 / np.sqrt(1 + n)
            else:
                n_k = learning_rate

            theta = theta - n_k * gradient
            n += 1
        # check for convergence
        new_loss = calculate_squared_loss(X, y, theta)
        if abs(new_loss - prev_loss) <= convergence:
            break
        prev_loss = new_loss

    ls_stochastic_gradient_descent.iterations = n

    return theta


def ls_closed_form_solution(X, y, reg_param=0):
    """
    Implements the closed form solution for least squares regression.

    Args:
        X: np.array, shape (N, d)
        y: np.array, shape (N,)
        reg_param: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    d = X.shape[1]
    XtX = np.dot(X.T, X)
    Xty = np.dot(X.T, y)

    if reg_param > 0:
      XtX = XtX + reg_param * np.eye(d)

    theta = np.dot(np.linalg.pinv(XtX), Xty)

    return theta


''' Uncomment this if you are attempting the extra credit
def weighted_ls_closed_form_solution(X, y, weights, reg_param=0):
    """
    Implements the closed form solution for weighted least squares regression.

    Args:
        X: np.array, shape (N, d)
        y: np.array, shape (N,)
        weights: np.array, shape (N,), the weights for each data point
        reg_param: float, an optional regularization parameter

    Returns:
        theta: np.array, shape (d,)
    """
    # TODO: Implement this function
    theta = ???
    return theta
'''


def part_1(fname_train):
    """
    This function should contain all the code you implement to complete part 1

    # Example of how to use the functions
    start = time.process_time()
    theta = ls_stochastic_gradient_descent(Phi_train, y_train, learning_rate=0.01)
    print('Time elapsed:', time.process_time() - start)
    """
    print("========== Part 1 ==========")

    X_train, y_train = load_data(fname_train)
    Phi_train = generate_polynomial_features(X_train, 1)

    # Test
    learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]

    print("\nGradient Descent:")
    for eta in learning_rates:
        start = time.process_time()
        theta = ls_gradient_descent(Phi_train, y_train, learning_rate=eta)
        print(f'η = {eta}: θ0 = {theta[0]}, θ1 = {theta[1]}, Time elapsed: {time.process_time() - start}')
        print(f'# iterations: {ls_gradient_descent.iterations}')

    print("\nStochastic Gradient Descent:")
    for eta in learning_rates:
        start = time.process_time()
        theta = ls_stochastic_gradient_descent(Phi_train, y_train, learning_rate=eta)
        print(f'η = {eta}: θ0 = {theta[0]}, θ1 = {theta[1]}, Time elapsed: {time.process_time() - start}')
        print(f'# iterations: {ls_stochastic_gradient_descent.iterations}')
    print("\nAdaptive Learning Rate:")
    start = time.process_time()
    theta = ls_stochastic_gradient_descent(Phi_train, y_train, learning_rate='adaptive')
    print(f'Time elapsed: {time.process_time() - start}')
    print(f'θ0 = {theta[0]:.6f}, θ1 = {theta[1]:.6f}')
    print(f'# iterations: {ls_stochastic_gradient_descent.iterations}')

    print("\nClosed-Form Solution:")
    start = time.process_time()
    theta = ls_closed_form_solution(Phi_train, y_train)
    print(f'θ0 = {theta[0]}, θ1 = {theta[1]}, Time elapsed: {time.process_time() - start}')

    print("Done!")


def part_2(fname_train, fname_validation):
    """
    This function should contain all the code you implement to complete part 2
    """
    print("=========== Part 2(b) ==========")

    X_train, y_train = load_data(fname_train)
    X_validation, y_validation = load_data(fname_validation)

    M_values = range(0, 11)
    train_errors = []
    val_errors = []

    for M in M_values:
        Phi_train = generate_polynomial_features(X_train, M)
        Phi_val = generate_polynomial_features(X_validation, M)

        theta = ls_closed_form_solution(Phi_train, y_train)

        # Calculate RMS Error
        train_rms = calculate_RMS_Error(Phi_train, y_train, theta)
        train_errors.append(train_rms)

        val_rms = calculate_RMS_Error(Phi_val, y_validation, theta)
        val_errors.append(val_rms)

        print(f"M = {M}: Train E_RMS = {train_rms}, Validation E_RMS = {val_rms}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(M_values, train_errors, 'ro-', label='Train', markersize=8)
    plt.plot(M_values, val_errors, 'bo-', label='Validation', markersize=8)
    plt.xlabel('M')
    plt.ylabel('E_RMS')
    plt.title('RMS Error vs M')
    plt.legend()
    plt.xticks(M_values)
    plt.savefig('3.2d.png')
    plt.show()

    print("=========== Part 2(e) ==========")

    M = 10
    lambdas = [0] + [10 ** (-k) for k in range(8, -1, -1)]

    Phi_train = generate_polynomial_features(X_train, M)
    Phi_val = generate_polynomial_features(X_validation, M)

    train_errors = []
    val_errors = []

    for l in lambdas:
        theta = ls_closed_form_solution(Phi_train, y_train, reg_param=l)

        # Calculate RMS Error
        train_rms = calculate_RMS_Error(Phi_train, y_train, theta)
        val_rms = calculate_RMS_Error(Phi_val, y_validation, theta)

        train_errors.append(train_rms)
        val_errors.append(val_rms)
        
        print(f"λ = {l}: Train = {train_rms}, Val = {val_rms}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, train_errors, 'ro-', label='Train')
    plt.plot(lambdas, val_errors, 'bo-', label='Validation')
    plt.xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('E_RMS')
    plt.title('RMS Error vs Lambda')
    plt.legend()
    plt.savefig('3.2e.png')
    plt.show()

    print("Done!")


''' Uncomment this if you are attempting the extra credit
def extra_credit(fname_train, fname_validation):
    """
    This function should contain all the code you implement to complete extra credit
    """
    print("=========== Extra Credit ==========")

    X_train, y_train, weights_train = load_data(fname_train, weighted=True)
    X_validation, y_validation, weights_validation = load_data(fname_validation, weighted=True)

    # TODO: Add more code here to complete the extra credit
    ##############################

    print("Done!")
'''


def main(fname_train, fname_validation):
    part_1(fname_train)
    part_2(fname_train, fname_validation)
#    extra_credit(fname_train, fname_validation)


if __name__ == '__main__':
    main("data/linreg_train.csv", "data/linreg_validation.csv")
