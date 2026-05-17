import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.sparse import lil_matrix, csr_matrix


def id(i, j, ny):
    return i * ny + j


def build_matrix(lam: np.ndarray, rho_c: np.ndarray, dx: float, dy: float, dt: float) -> csr_matrix:
    """
    Builds a sparse matrix of size (nx, ny) with which you can multiply vector to get another step

    :param lam: matrix of size (nx, ny), contains thermal conductivity of material
    :param rho_c: matrix of size (nx, ny), contains density and heat capacity of material
    :param dx: step of block in x coordinate
    :param dy: step of block in y coordinate
    :param dt: step of time
    :return: A sparse matrix of size (nx, ny)
    """
    nx, ny = rho_c.shape
    N = nx * ny
    A = lil_matrix((N, N))
    for i in range(nx):
        for j in range(ny):
            index_center = id(i, j, ny)
            if i == 0 or j == 0 or i == nx - 1 or j == ny - 1:
                A[index_center, index_center] = 1
                continue

            multiplier = 2 * dt / rho_c[i, j]

            lamb_center = lam[i, j]
            lamb_top = lam[i - 1, j]
            lamb_down = lam[i + 1, j]
            lamb_left = lam[i, j - 1]
            lamb_right = lam[i, j + 1]

            frac_left = multiplier / ((1 / lamb_center + 1 / lamb_left) * dx ** 2)
            frac_right = multiplier / ((1 / lamb_center + 1 / lamb_right) * dx ** 2)
            frac_top = multiplier / ((1 / lamb_center + 1 / lamb_top) * dy ** 2)
            frac_down = multiplier / ((1 / lamb_center + 1 / lamb_down) * dy ** 2)

            index_top = id(i - 1, j, ny)
            index_down = id(i + 1, j, ny)
            index_left = id(i, j - 1, ny)
            index_right = id(i, j + 1, ny)
            A[index_center, index_center] = 1 - (frac_left + frac_right + frac_top + frac_down)  # center

            A[index_center, index_top] = frac_top  # top
            A[index_center, index_down] = frac_down  # down
            A[index_center, index_left] = frac_left  # left
            A[index_center, index_right] = frac_right  # right
    return A.tocsr()


@jit(nopython=True)
def step(u: np.ndarray, lam: np.ndarray, rho_c: np.ndarray, dx: float, dy: float, dt: float) -> np.ndarray:
    """
    Computes another step in temperature change

    :param u: matrix of size (nx, ny), contains current temperature
    :param lam: matrix of size (nx, ny), contains thermal conductivity of material
    :param rho_c: matrix of size (nx, ny), contains density and heat capacity of material
    :param dx: step of block in x coordinate
    :param dy: step of block in y coordinate
    :param dt: step of time
    :return: matrix of size (nx, ny) with new temperatures
    """
    nx, ny = u.shape
    u_new = np.copy(u)
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            temp_center = u[i, j]
            rho_center = rho_c[i, j]
            lamb_center = lam[i, j]

            temp_top = u[i - 1, j]
            lamb_top = lam[i - 1, j]

            temp_down = u[i + 1, j]
            lamb_down = lam[i + 1, j]

            temp_left = u[i, j - 1]
            lamb_left = lam[i, j - 1]

            temp_right = u[i, j + 1]
            lamb_right = lam[i, j + 1]

            frac1 = (temp_left - temp_center) / (((1 / lamb_center) + (1 / lamb_left)) * dx ** 2)
            frac2 = (temp_top - temp_center) / (((1 / lamb_center) + (1 / lamb_top)) * dy ** 2)
            frac3 = (temp_right - temp_center) / (((1 / lamb_center) + (1 / lamb_right)) * dx ** 2)
            frac4 = (temp_down - temp_center) / (((1 / lamb_center) + (1 / lamb_down)) * dy ** 2)
            u_new[i, j] = temp_center + (2 * dt / rho_center) * (frac1 + frac2 + frac3 + frac4)
    return u_new


@jit(nopython=True)
def simulateByCycle(u: np.ndarray, lam: np.ndarray, rho_c: np.ndarray, dx: float, dy: float, dt: float,
                    numberOfSteps: int) -> np.ndarray:
    """
    Simulates a cycling step in temperature change by using cycle

    :param u: matrix of size (nx, ny), contains current temperature
    :param lam: matrix of size (nx, ny), contains thermal conductivity of material
    :param rho_c: matrix of size (nx, ny), contains density and heat capacity of material
    :param dx: step of block in x coordinate
    :param dy: step of block in y coordinate
    :param dt: step of time
    :param numberOfSteps: number of steps to simulate
    :return: matrix of size (nx, ny) with new temperatures after all steps
    """
    if lam.shape != rho_c.shape or lam.shape != u.shape:
        raise ValueError("lamb, rho and u must have the same shape")

    nx, ny = u.shape
    u_new = np.copy(u)
    for i in range(numberOfSteps + 1):
        u_new = step(u_new, lam, rho_c, dx, dy, dt)
    return u_new


def simulateByMatrix(u: np.ndarray, lam: np.ndarray, rho_c: np.ndarray, dx: float, dy: float, dt: float,
                     numberOfSteps: int) -> np.ndarray:
    """
    Simulates a cycling step in temperature change by using sparse matrix multiplication

    :param u: matrix of size (nx, ny), contains current temperature
    :param lam: matrix of size (nx, ny), contains thermal conductivity of material
    :param rho_c: matrix of size (nx, ny), contains density and heat capacity of material
    :param dx: step of block in x coordinate
    :param dy: step of block in y coordinate
    :param dt: step of time
    :param numberOfSteps: number of steps to simulate
    :return: matrix of size (nx, ny) with new temperatures after all steps
    """
    if lam.shape != rho_c.shape or lam.shape != u.shape:
        raise ValueError("lamb, rho and u must have the same shape")

    nx, ny = u.shape
    u_new = np.copy(u).flatten()
    matrix = build_matrix(lam, rho_c, dx, dy, dt)
    for i in range(numberOfSteps + 1):
        u_new = matrix @ u_new
    u_new = u_new.reshape(nx, ny)
    return u_new


def plot(u, title="Teplota"):
    """
    Plots the temperature matrix
    :param u: matrix of size (nx, ny) with temperatures
    :param title: title of the plot
    """
    fig = plt.figure()
    plt.imshow(u, cmap="hot", origin="lower")
    plt.colorbar(label="Teplota")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
