import numpy as np
import matplotlib.pyplot as plt
from numba import stencil
from numba import jit
from scipy.sparse import lil_matrix

def id(i, j, ny):
    return i*ny+j

def build_matrix(lam, rho_c, DX, DY, DT) -> lil_matrix:
    nx, ny = rho_c.shape
    N = nx * ny
    A = lil_matrix((N, N))
    for i in range(nx):
        for j in range(ny):
            index_center = id(i, j, ny)
            if i == 0 or j == 0 or i == nx - 1 or j == ny - 1:
                A[index_center, index_center] = 1
                continue

            multiplier = 2 * DT / rho_c[i, j]

            lamb_center = lam[i, j]
            lamb_top = lam[i-1, j]
            lamb_down = lam[i+1, j]
            lamb_left = lam[i, j-1]
            lamb_right = lam[i, j+1]

            frac_left = multiplier/((1/lamb_center + 1/lamb_left)*DX**2)
            frac_right = multiplier/((1/lamb_center + 1/lamb_right)*DX**2)
            frac_top = multiplier/((1/lamb_center + 1/lamb_top)*DY**2)
            frac_down = multiplier/((1/lamb_center + 1/lamb_down)*DY**2)

            index_top = id(i - 1, j, ny)
            index_down = id(i + 1, j, ny)
            index_left = id(i, j - 1, ny)
            index_right = id(i, j + 1, ny)
            A[index_center, index_center] = 1 - (frac_left + frac_right + frac_top + frac_down) # center

            A[index_center, index_top] = frac_top # top
            A[index_center, index_down] = frac_down # down
            A[index_center, index_left] = frac_left # left
            A[index_center, index_right] = frac_right # right
    return A.tocsr()


@jit(nopython=True)
def step(u, lam, rho_c, DX, DY, DT):
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

            frac1 = (temp_left - temp_center) / (((1 / lamb_center) + (1 / lamb_left)) * DX ** 2)
            frac2 = (temp_top - temp_center) / (((1 / lamb_center) + (1 / lamb_top)) * DY ** 2)
            frac3 = (temp_right - temp_center) / (((1 / lamb_center) + (1 / lamb_right)) * DX ** 2)
            frac4 = (temp_down - temp_center) / (((1 / lamb_center) + (1 / lamb_down)) * DY ** 2)
            u_new[i, j] = temp_center + (2 * DT / rho_center) * (frac1 + frac2 + frac3 + frac4)
    return u_new

@jit(nopython=True)
def simulateByCycle(u, lam, rho_c, DX, DY, DT,  numberOfSteps):
    if lam.shape != rho_c.shape or lam.shape != u.shape:
        raise ValueError("lamb, rho and u must have the same shape")

    nx, ny = u.shape
    u_new = np.copy(u)
    for i in range(numberOfSteps+1):
        u_new = step(u_new, lam, rho_c, DX, DY, DT)
    return u_new

def simulateByMatrix(u, lam, rho_c, DX, DY, DT, numberOfSteps):
    if lam.shape != rho_c.shape or lam.shape != u.shape:
        raise ValueError("lamb, rho and u must have the same shape")

    nx, ny = u.shape
    u_new = np.copy(u).flatten()
    matrix = build_matrix(lam, rho_c, DX, DY, DT)
    for i in range(numberOfSteps+1):
        u_new = matrix @ u_new
    u_new = u_new.reshape(nx, ny)
    return u_new

def plot(u, title="Teplota"):
    fig = plt.figure()
    plt.imshow(u, cmap="hot", origin="lower")
    plt.colorbar(label="Teplota")
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
