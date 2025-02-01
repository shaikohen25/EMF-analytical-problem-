import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
import math

# Constants
V = 1  # Voltage
R = 1  # Radius of the disc
d = 0.05  # Step size for discretization
epsilon_0 = constants.epsilon_0  # Permittivity of free space

# Helper Functions
def is_in_circle(R, d, i, j):
    """
    Check if a point lies inside the disc.
    """
    x = d * i + d / 2 - R
    y = d * j + d / 2 - R
    return (x ** 2 + y ** 2) <= R ** 2


def generate_coordinates(R, d):
    """
    Generate coordinates for points inside the circular disc.
    """
    n = int(np.floor(2 * R / d))
    points = []
    for i in range(n):
        for j in range(n):
            if is_in_circle(R, d, i, j):
                x = d * i + d / 2
                y = d * j + d / 2
                points.append((x, y))
    return points


def build_interaction_matrix(points, D=0):
    """
    Build the interaction matrix for the system.
    """
    N = len(points)
    matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                matrix[i][j] = (d / math.pi) * epsilon_0 * 0.8814
            else:
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                dist = math.sqrt(dx ** 2 + dy ** 2 + D ** 2)
                matrix[i][j] = 1 / dist
    return matrix


def solve_charge_distribution(V, R, d, D=0):
    """
    Solve for the charge distribution on the disc.
    """
    points = generate_coordinates(R, d)
    N = len(points)
    interaction_matrix = build_interaction_matrix(points, D)
    voltage_vector = np.full(N, V / 2)

    # Solve system of linear equations
    charge_distribution = (4 * math.pi * epsilon_0) * np.linalg.solve(interaction_matrix, voltage_vector)
    return charge_distribution


def calculate_total_charge(charge_distribution):
    """
    Calculate the total charge from the distribution.
    """
    return np.sum(np.abs(charge_distribution))


def plot_capacitance_vs_distance():
    """
    Plot the capacitance as a function of distance.
    """
    D_array = np.linspace(d / 3, 1, num=30)
    numerical_capacitance = []
    theoretical_capacitance = []

    for D in D_array:
        charge = calculate_total_charge(solve_charge_distribution(V, R, d, D)) * 1e-1
        numerical_capacitance.append(charge)
        theoretical_capacitance.append((epsilon_0 * math.pi) / D)

    plt.plot(D_array, numerical_capacitance, label="Numerical Capacitance")
    plt.plot(D_array, theoretical_capacitance, label="Theoretical Capacitance", linestyle='dashed')
    plt.xlabel("Distance (D)")
    plt.ylabel("Capacitance (C)")
    plt.legend()
    plt.title("Capacitance vs Distance")
    plt.show()

# Compute and Display Results
D_half = R / 2
D_fifth = R / 5
capacitance_half = calculate_total_charge(solve_charge_distribution(V, R, d, D_half)) * 1e-1
capacitance_fifth = calculate_total_charge(solve_charge_distribution(V, R, d, D_fifth)) * 1e-1

print(f"Capacitance for D = R/2: {capacitance_half:.4e} F")
print(f"Capacitance for D = R/5: {capacitance_fifth:.4e} F")

plot_capacitance_vs_distance()
