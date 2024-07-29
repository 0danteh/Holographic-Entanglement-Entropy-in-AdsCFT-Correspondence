import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import SparsePauliOp
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def hamiltonian():
    return SparsePauliOp.from_list([("XX", 1), ("ZZ", 1)])

def quantum_circuit(params):
    qc = QuantumCircuit(4)
    qc.h(range(4))
    for i in range(4):
        qc.ry(params[i], i)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)
    qc.cx(3, 0)
    for i in range(4):
        qc.rz(params[i+4], i)
    qc = qc.bind_parameters(params)
    return qc

def cost_function(params):
    qc = quantum_circuit(params)
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(qc, simulator).result()
    statevector = result.get_statevector()
    H = hamiltonian()
    expectation_value = H.expectation_value(statevector)
    return np.real(expectation_value)

def run_vqe():
    optimizer = COBYLA(maxiter=100)
    ansatz = TwoLocal(2, 'ry', 'cz', entanglement='linear', reps=1)
    estimator = Estimator()
    vqe = VQE(estimator, ansatz, optimizer)
    operator = hamiltonian()
    result = vqe.compute_minimum_eigenvalue(operator)
    return result

def schwarzschild_ads_metric(r, M, L, d):
    term1 = 1 - (2 * M / r**(d-3))
    term2 = r**2 / L**2
    return np.sqrt(term1 + term2)

def reissner_nordstrom_ads_metric(r, M, Q, L, d):
    term1 = 1 - (2 * M / r**(d-3))
    term2 = r**2 / L**2
    term3 = Q**2 / r**(2 * (d-3))
    return np.sqrt(term1 + term2 - term3)

def dz_dz(z, r, d):
    return r * z

def minimal_surface_area(params, z_min, z_max, d, metric_func, M, L, Q=0):
    def integrand(z):
        r = params[0]
        metric = metric_func(r, M, L, d) if metric_func is schwarzschild_ads_metric else metric_func(r, M, Q, L, d)
        dz_dz_value = dz_dz(z, r, d)
        return metric * np.sqrt(1 + dz_dz_value**2) * z**(d-2)
    
    area, _ = quad(integrand, z_min, z_max, epsabs=1.0e-6, epsrel=1.0e-6)
    return area

def compute_entropy(area):
    G_N = 1.0
    return area / (4 * G_N)

def calculate_hee(d, M, L, Q=0, metric_func=schwarzschild_ads_metric):
    z_min = 0.1
    z_max = 1.0
    initial_params = [1.0]
    result = minimize(lambda p: minimal_surface_area(p, z_min, z_max, d, metric_func, M, L, Q), initial_params, method='L-BFGS-B')
    optimal_params = result.x
    area = minimal_surface_area(optimal_params, z_min, z_max, d, metric_func, M, L, Q)
    entropy = compute_entropy(area)
    return entropy

def curved_spacetime_hamiltonian(r, M):
    g_tt = 1 - 2 * M / r
    g_rr = -1 / g_tt
    return SparsePauliOp.from_list([("XI", 0.1 * g_tt), ("IX", 0.1 * g_rr), ("ZZ", 0.1)])

def hawking_temperature(M):
    hbar = 1.0
    c = 1.0
    k_B = 1.0
    G = 1.0
    return hbar * c**3 / (8 * np.pi * G * M * k_B)

def simulate_hawking_radiation(M, r_range):
    optimizer = COBYLA(maxiter=200)
    ansatz = TwoLocal(2, 'ry', 'cz', entanglement='linear', reps=2)
    estimator = Estimator()
    
    radiation_spectrum = []
    for r in r_range:
        H = curved_spacetime_hamiltonian(r, M)
        vqe = VQE(estimator, ansatz, optimizer)
        result = vqe.compute_minimum_eigenvalue(H)
        
        T_H = hawking_temperature(M)
        
        if T_H == 0:
            print(f"Warning: Hawking temperature is zero for r = {r}")
            emission_prob = 0
        else:
            eigenvalue = np.real(result.eigenvalue)
            if np.isnan(eigenvalue) or np.isinf(eigenvalue):
                print(f"Warning: Invalid eigenvalue {eigenvalue} for r = {r}")
                emission_prob = 0
            else:
                # Using Bose-Einstein distribution for emission probability
                emission_prob = 1 / (np.exp(abs(eigenvalue) / T_H) - 1)
        
        radiation_spectrum.append(emission_prob)
    
    return radiation_spectrum

def black_hole_evaporation(M_0, t_max, dt):
    M = M_0
    t = 0
    mass_over_time = [(t, M)]
    
    while M > 0 and t < t_max:
        dM_dt = -1 / (15360 * np.pi * M**2)
        M += dM_dt * dt
        t += dt
        mass_over_time.append((t, M))
        
        if M <= 0:
            break
    
    return mass_over_time

def calculate_black_hole_entropy(A):
    k_B = 1.0
    G = 1.0
    hbar = 1.0
    c = 1.0
    return k_B * c**3 * A / (4 * G * hbar)

def information_paradox_analysis(radiation_spectrum, mass_over_time):
    initial_entropy = calculate_black_hole_entropy(4 * np.pi * mass_over_time[0][1]**2)   
    total_radiation = sum(radiation_spectrum)
    normalized_spectrum = [prob / total_radiation for prob in radiation_spectrum]
    final_entropy = -sum([p * np.log(p) if p > 0 else 0 for p in normalized_spectrum])
    
    print(f"Initial black hole entropy: {initial_entropy}")
    print(f"Final radiation entropy: {final_entropy}")
    
    entropy_difference = final_entropy - initial_entropy
    print(f"Entropy difference: {entropy_difference}")
    
    if entropy_difference < 0:
        print("Information might be lost during evaporation (classical paradox)")
    else:
        print("Information seems to be preserved (quantum resolution)")

quantum_result = run_vqe()
hee_result = calculate_hee(d=5, M=1.0, L=1.0)

print("Quantum result:", quantum_result)
print("Holographic Entanglement Entropy:", hee_result)

M = 1.0  # Black hole mass
r_range = np.linspace(2.1*M, 10*M, 100)
radiation_spectrum = simulate_hawking_radiation(M, r_range)

# Plot Hawking radiation spectrum
plt.figure(figsize=(12, 6))
plt.plot(r_range, radiation_spectrum, color='b', linestyle='-', marker='o')
plt.xlabel('Radius')
plt.ylabel('Emission Probability')
plt.title('Hawking Radiation Spectrum')
plt.grid(True)
plt.show()

# Black hole evaporation
M_0 = 10.0  # Initial black hole mass
t_max = 10  # Max time
dt = 0.1  # Time step
mass_over_time = black_hole_evaporation(M_0, t_max, dt)

times, masses = zip(*mass_over_time)
plt.figure(figsize=(12, 6))
plt.plot(times, masses, color='r', linestyle='-', marker='x')
plt.xlabel('Time')
plt.ylabel('Black Hole Mass')
plt.title('Black Hole Evaporation')
plt.grid(True)
plt.show()

# Information paradox analysis
information_paradox_analysis(radiation_spectrum, mass_over_time)


# Analysis of different black hole masses
masses = [0.5, 1.0, 2.0]
plt.figure(figsize=(12, 8))

for M in masses:
    r_range = np.linspace(2.1*M, 10*M, 100)
    radiation_spectrum = simulate_hawking_radiation(M, r_range)
    plt.plot(r_range, radiation_spectrum, label=f'M = {M}')

plt.xlabel('Radius')
plt.ylabel('Emission Probability')
plt.title('Hawking Radiation Spectrum for Different Black Hole Masses')
plt.legend()
plt.grid(True)
plt.show()

# Analysis of different spacetime dimensions
dimensions = [4, 5, 6]
plt.figure(figsize=(12, 8))

for d in dimensions:
    hee = calculate_hee(d=d, M=1.0, L=1.0)
    r_range = np.linspace(2, 10, 100)
    areas = [minimal_surface_area([r], 0.1, 1.0, d, schwarzschild_ads_metric, 1.0, 1.0) for r in r_range]
    plt.plot(r_range, areas, label=f'd = {d}')

plt.xlabel('Radius')
plt.ylabel('Minimal Surface Area')
plt.title('Minimal Surface Area in Different Spacetime Dimensions')
plt.legend()
plt.grid(True)
plt.show()

# Analysis of Reissner-Nordström-AdS black holes
charges = [0, 0.5, 0.9]
plt.figure(figsize=(12, 8))

for Q in charges:
    hee = calculate_hee(d=5, M=1.0, L=1.0, Q=Q, metric_func=reissner_nordstrom_ads_metric)
    r_range = np.linspace(2, 10, 100)
    areas = [minimal_surface_area([r], 0.1, 1.0, 5, reissner_nordstrom_ads_metric, 1.0, 1.0, Q) for r in r_range]
    plt.plot(r_range, areas, label=f'Q = {Q}')

plt.xlabel('Radius')
plt.ylabel('Minimal Surface Area')
plt.title('Minimal Surface Area for Reissner-Nordström-AdS Black Hole')
plt.legend()
plt.grid(True)
plt.show()

print("\nThis simulation combines quantum circuits, holographic entanglement entropy, and black hole thermodynamics")
print("to explore the information paradox and the behavior of quantum fields in curved spacetime.")
print("The results provide insights into Hawking radiation, black hole evaporation, and the preservation of information in quantum gravity.")
