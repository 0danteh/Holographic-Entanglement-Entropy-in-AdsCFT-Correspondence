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
    qc = QuantumCircuit(2)
    qc.h([0, 1])
    qc.cx(0, 1)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
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
    g_tt = 1 - 2*M/r
    g_rr = -1 / g_tt
    return SparsePauliOp.from_list([("XI", g_tt), ("IX", g_rr), ("ZZ", 1)])

def hawking_temperature(M):
    hbar = 1.0
    c = 1.0
    k_B = 1.0
    G = 1.0
    return hbar * c**3 / (8 * np.pi * G * M * k_B)

def simulate_hawking_radiation(M, r_range):
    optimizer = COBYLA(maxiter=200)  # Increase max iterations
    ansatz = TwoLocal(2, 'ry', 'cz', entanglement='linear', reps=2)  # Increase reps
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
                emission_prob = np.exp(-eigenvalue / T_H)
                if np.isnan(emission_prob) or np.isinf(emission_prob):
                    print(f"Warning: Invalid emission probability for r = {r}, eigenvalue = {eigenvalue}, T_H = {T_H}")
                    emission_prob = 0
                elif emission_prob > 1e10:  # Add a cut-off for very large probabilities
                    print(f"Warning: Extremely large emission probability ({emission_prob}) for r = {r}, capped at 1e10")
                    emission_prob = 1e10
        
        radiation_spectrum.append(emission_prob)
    
    return radiation_spectrum

print(f"Radiation spectrum: {radiation_spectrum}")
print(f"Sum of radiation spectrum: {sum(radiation_spectrum)}")


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
    
    # Normalize the radiation spectrum
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

# Main execution
quantum_result = run_vqe()
hee_result = calculate_hee(d=5, M=1.0, L=1.0)

print("Quantum result:", quantum_result)
print("Holographic Entanglement Entropy:", hee_result)

# Hawking radiation simulation
M = 1.0  # Black hole mass
r_range = np.linspace(2.1*M, 10*M, 100)  # Start from 2.1*M instead of 2*M
radiation_spectrum = simulate_hawking_radiation(M, r_range)

# Plot Hawking radiation spectrum
plt.figure(figsize=(10, 6))
plt.plot(r_range, radiation_spectrum)
plt.xlabel('Radius')
plt.ylabel('Emission Probability')
plt.title('Hawking Radiation Spectrum')
plt.show()

# Continuing from where we left off...

# Black hole evaporation
M_0 = 1.0  # Initial black hole mass
t_max = 1000  # Maximum simulation time
dt = 0.1  # Time step
mass_over_time = black_hole_evaporation(M_0, t_max, dt)

# Plot black hole evaporation
times, masses = zip(*mass_over_time)
plt.figure(figsize=(10, 6))
plt.plot(times, masses)
plt.xlabel('Time')
plt.ylabel('Black Hole Mass')
plt.title('Black Hole Evaporation')
plt.show()

# Information paradox analysis
information_paradox_analysis(radiation_spectrum, mass_over_time)

# Additional analysis: Page curve
def calculate_entanglement_entropy(radiation_entropy, black_hole_entropy):
    return min(radiation_entropy, black_hole_entropy)

radiation_entropies = np.cumsum(radiation_spectrum)
black_hole_entropies = [calculate_black_hole_entropy(4 * np.pi * m**2) for _, m in mass_over_time]
entanglement_entropies = [calculate_entanglement_entropy(r, b) for r, b in zip(radiation_entropies, black_hole_entropies)]

plt.figure(figsize=(10, 6))
plt.plot(times, radiation_entropies, label='Radiation Entropy')
plt.plot(times, black_hole_entropies, label='Black Hole Entropy')
plt.plot(times, entanglement_entropies, label='Entanglement Entropy')
plt.xlabel('Time')
plt.ylabel('Entropy')
plt.title('Page Curve')
plt.legend()
plt.show()

# Analyze the effect of different black hole masses
masses = [0.5, 1.0, 2.0]
plt.figure(figsize=(12, 8))

for M in masses:
    r_range = np.linspace(2*M, 10*M, 100)
    radiation_spectrum = simulate_hawking_radiation(M, r_range)
    plt.plot(r_range, radiation_spectrum, label=f'M = {M}')

plt.xlabel('Radius')
plt.ylabel('Emission Probability')
plt.title('Hawking Radiation Spectrum for Different Black Hole Masses')
plt.legend()
plt.show()

# Analyze the effect of spacetime dimensionality
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
plt.show()

# Analyze the Reissner-Nordström-AdS black hole
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
plt.show()

# Final summary
print("\nFinal Summary:")
print(f"Holographic Entanglement Entropy: {hee_result}")
print(f"Initial Black Hole Mass: {M_0}")
print(f"Final Black Hole Mass: {masses[-1]}")
print(f"Total Radiation Entropy: {radiation_entropies[-1]}")
print(f"Final Black Hole Entropy: {black_hole_entropies[-1]}")
print(f"Final Entanglement Entropy: {entanglement_entropies[-1]}")

if entanglement_entropies[-1] < black_hole_entropies[0]:
    print("The entanglement entropy is less than the initial black hole entropy, suggesting information preservation.")
else:
    print("The entanglement entropy exceeds the initial black hole entropy, indicating potential information loss.")

print("\nThis simulation combines quantum circuits, holographic entanglement entropy, and black hole thermodynamics")
print("to explore the information paradox and the behavior of quantum fields in curved spacetime.")
print("The results provide insights into Hawking radiation, black hole evaporation, and the preservation of information in quantum gravity.")