## Overview

This repository contains a comprehensive simulation that combines quantum circuits, holographic entanglement entropy, and black hole thermodynamics to explore the information paradox and the behavior of quantum fields in curved spacetime. The results provide insights into Hawking radiation, black hole evaporation, and the preservation of information in quantum gravity.

The main components of the code are:
1. Variational Quantum Eigensolver (VQE) for quantum simulations.
2. Holographic Entanglement Entropy (HEE) calculations.
3. Black hole thermodynamics and Hawking radiation.
4. Black hole evaporation and information paradox analysis.

## Mathematical Background

### Quantum Hamiltonian

The Hamiltonian used for quantum simulations is defined as:
$$ \hat{H} = \sigma_x \otimes \sigma_x + \sigma_z \otimes \sigma_z $$
where $\sigma_x$ and $\sigma_z$ are the Pauli matrices. This Hamiltonian represents the interactions between qubits and is used to find the ground state energy using the VQE.

### Quantum Circuit

The quantum circuit is constructed as follows:
1. Apply Hadamard gates to each qubit to create superposition states.
2. Apply rotation $R_y$ gates with parameters $\theta_i$ to each qubit:
   $$ R_y(\theta) = \exp(-i \theta \sigma_y / 2) $$
3. Use Controlled-X (CX) gates to entangle the qubits.
4. Apply rotation $R_z$ gates with parameters $\phi_i$ to each qubit:
   $$ R_z(\phi) = \exp(-i \phi \sigma_z / 2) $$

The cost function for the VQE is the expectation value of the Hamiltonian:
$$ E(\theta, \phi) = \langle \psi(\theta, \phi) | \hat{H} | \psi(\theta, \phi) \rangle $$
where $| \psi(\theta, \phi) \rangle$ is the state prepared by the quantum circuit.

### Schwarzschild-AdS Metric

The Schwarzschild-AdS metric in $d$ dimensions is given by:
$$ ds^2 = -\left(1 - \frac{2M}{r^{d-3}} + \frac{r^2}{L^2}\right) dt^2 + \left(1 - \frac{2M}{r^{d-3}} + \frac{r^2}{L^2}\right)^{-1} dr^2 + r^2 d\Omega_{d-2}^2 $$
Here, $M$ is the black hole mass, $L$ is the AdS radius, and $d\Omega_{d-2}^2$ represents the metric on a unit $(d-2)$-sphere.

### Reissner-Nordström-AdS Metric

The Reissner-Nordström-AdS metric in $d$ dimensions is given by:
$$ ds^2 = -\left(1 - \frac{2M}{r^{d-3}} + \frac{r^2}{L^2} - \frac{Q^2}{r^{2(d-3)}}\right) dt^2 + \left(1 - \frac{2M}{r^{d-3}} + \frac{r^2}{L^2} - \frac{Q^2}{r^{2(d-3)}}\right)^{-1} dr^2 + r^2 d\Omega_{d-2}^2 $$
where $Q$ is the charge of the black hole.

### Minimal Surface Area

To calculate the minimal surface area in AdS space, we consider the integral:
$$ \text{Area} = \int_{z_{\min}}^{z_{\max}} \sqrt{1 + \left(\frac{dz}{dr}\right)^2} r^{d-2} dz $$
where $ \frac{dz}{dr} = r \cdot z $. The metric functions depend on the specific black hole solution used.

### Entropy

The entropy associated with the minimal surface area is given by the Bekenstein-Hawking formula:
$$ S = \frac{\text{Area}}{4 G_N} $$
where $G_N$ is the Newtonian gravitational constant.

### Hawking Temperature

The Hawking temperature for a black hole of mass $M$ is:
$$ T_H = \frac{\hbar c^3}{8 \pi G k_B M} $$
where $\hbar$ is the reduced Planck constant, $c$ is the speed of light, $G$ is the gravitational constant, and $k_B$ is the Boltzmann constant.

### Black Hole Evaporation

The rate of mass loss due to Hawking radiation is described by:
$$ \frac{dM}{dt} = -\frac{1}{15360 \pi M^2} $$
This differential equation models the gradual evaporation of the black hole over time.

### Information Paradox

The information paradox is analyzed by comparing the initial entropy of the black hole with the final entropy of the emitted radiation. The entropy difference indicates whether information is preserved or lost during evaporation.
