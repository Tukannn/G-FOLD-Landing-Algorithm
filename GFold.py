import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
g_moon = 1.625  # Lunar gravity (m/s^2)
initial_altitude = 1000.0  # Initial altitude (m)
initial_velocity = -20.0  # Initial vertical velocity (m/s)
initial_mass = 2000.0  # Initial mass of the spacecraft (kg)
Isp = 300.0  # Specific impulse of the engine (s)
g0 = 9.81  # Standard gravity for Isp calculation (m/s^2)
T_max = 15000.0  # Maximum thrust (N)
dt = 0.5  # Time step (s)
total_time = 120.0  # Total simulation time (s)
n_steps = int(total_time / dt)
time = np.linspace(0, total_time, n_steps)

# --- Final Condition Tolerances ---
altitude_tol = 0.1  # Final altitude tolerance (m)
velocity_tol = 0.1  # Final velocity tolerance (m/s)

# --- Sequential Convex Programming (SCP) Parameters ---
max_iterations = 100  # A lower number of iterations for faster execution
relaxation = 0.2  # Damping factor for mass updates (0 < relaxation <= 1)
convergence_tol = 1e-2  # Convergence tolerance for thrust

# --- Initialization for the SCP Loop ---
# Initial guess for the mass profile (assumes constant mass)
mass_prev = initial_mass * np.ones(n_steps)
# Initialize previous thrust profile (bug fix)
thrust_prev = np.zeros(n_steps - 1)

print("Starting Sequential Convex Programming (SCP) loop...")

# --- SCP Loop ---
for iteration in range(max_iterations):
    # --- Define Convex Optimization Variables ---
    # Thrust is the control variable we are solving for
    thrust = cp.Variable(n_steps - 1, nonneg=True)
    # State variables
    altitude = cp.Variable(n_steps)
    velocity = cp.Variable(n_steps)

    # --- Define Constraints for the Convex Subproblem ---
    constraints = [
        # Initial conditions
        altitude[0] == initial_altitude,
        velocity[0] == initial_velocity,

        # Final (landing) conditions within tolerance
        altitude[-1] <= altitude_tol,
        altitude[-1] >= -altitude_tol,
        velocity[-1] <= velocity_tol,
        velocity[-1] >= -velocity_tol,
    ]

    # Dynamics constraints using the mass profile from the previous iteration (mass_prev)
    for i in range(n_steps - 1):
        # Thrust limit
        constraints += [thrust[i] <= T_max]

        # Acceleration is linearized using the previous mass estimate
        acceleration = thrust[i] / mass_prev[i] - g_moon

        # Dynamics using Euler integration for velocity
        constraints += [velocity[i + 1] == velocity[i] + acceleration * dt]

        # Dynamics using Trapezoidal integration for altitude (more accurate)
        constraints += [altitude[i + 1] == altitude[i] + 0.5 * (velocity[i] + velocity[i + 1]) * dt]

    # --- Objective Function ---
    # Minimize total fuel consumption (proportional to total thrust)
    # A small regularization term is added for numerical stability
    objective = cp.Minimize(cp.sum(thrust) * dt + 1e-4 * cp.sum_squares(thrust))

    # --- Solve the Convex Problem ---
    problem = cp.Problem(objective, constraints)
    try:
        # Using ECOS solver, which is good for this type of problem
        problem.solve(solver=cp.ECOS, verbose=True, max_iters=5000)
    except Exception as e:
        print(f"ECOS solver failed: {e}. Trying SCS solver.")
        # Fallback to SCS solver if ECOS fails
        problem.solve(solver=cp.SCS, verbose=False, max_iters=5000)

    # --- Post-Solve Processing ---
    # Check if the solver found a solution
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        print(f"Iteration {iteration + 1}: Solver failed. Status: {problem.status}")
        break

    # Check for solver failure before accessing .value
    if thrust.value is None:
        print(f"Iteration {iteration + 1}: Solver returned a non-optimal status and thrust value is None.")
        break

    # --- Mass Profile Update ---
    # Re-calculate the mass profile based on the new optimal thrust
    mass_new = np.ones(n_steps) * initial_mass
    for i in range(n_steps - 1):
        mass_new[i + 1] = mass_new[i] - (thrust.value[i] * dt) / (Isp * g0)

    # Apply relaxation to the mass update to aid convergence
    mass_prev = relaxation * mass_new + (1 - relaxation) * mass_prev

    # --- Convergence Check ---
    # Check if the change in thrust profile is below the tolerance
    thrust_diff = np.linalg.norm(thrust.value - thrust_prev)
    print(f"Iteration {iteration + 1}: Thrust change = {thrust_diff:.4f}")
    if thrust_diff < convergence_tol:
        print(f"\nConverged after {iteration + 1} iterations.")
        break

    # Store the current thrust profile for the next iteration's convergence check
    thrust_prev = thrust.value.copy()

# --- Final Simulation and Plotting ---
if problem.status in ["optimal", "optimal_inaccurate"] and thrust.value is not None:
    print(f"\nOptimization successful. Final fuel consumed: {(initial_mass - mass_prev[-1]):.2f} kg")

    # Reconstruct the final trajectory using the optimized thrust profile and non-linear dynamics
    sim_altitude = np.zeros(n_steps)
    sim_velocity = np.zeros(n_steps)
    sim_mass = np.zeros(n_steps)

    sim_altitude[0] = initial_altitude
    sim_velocity[0] = initial_velocity
    sim_mass[0] = initial_mass

    for i in range(n_steps - 1):
        T = thrust.value[i]
        sim_mass[i + 1] = sim_mass[i] - (T * dt) / (Isp * g0)

        # Use the actual, non-linear dynamics for simulation
        acceleration = T / sim_mass[i] - g_moon

        sim_velocity[i + 1] = sim_velocity[i] + acceleration * dt
        # Use trapezoidal rule for consistency
        sim_altitude[i + 1] = sim_altitude[i] + 0.5 * (sim_velocity[i] + sim_velocity[i + 1]) * dt

    # --- Plotting Results ---
    plt.figure(figsize=(12, 10))
    plt.suptitle("G-FOLD Lunar Landing Simulation Results", fontsize=16)

    # Altitude vs. Time
    plt.subplot(2, 2, 1)
    plt.plot(time, sim_altitude)
    plt.title('Altitude vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.grid(True)

    # Velocity vs. Time
    plt.subplot(2, 2, 2)
    plt.plot(time, sim_velocity)
    plt.title('Velocity vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)

    # Thrust vs. Time
    plt.subplot(2, 2, 3)
    plt.plot(time[:-1], thrust.value)
    plt.title('Thrust vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Thrust (N)')
    plt.grid(True)

    # Mass vs. Time
    plt.subplot(2, 2, 4)
    plt.plot(time, sim_mass)
    plt.title('Mass vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Mass (kg)')
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
else:
    print("\nFailed to find an optimal solution.")
