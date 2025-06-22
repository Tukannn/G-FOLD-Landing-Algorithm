# G-FOLD: Fuel Optimal Large Divert Guidance Algorithm

This project utilizes g-fold algorithm to optimize a landing trajectory based on input parameters, G-FOLD is a convex-optimization algorithm that generates fuel-optimal path to land a spacecraft at the desired location.

By default, the input parameters are representative of a small lunar lander.


## Simulation Parameters
- g_moon = 1.625  # Lunar gravity (m/s^2)
- initial_altitude = 1000.0  # Initial altitude (m)
- initial_velocity = -20.0  # Initial vertical velocity (m/s)
- initial_mass = 2000.0  # Initial mass of the spacecraft (kg)
- Isp = 300.0  # Specific impulse of the engine (s)
- g0 = 9.81  # Standard gravity for Isp calculation (m/s^2)
- T_max = 15000.0  # Maximum thrust (N)
- dt = 0.5  # Time step (s)
- total_time = 120.0  # Total simulation time (s)
## Resulting in such output:

![image](https://github.com/user-attachments/assets/9fc61578-e980-42a5-8f01-7704afa3de8e)

