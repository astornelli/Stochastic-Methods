import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pandas as pd

np.random.seed(42)

class LennardJones:
    def __init__(self, N_small, N_large, L, radius_small, radius_large, duration, nsteps, v0):
        self.N_small = N_small
        self.N_large = N_large
        self.N = N_small + N_large
        self.L = L
        self.radius_small = radius_small
        self.radius_large = radius_large
        self.duration = duration
        self.nsteps = nsteps
        self.dt = duration / nsteps
        self.v0 = v0
        self.rc = 2.5

        grid_size = int(np.ceil(np.sqrt(N_small + N_large)))
        spacing = L / grid_size
        x = np.linspace(radius_small + spacing / 2, L - radius_small - spacing / 2, grid_size)
        pos = list(product(x, x))
        self.positions_small = np.array(pos[:N_small])
        self.positions_large = np.array(pos[N_small:N_small + N_large])
        self.positions = np.concatenate((self.positions_small, self.positions_large), axis=0)

        theta_small = np.random.uniform(0, 2 * np.pi, size=N_small)
        theta_large = np.random.uniform(0, 2 * np.pi, size=N_large)
        vx_small, vy_small = v0 * np.cos(theta_small), v0 * np.sin(theta_small)
        vx_large, vy_large = v0 * np.cos(theta_large), v0 * np.sin(theta_large)

        self.velocities_small = np.stack((vx_small, vy_small), axis=1)
        self.velocities_large = np.stack((vx_large, vy_large), axis=1)
        self.velocities = np.concatenate((self.velocities_small, self.velocities_large), axis=0)

    def pair_vector(self, r1, r2):
        r = r2 - r1
        r -= np.rint(r / self.L) * self.L
        return r

    def force(self, r):
        dist = np.linalg.norm(r)
        if dist < self.rc:
            return (-48 / dist**14 + 24 / dist**8) * r
        return np.zeros(2)

    def get_forces(self):
        forces = np.zeros_like(self.positions)
        for i in range(self.N):
            for j in range(i + 1, self.N):
                r = self.pair_vector(self.positions[i], self.positions[j])
                f = self.force(r)
                forces[i] += f
                forces[j] -= f
        return forces

    def temperature(self, velocities):
        return 0.5 * np.sum(velocities**2) / self.N

    def step(self, noise_strength=0.1):
        forces = self.get_forces()
    
        # Add white Gaussian noise to forces
        noise = np.random.normal(0, noise_strength, size=self.positions.shape)
        forces += noise
    
        # Verlet-like integration
        self.positions[:] = self.positions + self.velocities * self.dt + 0.5 * forces * self.dt**2
        next_forces = self.get_forces()
    
        # Add noise again for the next forces
        next_noise = np.random.normal(0, noise_strength, size=self.positions.shape)
        next_forces += next_noise
    
        self.velocities[:] = self.velocities + 0.5 * (forces + next_forces) * self.dt
    
        # Reflective boundary conditions
        for i in range(self.N):
            for dim in range(2):
                if self.positions[i, dim] < self.radius_small:
                    self.positions[i, dim] = self.radius_small
                    self.velocities[i, dim] *= -1
                elif self.positions[i, dim] > self.L - self.radius_small:
                    self.positions[i, dim] = self.L - self.radius_small
                    self.velocities[i, dim] *= -1
    
        # Return the noise for each particle (instead of a single magnitude)
        return noise  # Now noise is a 2D array (N, 2)


    def animate(self, T0=None, tau=2):
        all_positions = np.zeros((self.nsteps, self.N, 2))
        all_velocities = np.zeros_like(all_positions)
        noise_record = np.zeros((self.nsteps, self.N, 2))  # Store noise for each particle
        
        for t in range(self.nsteps):
            all_positions[t] = self.positions
            all_velocities[t] = self.velocities
            noise = self.step()  # Now noise is (N, 2) for each particle
            noise_record[t] = noise  # Store the noise for the current timestep
            
            if T0:
                T = self.temperature(self.velocities)
                self.velocities *= np.sqrt(1 + (T0 / T - 1) * self.dt / tau)
    
        return all_positions, all_velocities, noise_record


# Run simulation
sim = LennardJones(N_small=20, N_large=5, L=10, radius_small=0.2, radius_large=0.6, duration=300, nsteps=15000, v0=1.0)
positions, velocities, noise_record = sim.animate()

# Create colormap for large particles
cmap = cm.get_cmap('viridis', sim.N_large)  # Choose a colormap with enough distinct colors
large_colors = [mcolors.to_hex(cmap(i)) for i in range(sim.N_large)]

# Combine small particles (blue) and large particles (colormap)
colors = ['black'] * sim.N_small + large_colors

# Plot animation
fig, ax = plt.subplots()
ax.set_xlim(0, sim.L)
ax.set_ylim(0, sim.L)
ax.set_aspect('equal')

ax.set_xticks([])  # Remove x-axis ticks
ax.set_yticks([])  # Remove y-axis ticks

sizes_small = (2 * sim.radius_small * 10) ** 2
sizes_large = (2 * sim.radius_large * 10) ** 2
sizes = np.concatenate((np.full(sim.N_small, sizes_small), np.full(sim.N_large, sizes_large)))

scat = ax.scatter(positions[0, :, 0], positions[0, :, 1], s=sizes, c=colors)

def update(frame):
    scat.set_offsets(positions[frame])
    scat.set_sizes(sizes)
    scat.set_color(colors)
    return scat,

ani = animation.FuncAnimation(fig, update, frames=sim.nsteps, interval=30, blit=True)
ani.save("brownian_with_noise.mp4", fps=60, dpi=200)
plt.show()
#%%
# Extract large particle indices and positions
start_idx = sim.N_small
end_idx = sim.N
large_positions = positions[:, start_idx:end_idx, :]

# Compute MSD
msd = np.zeros((sim.nsteps, sim.N_large))
initial_positions = large_positions[0]

for t in range(sim.nsteps):
    displacements = large_positions[t] - initial_positions
    squared_displacements = np.sum(displacements**2, axis=1)
    msd[t] = squared_displacements

# Plot MSD
plt.figure(figsize=(8, 5))
for i in range(sim.N_large):
    plt.plot(np.linspace(0, sim.duration, sim.nsteps), msd[:, i],
             label=f'Particle {i+1}', color=large_colors[i])  # Match the color of large particles
plt.xlabel('t')
plt.ylabel('B(t)')
plt.title('Brownian Motion with Noise')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("brownian_plot.pdf")
plt.show()
#%%
# Extract noise for large particles
noise_large = noise_record[:, sim.N_small:, :]  # shape: (nsteps, N_large, 2)

# Recalculate full displacements for MSD
displacements_full = large_positions - initial_positions  # shape: (nsteps, N_large, 2)
msd_time_series = np.sum(displacements_full**2, axis=2)  # shape: (nsteps, N_large)

# Prepare a DataFrame for CSV
data = {}
for i in range(sim.N_large):
    data[f'Particle_{i}_MSD'] = msd_time_series[:, i]
    data[f'Particle_{i}_Noise_X'] = noise_large[:, i, 0]  # x-direction noise for large particles
    data[f'Particle_{i}_Noise_Y'] = noise_large[:, i, 1]  # y-direction noise for large particles

df = pd.DataFrame(data)
df.index.name = 'Timestep'

# Save to CSV
df.to_csv("brownain_noise.csv")
