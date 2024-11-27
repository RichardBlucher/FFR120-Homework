
"""
Author: Terese Kärnell
Ensamble of light-sensitvie robots: clustering 

P1: Plot simulations at elapsed times with positive delay

P2: Repeat with negative delay

P3: Mark clusters in P1 and P2 for t = [100,500]*tau

Q1: Comment on differences in the clustering in P3.
"""
#%% Modules
import math
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib as mpl
from functools import reduce
import time
from scipy.constants import Boltzmann as kB 
from tkinter import *
from scipy.spatial import distance

#%% functions
def evolution_GI_posdelay(x0, y0, phi0, v_inf, v0, Ic, I0, r0, tau, dt, duration, delta):
    """
    Function to generate the trajectory of a light-sensitive robot in a Gaussian
    light intensity zone with positive delay.
    
    Parameters
    ==========
    x0, y0 : Initial position [m].
    phi0 : Initial orientation [rad].
    v_inf : Self-propulsion speed at I=0 [m/s]
    v0 : Self-propulsion speed at I=I0 [m/s]
    Ic : Intensity scale over which the speed decays.
    I0 : Maximum intensity.
    r0 : Standard deviation of the Gaussian intensity.
    tau : Time scale of the rotational diffusion coefficient [s]
    dt : Time step for the numerical solution [s].
    duration : Total time for which the solution is computed [s].
    delta : Positive delay [s].
    """
        
    # Coefficients for the finite difference solution.
    c_noise_phi = np.sqrt(2 / tau * dt)

    N = math.ceil(duration / dt)  # Number of time steps.

    x = np.zeros(N)
    y = np.zeros(N)
    phi = np.zeros(N)
    
    n_delay = int(delta / dt)  # Delay in units of time steps.

    rn = np.random.normal(0, 1, N - 1)
    
    x[0] = x0
    y[0] = y0
    phi[0] = phi0
    I_ref = I0 * np.exp(- (x0 ** 2 + y0 ** 2) / r0 ** 2)

    for i in range(N - 1):
        if i < n_delay:
            I = I_ref
        else:
            I = I0 * np.exp(- (x[i - n_delay] ** 2 + y[i - n_delay] ** 2) / r0 ** 2)
        
        v = v_inf + (v0 - v_inf) * np.exp(- I / Ic) 
        x[i + 1] = x[i] + v * dt * np.cos(phi[i])
        y[i + 1] = y[i] + v * dt * np.sin(phi[i])
        phi[i + 1] = phi[i] + c_noise_phi * rn[i]

    return x, y, phi

def plot_traj(r0,xp,yp):
    # Plot trajectory.
    plt.figure(figsize=(5, 5))
    # Plot a reference circle with radius r0.
    plt.plot(r0 * np.cos(2 * np.pi * np.arange(361) / 360),
            r0 * np.sin(2 * np.pi * np.arange(361) / 360),
            '--', color='k', linewidth=2)
    # Plot a reference point in the origin.
    plt.plot(0, 0, '.', color='k', markersize=10)
    # Plot the trajectory.
    plt.plot(xp, yp, '-', linewidth=1 ) 
    plt.title('Trajectory (positive delay)')
    plt.axis('equal')
    plt.xticks([]) 
    plt.yticks([]) 
    plt.show()     

def replicas(x, y, L):
    """
    Function to generate replicas of a single particle.
    
    Parameters
    ==========
    x, y : Position.
    L : Side of the squared arena.
    """    
    xr = np.zeros(9)
    yr = np.zeros(9)

    for i in range(3):
        for j in range(3):
            xr[3 * i + j] = x + (j - 1) * L
            yr[3 * i + j] = y + (i - 1) * L
    
    return xr, yr

def pbc(x, y, L):
    """
    Function to enforce periodic boundary conditions on the positions.
    
    Parameters
    ==========
    x, y : Position.
    L : Side of the squared arena.
    """   
    
    outside_left = np.where(x < - L / 2)[0]
    x[outside_left] = x[outside_left] + L

    outside_right = np.where(x > L / 2)[0]
    x[outside_right] = x[outside_right] - L

    outside_up = np.where(y > L / 2)[0]
    y[outside_up] = y[outside_up] - L

    outside_down = np.where(y < - L / 2)[0]
    y[outside_down] = y[outside_down] + L
    
    return x, y

def calculate_intensity(x, y, I0, r0, L, r_c):
    """
    Function to calculate the intensity seen by each particle.
    
    Parameters
    ==========
    x, y : Positions.
    r0 : Standard deviation of the Gaussian light intensity zone.
    I0 : Maximum intensity of the Gaussian.
    L : Dimension of the squared arena.
    r_c : Cut-off radius. Pre-set it around 3 * r0. 
    """
    
    N = np.size(x)

    I_particle = np.zeros(N)  # Intensity seen by each particle.
    
    # Preselect what particles are closer than r_c to the boundaries.
    replicas_needed = reduce( 
        np.union1d, (
            np.where(y + r_c > L / 2)[0], 
            np.where(y - r_c < - L / 2)[0],
            np.where(x + r_c > L / 2)[0],
            np.where(x - r_c > - L / 2)[0]
        )
    )

    for j in range(N - 1):   
        
        # Check if replicas are needed to find the interacting neighbours.
        if np.size(np.where(replicas_needed == j)[0]):
            # Use replicas.
            xr, yr = replicas(x[j], y[j], L)
            for nr in range(9):
                dist2 = (x[j + 1:] - xr[nr]) ** 2 + (y[j + 1:] - yr[nr]) ** 2 
                nn = np.where(dist2 <= r_c ** 2)[0] + j + 1
                
                # The list of nearest neighbours is set.
                # Contains only the particles with index > j
        
                if np.size(nn) > 0:
                    nn = nn.astype(int)
        
                    # Find total intensity
                    dx = x[nn] - xr[nr]
                    dy = y[nn] - yr[nr]
                    d2 = dx ** 2 + dy ** 2
                    I = I0 * np.exp(- d2 / r0 ** 2)
                    
                    # Contribution for particle j.
                    I_particle[j] += np.sum(I)

                    # Contribution for nn of particle j nr replica.
                    I_particle[nn] += I
                
        else:
            dist2 = (x[j + 1:] - x[j]) ** 2 + (y[j + 1:] - y[j]) ** 2 
            nn = np.where(dist2 <= r_c ** 2)[0] + j + 1
        
            # The list of nearest neighbours is set.
            # Contains only the particles with index > j
        
            if np.size(nn) > 0:
                nn = nn.astype(int)
        
                # Find interaction
                dx = x[nn] - x[j]
                dy = y[nn] - y[j]
                d2 = dx ** 2 + dy ** 2
                I = I0 * np.exp(- d2 / r0 ** 2)
                
                # Contribution for particle j.
                I_particle[j] += np.sum(I)

                # Contribution for nn of particle j.
                I_particle[nn] += I
                   
    return I_particle


def save_plot(x, y, phi, step, rp, vp, L, dt, delayType):
    plt.figure(figsize=(6, 6))
    
    # Ploting robots.
    plt.scatter(x, y, c='b', alpha=0.4, edgecolors='k', s=rp*1000)

    # Ploting circles with radius r0.
    for i in range(len(x)):
        circle = plt.Circle((x[i], y[i]), 3 * rp, color='red', fill=False, linestyle='-', alpha=0.5)
        plt.gca().add_artist(circle)

    # Adding lines to show orientation.
    for i in range(len(x)):
        plt.plot(
            [x[i], x[i] + vp * np.cos(phi[i])],
            [y[i], y[i] + vp * np.sin(phi[i])],
            color='green',
            linewidth=1
        )
    
    
    plt.xlim(-L / 2, L / 2)
    plt.ylim(-L / 2, L / 2)
    plt.title(f'Time {step * dt:.0f} with {delayType} delay')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.gca().set_aspect('equal', adjustable='box')  # Asserting that the circles are round.
    
    # Spara plott som bild.
    plt.savefig(f'plots/e2/snapshot_step_{step*dt:.0f}_{delayType}_delay.png')
    plt.close()

def find_clusters(positions, r_c):
    """
    Identifierar kluster av robotar baserat på deras positioner och ett gränsavstånd.
    
    Parameters
    ==========
    positions : np.ndarray
        En N x 2 matris med robotarnas positioner (x och y).
    r_c : float
        Avståndströskel för att avgöra om två robotar tillhör samma kluster.

    Returns
    =======
    clusters : list of lists
        En lista där varje element är en lista med index för robotar i samma kluster.
    """
    N = positions.shape[0]
    dist_matrix = distance.cdist(positions, positions)  # Avståndsmatris
    visited = np.zeros(N, dtype=bool)  # Spårar besökta noder
    clusters = []

    for i in range(N):
        if not visited[i]:
            cluster = []
            to_visit = [i]
            while to_visit:
                current = to_visit.pop()
                if not visited[current]:
                    visited[current] = True
                    cluster.append(current)
                    # Lägg till alla grannar inom r_c som ännu inte är besökta
                    neighbors = np.where(dist_matrix[current] <= r_c)[0]
                    to_visit.extend(neighbors[~visited[neighbors]])
            clusters.append(cluster)
    return clusters




#%%

N = 50              # Number of robots
tau = 1             # Timescale of the orientation diffusion.
I0 = 1              # Maximum intensity.
r0 = 0.3            # Standard deviation of the Gaussian light intensity zone [m].
Ic = 0.1            # Intensity scale where the speed decays.
vInf= 0.01          # Self-propulsion speed at I=+infty [m/s].
v0 = 0.1            # Self-propulsion speed at I=0 [m/s].
L = 30 * r0         # Arena side [m]
dt = 0.05           # Time step.
delta = 5 * tau    # Delay [s] 

rc = 4 * r0         # Cut-off radius [m].

#Random position and orientation. 
x = (np.random.rand(N)-0.5)*L
y = (np.random.rand(N)-0.5)*L
phi = (np.random.rand(N)-0.5) * 2 * np.pi 

# Coefficients for the finite difference solution.
c_noise_phi = np.sqrt(2 * dt / tau)
n_fit = 5
if delta < 0:
    # Negative delay.
     
    I_fit = np.zeros([n_fit, N])
    t_fit = np.arange(n_fit) * dt
    dI_dt = np.zeros(N)
    # Initialize.
    I_ref = I0 * np.exp(- (x ** 2 + y ** 2) / r0 ** 2)
    for i in range(n_fit):
        I_fit[i, :] += I_ref   
        
if delta > 0:
    # Positive delay.
    n_delay = int(delta / dt)  # Delay in units of time steps.
    I_memory = np.zeros([n_delay, N])
    # Initialize.
    I_ref = I0 * np.exp(- (x ** 2 + y ** 2) / r0 ** 2)
    for i in range(n_fit):
        I_memory[i, :] += I_ref   
        
save_times = [0, 10, 100, 500, 1000]  # Timesteps for plotting
save_indices = [int(t/dt) for t in save_times]
plotNo = 1

window_size = 600

rp = r0 / 3
vp = rp  # Length of the arrow indicating the velocity direction.
line_width = 1  # Width of the arrow line.

animationOn = False

N_skip = 2

tk = Tk()
if animationOn:    
    tk.geometry(f'{window_size + 20}x{window_size + 20}')
    tk.configure(background='#000000')

    canvas = Canvas(tk, background='#ECECEC')  # Generate animation window 
    tk.attributes('-topmost', 0)
    canvas.place(x=10, y=10, height=window_size, width=window_size)

    light_spots = []
    for j in range(N):
        light_spots.append(
            canvas.create_oval(
                (x[j] - r0) / L * window_size + window_size / 2, 
                (y[j] - r0) / L * window_size + window_size / 2,
                (x[j] + r0) / L * window_size + window_size / 2, 
                (y[j] + r0) / L * window_size + window_size / 2,
                outline='#FF8080', 
            )
        )
        
    particles = []
    for j in range(N):
        particles.append(
            canvas.create_oval(
                (x[j] - rp) / L * window_size + window_size / 2, 
                (y[j] - rp) / L * window_size + window_size / 2,
                (x[j] + rp) / L * window_size + window_size / 2, 
                (y[j] + rp) / L * window_size + window_size / 2,
                outline='#000000', 
                fill='#A0A0A0',
            )
        )

    velocities = []
    for j in range(N):
        velocities.append(
            canvas.create_line(
                x[j] / L * window_size + window_size / 2, 
                y[j] / L * window_size + window_size / 2,
                (x[j] + vp * np.cos(phi[j])) / L * window_size + window_size / 2, 
                (y[j] + vp * np.cos(phi[j])) / L * window_size + window_size / 2,
                width=line_width, 
            )
        )

step = 0

def stop_loop(event):
    global running
    running = False
tk.bind("<Escape>", stop_loop)  # Bind the Escape key to stop the loop.
running = True  # Flag to control the loop.
while running and step*dt <= 1000:
    
    # Calculate current I.
    I_particles = calculate_intensity(x, y, I0, r0, L, rc)
    
    if delta < 0:
        # Estimate the derivative of I linear using the last n_fit values.
        delayType = "negative"
        for i in range(N - 1):
            # Update I_fit.
            I_fit = np.roll(I_fit, -1, axis=0)
            I_fit[-1, :] = I_particles
            # Fit to determine the slope.
            for j in range(N):
                p = np.polyfit(t_fit, I_fit[:, j], 1)
                dI_dt[j] = p[0]
            # Determine forecast. Remember that here delta is negative.
            I = I_particles - delta * dI_dt  
            I[np.where(I < 0)[0]] = 0
    elif delta > 0:
        # Update I_memory.
        delayType = "positive"
        I_memory = np.roll(I_memory, -1, axis=0)
        I_memory[-1, :] = I_particles    
        I = I_memory[0, :]
    else:
        I = I_particles
        delayType = "no"
    #if step % 200 == 0:
    #    print(f"Iteration: {step}")
    # Spara stillbild om tidssteget matchar.
    if step in save_indices:
        save_plot(x, y, phi, step, rp, vp,L,dt, delayType)
        print(f"plot no {plotNo} done")
        plotNo +=1   

    if step == save_indices[2] or step == save_indices[3]:
        positions = np.column_stack((x, y))  # Kombinera x och y till en positionsmatris
        clusters = find_clusters(positions, r0)  # Hitta kluster
        colors = mpl.colormaps['tab20']
        large_clusters = [cluster for cluster in clusters if len(cluster) >= 2]
        num_clusters = len(large_clusters)
        plt.figure(figsize=(6, 6))
        # Plotta varje kluster i en unik färg
        for i, cluster in enumerate(large_clusters):
            cluster_positions = positions[cluster]
            color = colors(i)
            
            plt.scatter(cluster_positions[:, 0], cluster_positions[:, 1],color= color, label=f"Cluster {i+1}", edgecolors='k')

            for pos in cluster_positions:
                circle = plt.Circle((pos[0], pos[1]), r0, color=color, fill=False, linestyle='-', linewidth=1.5)
                plt.gca().add_artist(circle)
            
        
        plt.xlim(-L / 2, L / 2)
        plt.ylim(-L / 2, L / 2)
        plt.title(f'Cluster at t =  {step * dt:.0f} with {delayType} delay\n Number of clusters: {num_clusters}')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.axis('equal')
        #plt.tight_layout(rect=[0, 0, 0.8, 1])
        plt.savefig(f'plots/e2/clusters_step_{step*dt:.0f}_{delayType}_delay.png')
        print("Clusterplot done")
        

        
    # Calculate new positions and orientations. 
    v = vInf + (v0 - vInf) * np.exp(- I / Ic) 
    nx = x + v * dt * np.cos(phi)
    ny = y + v * dt * np.sin(phi)
    nphi = phi + c_noise_phi * np.random.normal(0, 1, N)


    # Apply pbc.
    nx, ny = pbc(nx, ny, L)
   

    # Update animation frame.
    if step % N_skip == 0 and animationOn:        
                    
        for j, light_spot in enumerate(light_spots):
            canvas.coords(
                light_spot,
                (nx[j] - r0) / L * window_size + window_size / 2,
                (ny[j] - r0) / L * window_size + window_size / 2,
                (nx[j] + r0) / L * window_size + window_size / 2,
                (ny[j] + r0) / L * window_size + window_size / 2,
            )
                    
        for j, particle in enumerate(particles):
            canvas.coords(
                particle,
                (nx[j] - rp) / L * window_size + window_size / 2,
                (ny[j] - rp) / L * window_size + window_size / 2,
                (nx[j] + rp) / L * window_size + window_size / 2,
                (ny[j] + rp) / L * window_size + window_size / 2,
            )

        for j, velocity in enumerate(velocities):
            canvas.coords(
                velocity,
                nx[j] / L * window_size + window_size / 2,
                ny[j] / L * window_size + window_size / 2,
                (nx[j] + vp * np.cos(nphi[j])) / L * window_size + window_size / 2,
                (ny[j] + vp * np.sin(nphi[j])) / L * window_size + window_size / 2,
            )

        
    
        tk.title(f'Time {step * dt:.1f} - Iteration {step}')
        tk.update_idletasks()
        tk.update()
        time.sleep(.001)  # Increase to slow down the simulation.   
    

    step += 1
    x[:] = nx[:]
    y[:] = ny[:]
    phi[:] = nphi[:]  
if animationOn:
    tk.update_idletasks()
    tk.update()
    tk.mainloop()  # Release animation handle (close window to finish).


# %%
