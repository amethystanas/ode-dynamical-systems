import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# --- Paramètres physiques globaux ---
g = 9.81  # gravité (m/s²)
l = 1.0   # longueur des tiges (m)
m = 1.0   # masse (kg)
dt = 0.02  # pas de temps (s)
T = 20     # durée totale de la simulation (s)

def double_pendulum_equations(t, y):
    """
    Équations différentielles du pendule double :
        y = [theta1, omega1, theta2, omega2]
    
    Retourne les dérivées [dtheta1/dt, domega1/dt, dtheta2/dt, domega2/dt]
    """
    theta1, omega1, theta2, omega2 = y
    delta = theta2 - theta1
    denom = (2 * m - m * np.cos(2 * delta))

    dtheta1 = omega1
    dtheta2 = omega2

    domega1 = (m * g * np.sin(theta2) * np.cos(delta)
               - m * np.sin(delta) * (l * omega2**2 * np.cos(delta) + l * omega1**2)
               - 2 * m * g * np.sin(theta1)) / (l * denom)

    domega2 = (2 * np.sin(delta) * (omega1**2 * l * m + g * m * np.cos(theta1)
               + omega2**2 * l * m * np.cos(delta))) / (l * denom)

    return [dtheta1, domega1, dtheta2, domega2]

def simulate_pendulum(theta1_0, theta2_0):
    """
    Simule le pendule double à partir des angles initiaux.
    
    Retourne les trajectoires x1, y1, x2, y2 ainsi que la grille temporelle.
    """
    y0 = [theta1_0, 0.0, theta2_0, 0.0]
    t_eval = np.arange(0, T, dt)
    sol = solve_ivp(double_pendulum_equations, [0, T], y0, t_eval=t_eval)
    
    theta1, theta2 = sol.y[0], sol.y[2]
    x1 = l * np.sin(theta1)
    y1 = -l * np.cos(theta1)
    x2 = x1 + l * np.sin(theta2)
    y2 = y1 - l * np.cos(theta2)

    return x1, y1, x2, y2, t_eval

def animate_pendulum(x1, y1, x2, y2, t_eval):
    """
    Anime le pendule double à partir des trajectoires données.
    """
    fig, ax = plt.subplots()
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal')
    ax.grid()
    ax.set_title("Pendule double (animation)")

    # Trajectoires et masses
    line, = ax.plot([], [], 'o-', lw=2, color='gray')
    mass1, = ax.plot([], [], 'bo', markersize=12)
    mass2, = ax.plot([], [], 'bo', markersize=12)

    def init():
        line.set_data([], [])
        mass1.set_data([], [])
        mass2.set_data([], [])
        return line, mass1, mass2

    def update(i):
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        line.set_data(thisx, thisy)
        mass1.set_data([thisx[1]], [thisy[1]])
        mass2.set_data([thisx[2]], [thisy[2]])
        return line, mass1, mass2

    ani = animation.FuncAnimation(
        fig, update, frames=len(t_eval),
        init_func=init, blit=True, interval=dt * 1000
    )

    plt.show()

def main():
    """
    Point d’entrée principal.
    Lance la simulation et l’animation du pendule double.
    """
    theta1_0 = np.pi / 2
    theta2_0 = np.pi / 2 + 0.01
    x1, y1, x2, y2, t_eval = simulate_pendulum(theta1_0, theta2_0)
    animate_pendulum(x1, y1, x2, y2, t_eval)

if __name__ == "__main__":
    main()
