from part1 import CauchyProblem, step_rk4, draw_vector_field, meth_n_step
import matplotlib.pyplot as plt
import numpy as np

# --------- Paramètres des modèles ---------
gamma = 0.5  # taux 
c = 100      # capacité 

# Fonctions différentielles 
def f_malthus(N, t):
    return gamma * N

def f_verhulst(N, t):
    return gamma * N * (1 - N / c)

# Solutions exactes
def sol_malthus_exact(N0, t):
    return N0 * np.exp(gamma * t)

def sol_verhulst_exact(N0, t):
    return (c * N0 * np.exp(gamma * t)) / (c + N0 * (np.exp(gamma * t) - 1))

def lotka_volterra(y, t):
    N, P = y
    dNdt = N * (a - b * P)
    dPdt = P * (c * N - d)
    return np.array([dNdt, dPdt])

if __name__ == "__main__":
    # Simulation pour plusieurs N0 
    t0  = 0
    tf = 20
    N_steps = 200
    h = (tf - t0) / N_steps
    N0_list = [5, 20, 90]  

    # Modèle de Malthus et Verhulst
    plt.figure(figsize=(12, 5))
    t_vals = np.linspace(t0, tf, N_steps+1)

    # Modèle de Malthus 
    plt.subplot(1, 2, 1)
    for N0 in N0_list:
        ts, Ns = meth_n_step(np.array([N0]), t0, N_steps, h, f_malthus, step_rk4)
        plt.plot(ts, Ns[:, 0], label=f"N₀ = {N0} (num)")
        plt.plot(t_vals, sol_malthus_exact(N0, t_vals), '--', label=f"N₀ = {N0} (exacte)")
    plt.title("Modèle de Malthus (croissance exponentielle)")
    plt.xlabel("Temps")
    plt.ylabel("Population N(t)")
    plt.grid()
    plt.legend()

    # Modèle de Verhulst 
    plt.subplot(1, 2, 2)
    for N0 in N0_list:
        ts, Ns = meth_n_step(np.array([N0]), t0, N_steps, h, f_verhulst, step_rk4)
        plt.plot(ts, Ns[:, 0], label=f"N₀ = {N0} (num)")
        plt.plot(t_vals, sol_verhulst_exact(N0, t_vals), '--', label=f"N₀ = {N0} (exacte)")
    plt.title("Modèle de Verhulst (croissance logistique)")
    plt.xlabel("Temps")
    plt.ylabel("Population N(t)")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


    # Paramètres pour Lotka-Volterra
    a = 1.2
    b = 0.6
    c = 0.8 
    d = 0.3

    # Conditions initiales
    y0 = np.array([10.0, 5.0])  # N(0), P(0)
    t0 = 0
    tf = 100
    N_steps = 2000
    h = (tf - t0) / N_steps

    # Intégration Lotka-Volterra
    ts, ys = meth_n_step(y0, t0, N_steps, h, lotka_volterra, step_rk4)
    N_vals = ys[:, 0]
    P_vals = ys[:, 1]

    # Tracé de N(t) et P(t)
    plt.figure()
    plt.plot(ts, N_vals, label="Proies N(t)")
    plt.plot(ts, P_vals, label="Prédateurs P(t)")
    plt.xlabel("Temps")
    plt.ylabel("Population")
    plt.title("Évolution temporelle des populations (Lotka-Volterra)")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("lotka_time_series.png", dpi=200)
    plt.show()

    # --- Plan de phase Lotka–Volterra ---
    plt.figure()
    plt.plot(N_vals, P_vals, '-', label="Trajectoire centrale")
    plt.xlabel("Proies N")
    plt.ylabel("Prédateurs P")
    plt.title("Plan de phase du modèle de Lotka–Volterra")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig("lotka_phase.png", dpi=200)
    plt.show()

    # Période approchée 
    peaks_indices = []
    for i in range(1, len(N_vals) - 1):
        if N_vals[i - 1] < N_vals[i] and N_vals[i] > N_vals[i + 1]:
            peaks_indices.append(i)

    if len(peaks_indices) >= 2:
        times_peaks = ts[peaks_indices]
        periods = np.diff(times_peaks)
        estimated_period = np.mean(periods)
        print(f"Période estimée ≈ {estimated_period}")
    else:
        print("Pas assez de pics détectés pour estimer la période.")

    # -------------------------------------------------------------------------
    #  Comportement local autour de y0 = (10,5)
    # -------------------------------------------------------------------------
    problem = CauchyProblem(y0, t0, lotka_volterra)
    x_min, x_max = y0[0] - 2, y0[0] + 2
    y_min, y_max = y0[1] - 2, y0[1] + 2
    ax = draw_vector_field(problem, x_range=(x_min, x_max), y_range=(y_min, y_max))

    ts_c, ys_c = meth_n_step(y0, t0, N_steps, h, lotka_volterra, step_rk4)
    ax.plot(ys_c[:, 0], ys_c[:, 1], 'k-', linewidth=2, label="Trajectoire centrale y0")
    for angle in np.linspace(0, 2 * np.pi, 4):
        perturb = 0.5 * np.array([np.cos(angle), np.sin(angle)])
        ts_p, ys_p = meth_n_step(y0 + perturb, t0, N_steps, h, lotka_volterra, step_rk4)
        ax.plot(ys_p[:, 0], ys_p[:, 1], '-', alpha=0.7, label="y0 perturbé")

    plt.title("Comportement local autour de y0=(10,5)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lotka_vector_field_y0.png", dpi=200)
    plt.show()

    # -------------------------------------------------------------------------
    #  Comportement autour de l’équilibre non trivial
    # -------------------------------------------------------------------------
    y_eq = np.array([d / c, a / b])
    problem_eq = CauchyProblem(y_eq, t0, lotka_volterra)
    x_min, x_max = y_eq[0] - 2, y_eq[0] + 2
    y_min, y_max = y_eq[1] - 2, y_eq[1] + 2
    ax = draw_vector_field(problem_eq, x_range=(x_min, x_max), y_range=(y_min, y_max))

    ts_e, ys_e = meth_n_step(y_eq, t0, N_steps, h, lotka_volterra, step_rk4)
    ax.plot(ys_e[:, 0], ys_e[:, 1], 'k-', label="Point singulier", markersize=6)
    for angle in np.linspace(0, 2 * np.pi, 4):
        perturb = 0.1 * np.array([np.cos(angle), np.sin(angle)])
        ts_pe, ys_pe = meth_n_step(y_eq + perturb, t0, N_steps, h, lotka_volterra, step_rk4)
        ax.plot(ys_pe[:, 0], ys_pe[:, 1], '-', alpha=0.7, label="y0 perturbé")

    plt.title("Autour de l’équilibre (d/c, a/b)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lotka_vector_field_eq.png", dpi=200)
    plt.show()
