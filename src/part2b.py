#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from part1 import step_rk4, meth_n_step

# -----------------------------------------------------------------------------
# 1. Pendule simple : fréquence vs angle initial
# -----------------------------------------------------------------------------

# --- Constantes physiques ---
g = 9.81  # gravité
l = 1.0   # longueur des tiges

def pendulum_eq_f(y, t):
    """
    y = [theta, omega]
    """
    theta, omega = y
    return np.array([omega, -g / l * np.sin(theta)])

def find_frequency(theta0):
    """
    Fréquence d’oscillation pour angle initial theta0,
    via pas fixe + RK4.
    """
    N = 2000
    T_sim = 20.0
    h = T_sim / N
    y0 = np.array([theta0, 0.0])
    ts, ys = meth_n_step(y0, 0.0, N, h, pendulum_eq_f, step_rk4)
    theta = ys[:, 0]
    max_idx = argrelextrema(theta, np.greater)[0]
    if len(max_idx) < 2:
        return np.nan
    T = np.mean(np.diff(ts[max_idx]))
    return 2 * np.pi / T

def plot_frequency_vs_theta():
    """
    Trace la fréquence mesurée en fonction de l’angle initial.
    """
    theta_vals = np.linspace(0.01, np.pi - 0.1, 50)
    freqs = [find_frequency(th) for th in theta_vals]

    plt.figure()
    plt.plot(theta_vals, freqs, label="Mesurée")
    plt.axhline(np.sqrt(g / l), color='r', linestyle='--', label="Linéraire")
    plt.xlabel("θ₀ (rad)")
    plt.ylabel("Fréquence (rad/s)")
    plt.title("Fréquence vs angle initial")
    plt.legend()
    plt.grid()
    plt.savefig("pendule_frequence_theta0.png", dpi=200)
    plt.show()

# -----------------------------------------------------------------------------
# 2. Pendule double : trajectoires pour conditions proches
# -----------------------------------------------------------------------------

m = 1.0  # masse identique pour les deux masses

def double_pendulum_eq_f(y, t):
    """
    y = [θ1, ω1, θ2, ω2]
    """
    θ1, ω1, θ2, ω2 = y
    Δ = θ2 - θ1
    denom = 2*m - m*np.cos(2*Δ)

    dθ1 = ω1
    dθ2 = ω2

    dω1 = (
        m*g*np.sin(θ2)*np.cos(Δ)
        - m*np.sin(Δ)*(l*ω2**2*np.cos(Δ) + l*ω1**2)
        - 2*m*g*np.sin(θ1)
    ) / (l * denom)

    dω2 = (
        2*np.sin(Δ)
        * (ω1**2*l*m + g*m*np.cos(θ1) + ω2**2*l*m*np.cos(Δ))
    ) / (l * denom)

    return np.array([dθ1, dω1, dθ2, dω2])

def simulate_double_pendulum(θ1_0, θ2_0, T=20.0, steps=3000):
    """
    Intègre double_pendulum_eq_f à pas fixe et trace (x2, y2).
    """
    h = T / steps
    y0 = np.array([θ1_0, 0.0, θ2_0, 0.0])
    ts, ys = meth_n_step(y0, 0.0, steps, h, double_pendulum_eq_f, step_rk4)

    θ1 = ys[:, 0]
    θ2 = ys[:, 2]
    x1 = l * np.sin(θ1)
    y1 = -l * np.cos(θ1)
    x2 = x1 + l * np.sin(θ2)
    y2 = y1 - l * np.cos(θ2)

    plt.plot(x2, y2, lw=0.8, label=f"Δθ₂ = {θ2_0 - np.pi/2:.3f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid()

def plot_multiple_trajectories():
    """
    Trace plusieurs trajectoires pour conditions initiales proches.
    """
    plt.figure(figsize=(8,6))
    base = np.pi/2
    for δ in [0.0, 0.001, 0.002, 0.003]:
        simulate_double_pendulum(base, base + δ)
    plt.title("Trajectoires pour conditions initiales proches")
    plt.legend()
    plt.tight_layout()
    plt.savefig("trajectoires_double_pendule.png", dpi=200)
    plt.show()

# -----------------------------------------------------------------------------
# 3. Carte fractale du temps de premier retournement
# -----------------------------------------------------------------------------

def first_flip_time(theta1_0, theta2_0, max_time=10.0):
    """
    Renvoie t tel que |θ₂|>π via pas fixe, ou NaN sinon.
    """
    h = 0.01
    N = int(max_time / h)
    y0 = np.array([theta1_0, 0.0, theta2_0, 0.0])
    ts, ys = meth_n_step(y0, 0.0, N, h, double_pendulum_eq_f, step_rk4)
    for ti, state in zip(ts, ys):
        if abs(state[2]) > np.pi:
            return ti
    return np.nan

def plot_flip_time_map(N=200, max_time=10.0):
    thetas = np.linspace(-np.pi, np.pi, N)
    T_map = np.empty((N, N))
    for i, t1 in enumerate(thetas):
        for j, t2 in enumerate(thetas):
            T_map[i, j] = first_flip_time(t1, t2, max_time)

    T_mask = np.ma.masked_invalid(T_map)
    cmap = plt.cm.RdYlGn.reversed()
    cmap.set_bad(color='white')

    plt.figure(figsize=(6,6))
    plt.imshow(
        T_mask, origin='lower',
        extent=[-np.pi, np.pi, -np.pi, np.pi],
        cmap=cmap, aspect='auto', vmin=0, vmax=max_time
    )
    plt.colorbar(label="Temps avant retournement (s)")
    plt.xlabel("θ₁ initial")
    plt.ylabel("θ₂ initial")
    plt.title("Carte fractale du temps de retournement")
    plt.tight_layout()
    plt.savefig("flip_time_map_fractal.png", dpi=200)
    plt.show()

# -----------------------------------------------------------------------------
# Entrée principale
# -----------------------------------------------------------------------------

def main():
    plot_frequency_vs_theta()
    plot_multiple_trajectories()
    print("Génération de la carte fractale du temps de retournement…")
    plot_flip_time_map(N=150, max_time=8.0)

if __name__ == "__main__":
    main()
