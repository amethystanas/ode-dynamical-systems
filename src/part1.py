import numpy as np
import matplotlib.pyplot as plt

class CauchyProblem:
    """
    Représente un problème de Cauchy :
        y(t0) = y0
        y'(t) = f(y(t), t)

    Attributs :
        f   : fonction représentant l'équation différentielle, f(y, t)
        y0  : vecteur ou scalaire représentant la condition initiale
        t0  : instant initial
    """
    def __init__(self, y0, t0, f):
        self.y0 = y0      # condition initiale
        self.t0 = t0      # temps initial
        self.f = f        # fonction différentielle
        self.dim = self.y0.shape[0] if len(self.y0.shape) > 0 else 1

### Methodes de résolution ###

def step_euler(y, t, h, f):
    """
    Méthode d'euler
    """
    return y + h * f(y, t)

def step_milieu(y, t, h, f):
    """
    Méthode du p milieu
    """
    k1 = f(y, t)
    k2 = f(y + 0.5 * h * k1, t + 0.5 * h)
    return y + h * k2

def step_heun(y, t, h, f):
    """
    Méthode de Heun
    """
    k1 = f(y, t)
    k2 = f(y + h * k1, t + h)
    return y + h * 0.5 * (k1 + k2)

def step_rk4(y, t, h, f):
    """
    Méthode de Runge-Kutta d’ordre 4.
    """
    k1 = f(y, t)
    k2 = f(y + 0.5 * h * k1, t + 0.5 * h)
    k3 = f(y + 0.5 * h * k2, t + 0.5 * h)
    k4 = f(y + h * k3, t + h)
    return y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)


def meth_n_step(y0, t0, N, h, f, meth):
    """
    Methode calculant un nombre N de pas de taille constante h  
    """
    ys = [y0]
    ts = [t0]
    y = y0
    t = t0
    for _ in range(N):
        y = meth(y, t, h, f)
        t += h
        ys.append(y)
        ts.append(t)
    return np.array(ts), np.array(ys)


def meth_epsilon(y0, t0, tf, eps, f, meth):
    """
    Méthode avec une solution approchée avec un paramétre d'erreur epsilon
    """
    ys = [y0]
    ts = [t0]
    y = y0
    t = t0
    h = (tf - t0) / 10  

    while t < tf:
        if t + h > tf:
            h = tf - t

        y_big = meth(y, t, h, f)
        y_half = meth(y, t, h / 2, f)
        y_small = meth(y_half, t + h / 2, h / 2, f)

        error = np.linalg.norm(y_small - y_big, ord=np.inf)

        if error < eps:
            t += h
            y = y_small
            ts.append(t)
            ys.append(y)
            h *= 1.5  
        else:
            h *= 0.5  

    return np.array(ts), np.array(ys)


def draw_vector_field(problem, x_range, y_range, d=20):
    """
    Dessine le champ des tangentes d'une équation différentielle en dimension 2
    """

    if problem.dim != 2:
        raise ValueError("Le champ des tangentes ne peut être tracé qu'en dimension 2")
    
    x = np.linspace(x_range[0], x_range[1], d)
    y = np.linspace(y_range[0], y_range[1], d)
    X, Y = np.meshgrid(x, y)
    
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(d):
        for j in range(d):
            p = np.array([X[i, j], Y[i, j]])
            derivatives = problem.f(p, 0)  # on prend le temps égal 0
            U[i, j] = derivatives[0]
            V[i, j] = derivatives[1]
    
    norm = np.sqrt(U**2 + V**2)

    # Éviter la division par zéro
    for i in range(len(norm[0])):
        for j in range(len(norm[1])):
            if norm[i, j] == 0:
                norm[i, j] = 1

    U = U / norm
    V = V / norm
    
    # dessine le champ de tangente
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=30, width=0.003)
    plt.grid(True)
    plt.xlabel('y_1')
    plt.ylabel('y_2')
    plt.title('Champ des tangentes')
    plt.xlim(x_range)
    plt.ylim(y_range)
    
    return plt.gca()


if __name__ == "__main__":
    # y'(t) = y(t)/(1+t²), y(0) = 1
    # Solution exacte: y(t) = exp(arctan(t))
    
    print("Test 1")
    
    # Définition du problème
    y0_1 = np.array([1.0])
    t0_1 = 0.0
    def f1(y, t):
        return np.array([y[0]/(1+t**2)])
    problem1 = CauchyProblem(y0_1, t0_1, f1)
    
    # Solution exacte
    def exact_sol_1(t):
        return np.exp(np.arctan(t))
    
    # Résolution avec différentes méthodes
    tf_1 = 10.0
    N_1 = 200
    h_1 = (tf_1 - t0_1) / N_1
    
    # Méthode d'Euler
    ts_euler_1, ys_euler_1 = meth_n_step(problem1.y0, problem1.t0, N_1, h_1, problem1.f, step_euler)
    
    # Méthode du milieu
    ts_mid_1, ys_mid_1 = meth_n_step(problem1.y0, problem1.t0, N_1, h_1, problem1.f, step_milieu)
    
    # Méthode de Heun
    ts_heun_1, ys_heun_1 = meth_n_step(problem1.y0, problem1.t0, N_1, h_1, problem1.f, step_heun)
    
    # Méthode RK4
    ts_rk4_1, ys_rk4_1 = meth_n_step(problem1.y0, problem1.t0, N_1, h_1, problem1.f, step_rk4)
    
    exact_y_1 = np.array([exact_sol_1(t) for t in ts_euler_1])
    
    # trace des solutions
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(ts_euler_1, ys_euler_1[:, 0], 'b-', label="Euler")
    plt.plot(ts_mid_1, ys_mid_1[:, 0], 'g--', label=" Milieu")
    plt.plot(ts_heun_1, ys_heun_1[:, 0], 'm-.', label="Heun")
    plt.plot(ts_rk4_1, ys_rk4_1[:, 0], 'r:', label="RK4")
    plt.plot(ts_euler_1, exact_y_1, 'k-', label="Solution exacte")
    plt.grid(True)
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('y(t)')
    plt.title('Comparaison pour y\'(t) = y(t)/(1+t²), y(0) = 1')
    
    # Calcul des erreurs
    err_euler_1 = np.abs(ys_euler_1[:, 0] - exact_y_1)
    err_mid_1 = np.abs(ys_mid_1[:, 0] - exact_y_1)
    err_heun_1 = np.abs(ys_heun_1[:, 0] - exact_y_1)
    err_rk4_1 = np.abs(ys_rk4_1[:, 0] - exact_y_1)
    
    # trace des erreurs (echelle log)
    plt.subplot(2, 1, 2)
    plt.semilogy(ts_euler_1, err_euler_1, 'b-', label="Erreur Euler")
    plt.semilogy(ts_mid_1, err_mid_1, 'g--', label="Erreur Milieu")
    plt.semilogy(ts_heun_1, err_heun_1, 'm-.', label="Erreur Heun")
    plt.semilogy(ts_rk4_1, err_rk4_1, 'r:', label="Erreur RK4")
    plt.grid(True)
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Erreur absolu')
    plt.title('Erreurs des methodes numériques (Équation 1)')
    
    # Affichage des erreurs maximales
    print(f"Erreur max Euler: {np.max(err_euler_1)}")
    print(f"Erreur max Milieu: {np.max(err_mid_1)}")
    print(f"Erreur max Heun: {np.max(err_heun_1)}")
    print(f"Erreur max RK4: {np.max(err_rk4_1)}")
    
    # y'(t) = [-y2(t), y1(t)], y(0) = [1, 0]
    # Solution exacte: y(t) = [cos(t), sin(t)]
    
    print("\nTest 2")
    
    y0_2 = np.array([1.0, 0.0])
    t0_2 = 0.0
    def f2(y, t):
        return np.array([-y[1], y[0]])
    problem2 = CauchyProblem(y0_2, t0_2, f2)
    
    # Solution exacte
    def exact_sol_2(t):
        return np.array([np.cos(t), np.sin(t)])
    
    # Résolution avec différentes méthodes
    tf_2 = 10.0
    N_2 = 200
    h_2 = (tf_2 - t0_2) / N_2
    
    # Méthode d'Euler
    ts_euler_2, ys_euler_2 = meth_n_step(problem2.y0, problem2.t0, N_2, h_2, problem2.f, step_euler)
    
    # Méthode du p milieu
    ts_mid_2, ys_mid_2 = meth_n_step(problem2.y0, problem2.t0, N_2, h_2, problem2.f, step_milieu)
    
    # Méthode de Heun
    ts_heun_2, ys_heun_2 = meth_n_step(problem2.y0, problem2.t0, N_2, h_2, problem2.f, step_heun)
    
    # Méthode RK4
    ts_rk4_2, ys_rk4_2 = meth_n_step(problem2.y0, problem2.t0, N_2, h_2, problem2.f, step_rk4)
    
    # Solution exacte aux temps de la grille
    exact_y_2 = np.array([exact_sol_2(t) for t in ts_euler_2])
    
    # Tracé des solutions
    plt.figure(figsize=(15, 12))
    
    # Tracé de y1(t)
    plt.subplot(2, 2, 1)
    plt.plot(ts_euler_2, ys_euler_2[:, 0], 'b-', label="Euler")
    plt.plot(ts_mid_2, ys_mid_2[:, 0], 'g--', label=" Milieu")
    plt.plot(ts_heun_2, ys_heun_2[:, 0], 'm-.', label="Heun")
    plt.plot(ts_rk4_2, ys_rk4_2[:, 0], 'r:', label="RK4")
    plt.plot(ts_euler_2, exact_y_2[:, 0], 'k-', label="Exacte")
    plt.grid(True)
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('y1(t)')
    plt.title('Composante y1(t)')
    
    # Trace de y2(t)
    plt.subplot(2, 2, 2)
    plt.plot(ts_euler_2, ys_euler_2[:, 1], 'b-', label="Euler")
    plt.plot(ts_mid_2, ys_mid_2[:, 1], 'g--', label=" Milieu")
    plt.plot(ts_heun_2, ys_heun_2[:, 1], 'm-.', label="Heun")
    plt.plot(ts_rk4_2, ys_rk4_2[:, 1], 'r:', label="RK4")
    plt.plot(ts_euler_2, exact_y_2[:, 1], 'k-', label="Exacte")
    plt.grid(True)
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('y2(t)')
    plt.title('Composante y2(t)')
    
    plt.subplot(2, 2, 3)
    plt.plot(ys_euler_2[:, 0], ys_euler_2[:, 1], 'b-', label="Euler")
    plt.plot(ys_mid_2[:, 0], ys_mid_2[:, 1], 'g--', label=" Milieu")
    plt.plot(ys_heun_2[:, 0], ys_heun_2[:, 1], 'm:', label="Heun")
    plt.plot(ys_rk4_2[:, 0], ys_rk4_2[:, 1], 'r:', label="RK4")
    plt.plot(exact_y_2[:, 0], exact_y_2[:, 1], 'k-', label="Exacte")
    plt.grid(True)
    plt.legend()
    plt.xlabel('y1')
    plt.ylabel('y2')
    plt.title('Plan de phase')
    plt.axis('equal')
    
    # Calcul des erreur
    err_euler_2 = np.sqrt(np.sum((ys_euler_2 - exact_y_2)**2, axis=1))
    err_mid_2 = np.sqrt(np.sum((ys_mid_2 - exact_y_2)**2, axis=1))
    err_heun_2 = np.sqrt(np.sum((ys_heun_2 - exact_y_2)**2, axis=1))
    err_rk4_2 = np.sqrt(np.sum((ys_rk4_2 - exact_y_2)**2, axis=1))
    
    # Tracé des erreurs (échelle log)
    plt.subplot(2, 2, 4)
    plt.semilogy(ts_euler_2, err_euler_2, 'b-', label="Erreur Euler")
    plt.semilogy(ts_mid_2, err_mid_2, 'g--', label="Erreur  Milieu")
    plt.semilogy(ts_heun_2, err_heun_2, 'm-.', label="Erreur Heun")
    plt.semilogy(ts_rk4_2, err_rk4_2, 'r:', label="Erreur RK4")
    plt.grid(True)
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Erreur')
    plt.title('Erreurs des methodes numériques (Équation 2)')
    
    # erreurs maximales
    print(f"Erreur max Euler: {np.max(err_euler_2)}")
    print(f"Erreur max Milieu: {np.max(err_mid_2)}")
    print(f"Erreur max Heun: {np.max(err_heun_2)}")
    print(f"Erreur max RK4: {np.max(err_rk4_2)}")
    
    #champ des tangentes pour l'équation différentielle en dimension 2
    ax = draw_vector_field(problem2, (-1.5, 1.5), (-1.5, 1.5), d=20)
    
    ax.plot(ys_euler_2[:, 0], ys_euler_2[:, 1], 'b-', label="Euler", linewidth=1.5)
    ax.plot(ys_mid_2[:, 0], ys_mid_2[:, 1], 'g--', label="p Milieu", linewidth=1.5)
    ax.plot(ys_heun_2[:, 0], ys_heun_2[:, 1], 'm-.', label="Heun", linewidth=1.5)
    ax.plot(ys_rk4_2[:, 0], ys_rk4_2[:, 1], 'r:', label="RK4", linewidth=1.5)
    ax.plot(exact_y_2[:, 0], exact_y_2[:, 1], 'k-', label="Solution exacte", linewidth=2)
    ax.legend()
    ax.set_title('Champ des tangentes et trajectoires pour y\'(t) = [-y2(t), y1(t)]')
    
    plt.tight_layout()
    plt.show()
