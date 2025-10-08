

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import ipywidgets as widgets
import plotly.graph_objects as go

# -----------------------
# Funciones numéricas (FEM beam Euler-Bernoulli)
# -----------------------
def beam_element_stiffness(EI, L):
    return (EI / L**3) * np.array([
        [12.0, 6.0*L, -12.0, 6.0*L],
        [6.0*L, 4.0*(L**2), -6.0*L, 2.0*(L**2)],
        [-12.0, -6.0*L, 12.0, -6.0*L],
        [6.0*L, 2.0*(L**2), -6.0*L, 4.0*(L**2)]
    ])

def hermite_N(xi, L):
    N1 = 1 - 3*xi**2 + 2*xi**3
    N2 = L*(xi - 2*xi**2 + xi**3)
    N3 = 3*xi**2 - 2*xi**3
    N4 = L*(-xi**2 + xi**3)
    return np.array([N1, N2, N3, N4])

def hermite_d2N_dx2(xi, L):
    d2N1 = -6 + 12*xi
    d2N2 = L*(-4 + 6*xi)
    d2N3 = 6 - 12*xi
    d2N4 = L*(-2 + 6*xi)
    return np.array([d2N1, d2N2, d2N3, d2N4]) / (L**2)

def gauss_legendre_integration(func, a, b, n=6):
    xs, ws = np.polynomial.legendre.leggauss(n)
    t = 0.5*(b - a)*xs + 0.5*(b + a)
    w = 0.5*(b - a)*ws
    s = 0.0
    for ti, wi in zip(t, w):
        s = s + wi * func(ti)
    return s

def element_equivalent_loads_udl(q_val, L, a=0.0, b=None, nint=6):
    if b is None:
        b = L
    def integrand(x):
        if (x < a - 1e-12) or (x > b + 1e-12):
            return np.zeros(4)
        xi = x / L
        return hermite_N(xi, L) * q_val
    return gauss_legendre_integration(integrand, 0.0, L, n=nint)

def element_equivalent_loads_point(P, a, L):
    xi = a / L
    N = hermite_N(xi, L)
    return P * N

# -----------------------
# Modelo
# -----------------------
class Load:
    def __init__(self, kind, value, span_index, x_start=None, x_end=None, x_rel=None):
        self.kind = kind
        self.value = float(value)
        self.span = int(span_index)
        self.x_start = x_start
        self.x_end = x_end
        self.x_rel = x_rel

class BeamModel:
    def __init__(self, E, I, spans, supports, loads):
        self.E = float(E)
        self.I = float(I)
        self.spans = np.array(spans, dtype=float)
        self.n_spans = len(spans)
        self.n_nodes = self.n_spans + 1
        self.supports = supports[:]  # 'fixed','simple','free'
        self.loads = loads[:]
        self.ndof = 2 * self.n_nodes
        self.K = np.zeros((self.ndof, self.ndof))
        self.F = np.zeros(self.ndof)
        self.U = np.zeros(self.ndof)
        self.reactions = np.zeros(self.ndof)
        self.M_funcs = []
        self.V_funcs = []

    def assemble(self):
        EI = self.E * self.I
        self.K.fill(0.0)
        self.F.fill(0.0)
        for e in range(self.n_spans):
            L = self.spans[e]
            k_e = beam_element_stiffness(EI, L)
            idx = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
            for i in range(4):
                for j in range(4):
                    self.K[idx[i], idx[j]] += k_e[i,j]
            # loads on element
            fe_total = np.zeros(4)
            for ld in [l for l in self.loads if l.span == e]:
                if ld.kind == 'udl':
                    a = ld.x_start if ld.x_start is not None else 0.0
                    b = ld.x_end if ld.x_end is not None else L
                    fe = element_equivalent_loads_udl(ld.value, L, a=a, b=b, nint=6)
                    fe_total += fe
                elif ld.kind == 'point':
                    x_rel = float(ld.x_rel)
                    fe = element_equivalent_loads_point(ld.value, x_rel, L)
                    fe_total += fe
            for i in range(4):
                self.F[idx[i]] += fe_total[i]

    def apply_supports_and_solve(self):
        K = self.K.copy()
        F = self.F.copy()
        large = 1e20
        # penalty for supports (prescribed zero displacement/rotation)
        for node in range(self.n_nodes):
            sup = self.supports[node]
            if sup == 'fixed':
                K[2*node, 2*node] += large
                K[2*node+1, 2*node+1] += large
            elif sup == 'simple':
                K[2*node, 2*node] += large
        # solve linear system
        K = 0.5*(K + K.T)
        self.U = np.linalg.solve(K, F)
        self.reactions = self.K.dot(self.U) - self.F

    def postprocess(self, n_samples=200):
        EI = self.E * self.I
        self.M_funcs = []
        self.V_funcs = []
        for e in range(self.n_spans):
            L = self.spans[e]
            idx = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
            u_e = self.U[idx]
            xs = np.linspace(0, L, n_samples)
            Ms = np.zeros_like(xs)
            Vs = np.zeros_like(xs)
            for i, x in enumerate(xs):
                xi = x / L
                d2N = hermite_d2N_dx2(xi, L)
                curvature = np.dot(d2N, u_e)
                Ms[i] = EI * curvature
            # shear from derivative of M (match sizes)
            Vs[:-1] = np.diff(Ms) / np.diff(xs)
            Vs[-1] = Vs[-2]
            self.M_funcs.append((xs, Ms))
            self.V_funcs.append((xs, Vs))

    def get_node_results(self):
        out = []
        for n in range(self.n_nodes):
            v = self.U[2*n]
            th = self.U[2*n+1]
            Rv = self.reactions[2*n]
            Rm = self.reactions[2*n+1]
            out.append((v, th, self.supports[n], Rv, Rm))
        return out

# -----------------------
# INTERFAZ con ipywidgets (valores por defecto de tu viga)
# -----------------------

# Valores por defecto tomados de tus imágenes/datos
fc_default = 28.0
Ec_default_MPa = 5000.0 * np.sqrt(fc_default)   # MPa
E_default = Ec_default_MPa * 1e6                # Pa
I_mm4_default = 2.7762e9                        # mm^4 from your image
I_default = I_mm4_default * 1e-12               # m^4

# default geometry and loads (the viga you provided)
L1_default = 7.0
L2_default = 8.0
L3_default = 2.0
q_default = 10.0   # kN/m default; change as needed

# Widgets
E_w = widgets.FloatText(value=E_default, description='E [Pa]:', layout=widgets.Layout(width='360px'))
I_w = widgets.FloatText(value=I_default, description='I [m^4]:', layout=widgets.Layout(width='360px'))
L1_w = widgets.FloatText(value=L1_default, description='L1 [m]:', layout=widgets.Layout(width='180px'))
L2_w = widgets.FloatText(value=L2_default, description='L2 [m]:', layout=widgets.Layout(width='180px'))
L3_w = widgets.FloatText(value=L3_default, description='L3 [m]:', layout=widgets.Layout(width='180px'))
q_w  = widgets.FloatText(value=q_default, description='q [kN/m]:', layout=widgets.Layout(width='180px'))

# support options default: simple at nodes A,B,C,D
support0 = widgets.Dropdown(options=['simple','fixed','free'], value='simple', description='Node A:')
support1 = widgets.Dropdown(options=['simple','fixed','free'], value='simple', description='Node B:')
support2 = widgets.Dropdown(options=['simple','fixed','free'], value='simple', description='Node C:')
support3 = widgets.Dropdown(options=['simple','fixed','free'], value='simple', description='Node D:')

exag_w = widgets.FloatSlider(value=100.0, min=1.0, max=1000.0, step=1.0, description='Exag defl')

btn_calc = widgets.Button(description='Calcular y Mostrar', button_style='success')
out = widgets.Output(layout={'border': '1px solid black'})

left = widgets.VBox([E_w, I_w, q_w, exag_w, btn_calc])
right = widgets.VBox([widgets.HBox([L1_w, L2_w, L3_w]), widgets.HBox([support0, support1, support2, support3])])
ui = widgets.HBox([left, right])
display(ui, out)

# -----------------------
# Acción al presionar botón
# -----------------------
def on_calculate(_):
    with out:
        clear_output()
        # read values
        E = float(E_w.value)
        I = float(I_w.value)
        spans = [float(L1_w.value), float(L2_w.value), float(L3_w.value)]
        supports = [support0.value, support1.value, support2.value, support3.value]
        qkN = float(q_w.value)
        qN = qkN * 1e3

        # build loads: apply UDL full length of each span (default)
        loads = [Load('udl', qN, i, x_start=0.0, x_end=spans[i]) for i in range(len(spans))]

        # create model, assemble, solve
        model = BeamModel(E, I, spans, supports, loads)
        model.assemble()
        model.apply_supports_and_solve()
        model.postprocess(n_samples=300)

        # Print detailed results
        print("===== DATOS DE ENTRADA =====")
        print(f"E = {E:.6e} Pa")
        print(f"I = {I:.6e} m^4")
        print(f"Spans = {spans} m")
        print(f"Supports per node = {supports}")
        print(f"UDL q = {qkN:.3f} kN/m (aplicado en cada tramo por defecto)")
        print("Loads: UDL aplicado en cada tramo (configurable en código).")
        print("\n===== MATRIZ GLOBAL DE RIGIDEZ (K) =====")
        np.set_printoptions(precision=3, suppress=True)
        print(model.K)
        print("\n===== VECTOR DE CARGAS EQUIVALENTES (F) =====")
        print(model.F)
        print("\n===== RESULTADO DE DESPLAZAMIENTOS NODALES (U) =====")
        print(model.U)
        print("\n===== REACCIONES CALCULADAS (R = K U - F) =====")
        print(model.reactions)
        print("\n===== RESULTADO POR NUDO =====")
        node_res = model.get_node_results()
        for i, (v, th, sup, Rv, Rm) in enumerate(node_res):
            print(f"Nudo {i}: apoyo={sup}, v = {v:.6e} m, theta = {th:.6e} rad, R_vertical = {Rv/1e3:.3f} kN, R_momento = {Rm/1e3:.3f} kN·m")

        # compute global X, M, V arrays and maxima
        Xs, Ms, Vs = [], [], []
        offset = 0.0
        maxM = {'val':0.0,'x':0.0}
        maxV = {'val':0.0,'x':0.0}
        for e in range(model.n_spans):
            xs_e, Ms_e = model.M_funcs[e]
            _, Vs_e = model.V_funcs[e]
            X_local = xs_e + offset
            Xs.append(X_local)
            Ms.append(Ms_e)
            Vs.append(Vs_e)
            # check maxima
            idxM = np.argmax(np.abs(Ms_e))
            if abs(Ms_e[idxM]) > abs(maxM['val']):
                maxM['val'] = Ms_e[idxM]
                maxM['x'] = X_local[idxM]
            idxV = np.argmax(np.abs(Vs_e))
            if abs(Vs_e[idxV]) > abs(maxV['val']):
                maxV['val'] = Vs_e[idxV]
                maxV['x'] = X_local[idxV]
            offset += spans[e]
        X = np.concatenate(Xs)
        M = np.concatenate(Ms)
        V = np.concatenate(Vs)

        print("\n===== MÁXIMOS (valor y ubicación) =====")
        print(f"Max |M| = {abs(maxM['val']):.3f} N·m at x = {maxM['x']:.3f} m (={abs(maxM['val'])/1e3:.3f} kN·m)")
        print(f"Max |V| = {abs(maxV['val']):.3f} N at x = {maxV['x']:.3f} m (={abs(maxV['val'])/1e3:.3f} kN)")

        # 2D Plots (Matplotlib)
        fig, axs = plt.subplots(3,1, figsize=(10,10), constrained_layout=True)
        # Geometry & supports
        total_length = sum(spans)
        nodes_pos = np.concatenate([[0.0], np.cumsum(spans)])
        axs[0].hlines(0, 0, total_length, colors='k', linewidth=4)
        # draw supports markers
        for xi, sup in zip(nodes_pos, supports):
            if sup == 'fixed':
                axs[0].plot(xi, 0, marker='s', markersize=12, color='k')
                axs[0].text(xi, 0.06, f"{sup}", ha='center')
            elif sup == 'simple':
                axs[0].plot(xi, 0, marker='^', markersize=12, color='k')
                axs[0].text(xi, 0.06, f"{sup}", ha='center')
            else:
                axs[0].plot(xi, 0, marker='o', markersize=8, color='k')
                axs[0].text(xi, 0.06, f"{sup}", ha='center')
        axs[0].set_xlim(-0.5, total_length + 0.5)
        axs[0].set_ylim(-1.0, 0.6)
        axs[0].set_title("Geometría de la viga y apoyos (cargas mostradas aprox.)")
        # show UDL arrows (approx) - ensure X and Y same length
        step = 0.5
        x_ar = np.arange(0, total_length+1e-6, step)
        y_ar = np.full_like(x_ar, 0.08)
        axs[0].quiver(x_ar, y_ar, np.zeros_like(x_ar), -0.15*np.ones_like(x_ar),
                      angles='xy', scale_units='xy', scale=1, width=0.003, color='red')
        axs[0].text(0.1, 0.45, f"UDL q = {qkN:.2f} kN/m (aplicado uniformemente)", color='red')

        # Moment diagram
        axs[1].plot(X, M/1e3)
        axs[1].axhline(0, color='k', linewidth=0.6)
        axs[1].set_ylabel("M [kN·m]")
        axs[1].set_title("Diagrama de Momentos M(x)")
        axs[1].grid(True)

        # Shear diagram
        axs[2].plot(X, V/1e3)
        axs[2].axhline(0, color='k', linewidth=0.6)
        axs[2].set_ylabel("V [kN]")
        axs[2].set_xlabel("x [m]")
        axs[2].set_title("Diagrama de Fuerza Cortante V(x)")
        axs[2].grid(True)

        plt.show()

        # -----------------------
        # 3D Visualization con Plotly
        # -----------------------
        # Build deformed shape by sampling each element using shape functions
        n_sample_elem = 80
        X_def = []
        Z_def = []   # vertical deflection plotted in Z axis
        X_unde = []
        Z_unde = []
        offset = 0.0
        U_nodes = model.U  # nodal dofs [v1,th1,v2,th2...]
        for e in range(model.n_spans):
            L = model.spans[e]
            idx = [2*e,2*e+1,2*(e+1),2*(e+1)+1]
            u_e = U_nodes[idx]
            xs = np.linspace(0, L, n_sample_elem)
            for x in xs:
                xi = x / L
                N = hermite_N(xi, L)
                v_x = np.dot(N, u_e)  # deflection
                X_def.append(offset + x)
                Z_def.append(-v_x)    # negative so sagging goes down in Z
                X_unde.append(offset + x)
                Z_unde.append(0.0)
            offset += L

        # create 3D figure: x horizontal, y depth (0), z vertical deflection exaggerated
        exag = float(exag_w.value)
        Z_def_ex = np.array(Z_def) * exag

        fig3d = go.Figure()
        # undeformed beam
        fig3d.add_trace(go.Scatter3d(x=X_unde, y=[0]*len(X_unde), z=Z_unde,
                                     mode='lines', name='Original', line=dict(color='black', width=6)))
        # deformed beam
        fig3d.add_trace(go.Scatter3d(x=X_def, y=[0.2]*len(X_def), z=Z_def_ex,
                                     mode='lines+markers', name=f'Deformed x{exag:.1f}',
                                     line=dict(color='red', width=4), marker=dict(size=2)))
        # show supports as markers in 3D
        nodes_x = nodes_pos
        nodes_z = np.zeros_like(nodes_x)
        sup_colors = []
        for s in supports:
            if s == 'fixed':
                sup_colors.append('blue')
            elif s == 'simple':
                sup_colors.append('green')
            else:
                sup_colors.append('orange')
        fig3d.add_trace(go.Scatter3d(x=nodes_x, y=[0.0]*len(nodes_x), z=nodes_z,
                                     mode='markers+text', marker=dict(size=6, color=sup_colors),
                                     text=[f"n{i}" for i in range(len(nodes_x))], textposition='top center',
                                     name='Nudos'))

        fig3d.update_layout(scene=dict(
            xaxis_title='x [m]',
            yaxis_title='y [m]',
            zaxis_title=f'z (deflection x{exag:.1f}) [m]' ),
            width=900, height=500, title='Visualización 3D: viga original y deformada (exagerada)')
        fig3d.show()

# bind button
btn_calc.on_click(on_calculate)

# Fin del bloque
