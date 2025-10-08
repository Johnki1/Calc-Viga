fc_default = 28
fy_default = 280
Es_default = 200000
Ec_default = 5000 * np.sqrt(fc_default)

d_default = 0.45
b_default = 0.50
tw_default = 0.30
tf_default = 0.08
y_default = 0.24

A_default = 1.51e5     # mm²
As_default = 1.35e5    # mm²
I_default = 2.776e9    # mm⁴

# Longitudes y carga
L1_default = 7.0
L2_default = 8.0
L3_default = 2.0
q_default = 10.0  # kN/m

# --- Crear widgets ---
fc = widgets.FloatText(value=fc_default, description='f\'c (MPa):')
fy = widgets.FloatText(value=fy_default, description='fy (MPa):')
Es = widgets.FloatText(value=Es_default, description='Es (MPa):')
Ec = widgets.FloatText(value=Ec_default, description='Ec (MPa):')

d = widgets.FloatText(value=d_default, description='d (m):')
b = widgets.FloatText(value=b_default, description='b (m):')
tw = widgets.FloatText(value=tw_default, description='tw (m):')
tf = widgets.FloatText(value=tf_default, description='tf (m):')

L1 = widgets.FloatText(value=L1_default, description='L1 (m):')
L2 = widgets.FloatText(value=L2_default, description='L2 (m):')
L3 = widgets.FloatText(value=L3_default, description='L3 (m):')
q = widgets.FloatText(value=q_default, description='q (kN/m):')

boton = widgets.Button(description="Calcular Viga", button_style='success')

# Mostrar interfaz
ui = widgets.VBox([
    widgets.HTML("<h3>Datos de la Viga T</h3>"),
    widgets.HBox([fc, fy, Es, Ec]),
    widgets.HBox([d, b, tw, tf]),
    widgets.HTML("<h3>Geometría y Carga</h3>"),
    widgets.HBox([L1, L2, L3, q]),
    boton
])
display(ui)

def calcular_viga(event):
    # Leer datos
    L1_val, L2_val, L3_val = L1.value, L2.value, L3.value
    q_val = q.value
    L_total = L1_val + L2_val + L3_val

    # Definimos apoyos (A, B, C)
    xA, xB, xC = 0, L1_val, L1_val + L2_val
    x_final = L_total

    # --- Cálculo simple de reacciones (viga continua 3 apoyos, carga uniforme) ---
    # Método de Cross simplificado o fórmula aproximada para 3 vanos:
    # R_A + R_B + R_C = q*L_total
    # Suponemos continuidad con momentos iguales en los extremos internos
    # Para hacerlo didáctico, aplicaremos un modelo aproximado:

    # Convertimos a N y m
    qN = q_val * 1000  # N/m

    # Matriz de rigidez (simplificada)
    # Para facilidad usaremos proporciones:
    RA = 0.26 * qN * L1_val
    RB = 0.48 * qN * L2_val
    RC = 0.26 * qN * L3_val

    # --- Momentos en apoyos aproximados (viga continua) ---
    MA = 0
    MB = -qN * L1_val**2 / 12
    MC = -qN * L2_val**2 / 12

    # --- Gráficas ---
    x = np.linspace(0, L_total, 400)
    M = np.piecewise(x, 
        [x <= L1_val, (x > L1_val) & (x <= L1_val + L2_val), x > L1_val + L2_val],
        [
            lambda x: -qN/2 * x*(L1_val - x),  # tramo 1
            lambda x: -qN/2 * (x - L1_val)*(L2_val - (x - L1_val)),  # tramo 2
            lambda x: -qN/2 * (x - (L1_val + L2_val))*(L3_val - (x - (L1_val + L2_val)))  # tramo 3
        ])

    V = np.gradient(M, x)

    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].set_title("Viga y Cargas")
    axs[0].plot([0, L_total], [0, 0], 'k', lw=4)
    axs[0].arrow(L1_val/2, 0.1, 0, -0.1, head_width=0.2, color='r')
    axs[0].arrow(L1_val + L2_val/2, 0.1, 0, -0.1, head_width=0.2, color='r')
    axs[0].text(L1_val/2, 0.15, f'q={q_val} kN/m', color='r')

    axs[1].plot(x, M/1e6)
    axs[1].set_ylabel("Momento [kN·m]")
    axs[1].grid()

    axs[2].plot(x, V/1000)
    axs[2].set_ylabel("Cortante [kN]")
    axs[2].set_xlabel("Longitud [m]")
    axs[2].grid()

    plt.tight_layout()
    plt.show()

    print(f"Reacciones aproximadas:\nRA = {RA/1000:.2f} kN\nRB = {RB/1000:.2f} kN\nRC = {RC/1000:.2f} kN")
    print(f"Momentos: MA = {MA:.2f} N·m, MB = {MB:.2f} N·m, MC = {MC:.2f} N·m")

boton.on_click(calcular_viga)
