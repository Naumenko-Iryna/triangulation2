import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Геометричні допоміжні функції ---

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1]) # Евклідова відстань між вузлами

def triangle_min_angle(p1, p2, p3):
    a, b, c = dist(p2, p3), dist(p1, p3), dist(p1, p2)
    def angle(adj1, adj2, opp):
        if adj1 * adj2 == 0: return 0
        # Теорема косинусів із захистом від похибок комп'ютерного округлення
        val = max(-1.0, min(1.0, (adj1**2 + adj2**2 - opp**2) / (2 * adj1 * adj2)))
        return math.acos(val)
    return min(angle(b, c, a), angle(a, c, b), angle(a, b, c))

def parse_point(text, default):
    try:
        parts = text.replace(",", " ").split()
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except ValueError:
        pass
    return default

def check_convexity(verts):
    # Перевірка опуклості через знакосталість векторного добутку суміжних сторін
    n = len(verts)
    signs = []
    for i in range(n):
        p0, p1, p2 = verts[i], verts[(i + 1) % n], verts[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if abs(cross) > 1e-7:
            signs.append(cross > 0)
    return len(set(signs)) <= 1

# --- 2. Універсальний генератор сітки для довільного многокутника ---

def generate_mesh(nx, ny, verts, edge_types):
    nodes, node_boundaries, elements = [], [], []
    num_edges = len(verts)

    # 1. Знаходження центроїда фігури
    center = (sum(v[0] for v in verts) / num_edges, sum(v[1] for v in verts) / num_edges)

    # 2. Побудова унікальних вузлів (шарами від контуру до центру)
    node_map = {}
    for s in range(num_edges):
        p_start = verts[s]
        p_end = verts[(s + 1) % num_edges]
        b_val = edge_types[s] if s < len(edge_types) else 0

        for j in range(ny + 1):
            t_rad = j / ny  # 0 на контурі, 1 в центрі
            for i in range(nx + 1):
                t_edge = i / nx
                # Подвійна лінійна інтерполяція (ребро многокутника -> центр)
                bx = (1 - t_edge) * p_start[0] + t_edge * p_end[0]
                by = (1 - t_edge) * p_start[1] + t_edge * p_end[1]
                rx = (1 - t_rad) * bx + t_rad * center[0]
                ry = (1 - t_rad) * by + t_rad * center[1]

                key = (round(rx, 5), round(ry, 5))
                if key not in node_map: # дедуплікація вузлів на межах секторів
                    node_map[key] = len(nodes)
                    nodes.append((rx, ry))
                    # Маркування крайовими умовами: лише зовнішній шар (j == 0)
                    node_boundaries.append(b_val if j == 0 else 0)

    # 3. Формування трикутників із локальним критерієм Делоне
    for s in range(num_edges):
        p_start = verts[s]
        p_end = verts[(s + 1) % num_edges]
        for j in range(ny):
            for i in range(nx):
                def get_id(edge_idx, rad_idx):
                    t_r = rad_idx / ny
                    t_e = edge_idx / nx
                    bx = (1 - t_e) * p_start[0] + t_e * p_end[0]
                    by = (1 - t_e) * p_start[1] + t_e * p_end[1]
                    return node_map[(round((1 - t_r) * bx + t_r * center[0], 5), 
                                     round((1 - t_r) * by + t_r * center[1], 5))]

                bl, br = get_id(i, j), get_id(i + 1, j)
                tl, tr = get_id(i, j + 1), get_id(i + 1, j + 1)

                if tl == tr:
                    # Вершина біля центру: вироджена комірка стає одним трикутником
                    if len({bl, br, tl}) == 3:
                        elements.append((bl, br, tl))
                else:
                    # Критерій Делоне: максимізація мінімального кута між двома діагоналями
                    a1 = min(triangle_min_angle(nodes[bl], nodes[br], nodes[tr]),
                             triangle_min_angle(nodes[bl], nodes[tr], nodes[tl]))
                    a2 = min(triangle_min_angle(nodes[bl], nodes[br], nodes[tl]),
                             triangle_min_angle(nodes[br], nodes[tr], nodes[tl]))
                    
                    if a1 >= a2:
                        elements.extend([(bl, br, tr), (bl, tr, tl)])
                    else:
                        elements.extend([(bl, br, tl), (br, tr, tl)])
                        
    return nodes, elements, node_boundaries

# --- 3. Інтерфейс додатку (Streamlit) ---

st.set_page_config(page_title="Триангуляція Делоне (МСЕ)", layout="wide")
st.title("Триангуляція Делоне для довільних областей")

# Бічна панель
st.sidebar.header("Параметри дискретизації")
nx = st.sidebar.number_input("Густина по контуру (Nx)", min_value=1, max_value=20, value=2)
ny = st.sidebar.number_input("Густина до центру (Ny)", min_value=1, max_value=20, value=2)
angle_threshold = st.sidebar.slider("Критичний кут (°)", min_value=5, max_value=40, value=20)

st.sidebar.header("Контур фігури")
poly_choice = st.sidebar.selectbox("Шаблон фігури:", 
                                   ["Варіант 12 (4-кутник)", "Трикутник (3 кути)", "П'ятикутник (5 кутів)"])

if poly_choice == "Варіант 12 (4-кутник)":
    raw_verts = ["1.0, 0.0", "2.0, 0.0", "0.0, 2.0", "0.0, 1.0"]
    b_types = [2, 1, 2, 3] # Границі: 1-Діріхле, 2-Неймана, 3-Робіна
elif poly_choice == "Трикутник (3 кути)":
    raw_verts = ["0.0, 0.0", "2.0, 0.0", "1.0, 1.73"]
    b_types = [1, 2, 3]
else:
    raw_verts = ["0.0, 0.0", "2.0, 0.0", "2.5, 1.5", "1.0, 2.5", "-0.5, 1.5"]
    b_types = [1, 2, 2, 3, 2]

verts = [parse_point(st.sidebar.text_input(f"V{i+1}:", val), (0.0, 0.0)) for i, val in enumerate(raw_verts)]

# Перевірка опуклості
if not check_convexity(verts):
    st.sidebar.warning("⚠️ Фігура неопукла! Можливі спотворення сітки.")
else:
    st.sidebar.success("✓ Фігура опукла.")

# Розрахунок сітки
nodes, elements, boundaries = generate_mesh(nx, ny, verts, b_types)

# Аудит кутів
element_angles = [math.degrees(triangle_min_angle(nodes[e[0]], nodes[e[1]], nodes[e[2]])) for e in elements]
bad_elements = [i for i, a in enumerate(element_angles) if a < angle_threshold]
min_angle = min(element_angles) if element_angles else 0.0

# --- 4. Візуалізація та структурований вивід ---

col1, col2 = st.columns([1.1, 0.9])

with col1:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    
    # Побудова скінченних елементів
    for idx, e in enumerate(elements):
        pts = [nodes[n] for n in e]
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        
        # Підсвічування трикутників нижче порогу
        if idx in bad_elements:
            ax.fill(xs, ys, color='red', alpha=0.3)
        ax.plot(xs, ys, color='black', linewidth=0.8)
        
        # Центроїд трикутника для підпису номера
        cx, cy = sum(p[0] for p in pts)/3, sum(p[1] for p in pts)/3
        ax.text(cx, cy, f"E{idx}", color='blue', fontsize=7, ha='center', va='center')

    # Побудова вузлів з кольоровим кодуванням границь
    colors = {0: 'gray', 1: 'red', 2: 'green', 3: 'purple'}
    for idx, (node, b) in enumerate(zip(nodes, boundaries)):
        ax.plot(node[0], node[1], marker='o', markersize=5, color=colors.get(b, 'black'))
        ax.text(node[0] + 0.02, node[1] + 0.02, str(idx), fontsize=8)

    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"Елементів: {len(elements)} | Мін. кут: {min_angle:.1f}°")
    st.pyplot(fig)

with col2:
    st.metric("Найменший кут сітки", f"{min_angle:.2f}°")
    if bad_elements:
        st.warning(f"Дефектних елементів (< {angle_threshold}°): {len(bad_elements)} шт.")
    else:
        st.success(f"Усі елементи відповідають порогу >= {angle_threshold}°")

    # Масив зв'язності у вигляді таблиці
    st.write("**Таблиця зв'язності (Topology)**")
    df_elements = pd.DataFrame({
        "Елемент": [f"E{i}" for i in range(len(elements))],
        "N1": [e[0] for e in elements],
        "N2": [e[1] for e in elements],
        "N3": [e[2] for e in elements],
        "Мін. кут (°)": [round(a, 1) for a in element_angles]
    })
    st.dataframe(df_elements, height=180, use_container_width=True)

    # Список вузлів у вигляді таблиці
    st.write("**Таблиця вузлів (Nodes & Boundary Markers)**")
    df_nodes = pd.DataFrame({
        "Вузол": [f"N{i}" for i in range(len(nodes))],
        "X": [round(n[0], 3) for n in nodes],
        "Y": [round(n[1], 3) for n in nodes],
        "Маркер": boundaries
    })
    st.dataframe(df_nodes, height=180, use_container_width=True)
