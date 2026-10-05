import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay

# --- 1. Геометричні розрахунки та перевірка опуклості ---

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def triangle_min_angle(p1, p2, p3):
    a, b, c = dist(p2, p3), dist(p1, p3), dist(p1, p2)
    def angle(adj1, adj2, opp):
        if adj1 * adj2 == 0: return 0
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
    """Перевірка опуклості через знакосталість векторного добутку."""
    n = len(verts)
    signs = []
    for i in range(n):
        p0, p1, p2 = verts[i], verts[(i + 1) % n], verts[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if abs(cross) > 1e-7:
            signs.append(cross > 0)
    return len(set(signs)) <= 1

def point_in_poly(pt, verts):
    """Перевірка, чи лежить точка всередині многокутника (Ray casting)."""
    x, y = pt
    inside = False
    n = len(verts)
    p1x, p1y = verts[0]
    for i in range(n + 1):
        p2x, p2y = verts[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

# --- 2. Генерація сітки Делоне з адаптивним подрібненням ---

def generate_mesh_delaunay(verts, edge_types, min_angle_target, max_iter=25):
    """
    Побудова сітки Делоне за алгоритмом Рупперта (Delaunay Refinement):
    Вставляє точки Штайнера в центроїди дефектних трикутників, 
    доки мінімальний кут не перевищить цільовий поріг.
    """
    n_v = len(verts)
    # Початковий набір точок — вершини контуру
    pts = list(verts)
    node_boundaries = [edge_types[i] if i < len(edge_types) else 0 for i in range(n_v)]

    # Додаємо вузли розбиття вздовж зовнішніх ребер для початкової дискретизації
    edge_steps = 3
    for s in range(n_v):
        p_start = verts[s]
        p_end = verts[(s + 1) % n_v]
        b_val = edge_types[s] if s < len(edge_types) else 0
        for step in range(1, edge_steps):
            t = step / edge_steps
            new_pt = ((1 - t) * p_start[0] + t * p_end[0], (1 - t) * p_start[1] + t * p_end[1])
            pts.append(new_pt)
            node_boundaries.append(b_val)

    # Адаптивний цикл вставки точок Штайнера
    for _ in range(max_iter):
        tri = Delaunay(pts)
        elements = []
        worst_elem = None
        min_deg_found = 180.0

        for simplex in tri.simplices:
            p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            cx, cy = (p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3
            
            # Відсікаємо трикутники поза межами контуру
            if point_in_poly((cx, cy), verts):
                elements.append(tuple(simplex))
                deg = math.degrees(triangle_min_angle(p1, p2, p3))
                if deg < min_deg_found:
                    min_deg_found = deg
                    worst_elem = (cx, cy)

        # Якщо всі трикутники досягли бажаного кута — зупиняємось
        if min_deg_found >= min_angle_target or worst_elem is None:
            break

        # Вставка нової внутрішньої точки (маркер 0)
        pts.append(worst_elem)
        node_boundaries.append(0)

    # Фінальна тріангуляція
    tri = Delaunay(pts)
    final_elements = []
    for simplex in tri.simplices:
        p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        cx, cy = (p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3
        if point_in_poly((cx, cy), verts):
            final_elements.append(tuple(simplex))

    return pts, final_elements, node_boundaries

# --- 3. Інтерфейс додатка (Streamlit) ---

st.set_page_config(page_title="Тріангуляція Делоне (МСЕ)", layout="wide")
st.title("Адаптивна тріангуляція Делоне за критерієм якості")

st.sidebar.header("Критерій якості сітки")
min_angle_target = st.sidebar.slider(
    "Мінімальний кут сітки (°):", 
    min_value=10, 
    max_value=32, 
    value=20,
    step=1,
    help="Алгоритм автоматично вставляє точки Штайнера, доки всі елементи не задовольнять цей кут."
)

st.sidebar.header("Геометрія області")
poly_choice = st.sidebar.selectbox("Контур фігури:", 
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
    st.sidebar.warning("⚠️ Фігура неопукла! Перевірте порядок вершин.")
else:
    st.sidebar.success("✓ Фігура опукла.")

# Генерація адаптивної сітки Делоне
nodes, elements, boundaries = generate_mesh_delaunay(verts, b_types, min_angle_target)

# Розрахунок реальних кутів
element_angles = [math.degrees(triangle_min_angle(nodes[e[0]], nodes[e[1]], nodes[e[2]])) for e in elements]
actual_min_angle = min(element_angles) if element_angles else 0.0

# --- 4. Візуалізація та таблиці ---

col1, col2 = st.columns([1.1, 0.9])

with col1:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    
    for idx, e in enumerate(elements):
        pts = [nodes[n] for n in e]
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        ax.plot(xs, ys, color='black', linewidth=0.8)
        
        cx, cy = sum(p[0] for p in pts)/3, sum(p[1] for p in pts)/3
        ax.text(cx, cy, f"E{idx}", color='blue', fontsize=7, ha='center', va='center')

    # Побудова граничних вузлів (1-червоний, 2-зелений, 3-фіолетовий, 0-сірий)
    colors = {0: 'gray', 1: 'red', 2: 'green', 3: 'purple'}
    for idx, (node, b) in enumerate(zip(nodes, boundaries)):
        ax.plot(node[0], node[1], marker='o', markersize=5, color=colors.get(b, 'black'))
        ax.text(node[0] + 0.02, node[1] + 0.02, str(idx), fontsize=8)

    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"Delaunay Refinement | Фактичний мін. кут: {actual_min_angle:.1f}°")
    st.pyplot(fig)

with col2:
    st.metric("Досягнутий найменший кут", f"{actual_min_angle:.2f}°")
    st.write(f"**Згенеровано вузлів:** {len(nodes)} | **Скінченних елементів:** {len(elements)}")
    
    if actual_min_angle >= min_angle_target:
        st.success(f"Критерій якості виконано: усі кути >= {min_angle_target}°")
    else:
        st.info(f"Досягнуто геометричної межі поділу: {actual_min_angle:.1f}°")

    # 1. Таблиця зв'язності (Topology)
    st.write("**Масив зв'язності елементів (Topology)**")
    df_elements = pd.DataFrame({
        "Елемент": [f"E{i}" for i in range(len(elements))],
        "N1": [e[0] for e in elements],
        "N2": [e[1] for e in elements],
        "N3": [e[2] for e in elements],
        "Мін. кут (°)": [round(a, 1) for a in element_angles]
    })
    st.dataframe(df_elements, height=180, use_container_width=True)

    # 2. Таблиця вузлів із крайовими умовами для МСЕ
    st.write("**Таблиця вузлів та крайових міток**")
    b_meanings = {0: "0 (Внутрішній)", 1: "1 (Діріхле)", 2: "2 (Неймана)", 3: "3 (Робіна)"}
    df_nodes = pd.DataFrame({
        "Вузол": [f"N{i}" for i in range(len(nodes))],
        "X": [round(n[0], 3) for n in nodes],
        "Y": [round(n[1], 3) for n in nodes],
        "Маркер": boundaries,
        "Умова": [b_meanings.get(b, str(b)) for b in boundaries]
    })
    st.dataframe(df_nodes, height=180, use_container_width=True)
