import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay

# --- 1. Геометрія: відстані, кути, центри описаних кіл ---

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def triangle_min_angle(p1, p2, p3):
    a, b, c = dist(p2, p3), dist(p1, p3), dist(p1, p2)
    def angle(adj1, adj2, opp):
        if adj1 * adj2 == 0: return 0.0
        val = max(-1.0, min(1.0, (adj1**2 + adj2**2 - opp**2) / (2 * adj1 * adj2)))
        return math.acos(val)
    return min(angle(b, c, a), angle(a, c, b), angle(a, b, c))

def get_circumcenter(p1, p2, p3):
    """
    Знаходить центр описаного кола трикутника (класична точка Штайнера Рупперта).
    """
    d = 2 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
    if abs(d) < 1e-9:
        return (p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3
    
    ux = ((p1[0]**2 + p1[1]**2) * (p2[1] - p3[1]) + 
          (p2[0]**2 + p2[1]**2) * (p3[1] - p1[1]) + 
          (p3[0]**2 + p3[1]**2) * (p1[1] - p2[1])) / d
    uy = ((p1[0]**2 + p1[1]**2) * (p3[0] - p2[0]) + 
          (p2[0]**2 + p2[1]**2) * (p1[0] - p3[0]) + 
          (p3[0]**2 + p3[1]**2) * (p2[0] - p1[0])) / d
    return ux, uy

def point_in_poly(pt, verts):
    """Перевірка знаходження точки строго всередині многокутника (Ray casting)."""
    x, y = pt
    inside = False
    n = len(verts)
    p1x, p1y = verts[0]
    for i in range(n + 1):
        p2x, p2y = verts[i % n]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def check_convexity(verts):
    n = len(verts)
    signs = []
    for i in range(n):
        p0, p1, p2 = verts[i], verts[(i + 1) % n], verts[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if abs(cross) > 1e-7:
            signs.append(cross > 0)
    return len(set(signs)) <= 1

def parse_point(text, default):
    try:
        parts = text.replace(",", " ").split()
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except ValueError:
        pass
    return default

# --- 2. Автоматична генерація сітки за алгоритмом Рупперта ---

def generate_mesh_by_angle(verts, edge_types, target_angle_deg, max_iter=80):
    n_v = len(verts)
    pts = list(verts)
    boundaries = [edge_types[i] if i < len(edge_types) else 0 for i in range(n_v)]

    # Базова дискретизація контуру
    edge_steps = 3
    for s in range(n_v):
        p1 = verts[s]
        p2 = verts[(s + 1) % n_v]
        b_val = edge_types[s] if s < len(edge_types) else 0
        for step in range(1, edge_steps):
            t = step / edge_steps
            pts.append(((1 - t) * p1[0] + t * p2[0], (1 - t) * p1[1] + t * p2[1]))
            boundaries.append(b_val)

    # Оцінка характерного масштабу для захисту від злипання
    total_len = sum(dist(verts[i], verts[(i + 1) % n_v]) for i in range(n_v))
    min_dist_threshold = (total_len / (n_v * edge_steps)) * 0.25

    # Ітераційний рефайнінг: вставка центрів описаних кіл найгірших трикутників
    for _ in range(max_iter):
        tri = Delaunay(pts)
        worst_elem = None
        min_deg = 180.0

        for simplex in tri.simplices:
            p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            cx, cy = (p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3
            if point_in_poly((cx, cy), verts):
                deg = math.degrees(triangle_min_angle(p1, p2, p3))
                if deg < min_deg:
                    min_deg = deg
                    # Розрахунок центру описаного кола
                    cc = get_circumcenter(p1, p2, p3)
                    # Якщо центр описаного кола лежить у межах області — беремо його, інакше центроїд
                    worst_elem = cc if point_in_poly(cc, verts) else (cx, cy)

        # Критерій зупинки: якість досягнута
        if min_deg >= target_angle_deg or worst_elem is None:
            break

        # Захист від злипання: не додавати вузол, якщо поруч вже є інший
        too_close = any(dist(worst_elem, p) < min_dist_threshold for p in pts)
        if too_close:
            # Зменшуємо поріг захисту пропорційно або перериваємо ітерацію
            min_dist_threshold *= 0.8
            if min_dist_threshold < 1e-4:
                break
            continue

        pts.append(worst_elem)
        boundaries.append(0)  # Внутрішній вузол

    # Фінальна збірка скінченних елементів
    tri = Delaunay(pts)
    final_elements = []
    for simplex in tri.simplices:
        p1, p2, p3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        cx, cy = (p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3
        if point_in_poly((cx, cy), verts):
            final_elements.append(tuple(simplex))

    return pts, final_elements, boundaries

# --- 3. Інтерфейс Streamlit ---

st.set_page_config(page_title="Адаптивна тріангуляція Делоне", layout="wide")
st.title("Генерація сітки МСЕ за критерієм мінімального кута")

st.sidebar.header("Контроль якості")
# ЄДИНИЙ ПОВЗУНОК ДЛЯ РОЗБИТТЯ
target_angle = st.sidebar.slider(
    "Мінімальний кут (°):", 
    min_value=12, 
    max_value=32, 
    value=22, 
    step=1,
    help="Алгоритм автоматично збільшує кількість елементів, щоб усі кути задовольняли цей поріг."
)

st.sidebar.header("Геометрія області")
poly_choice = st.sidebar.selectbox("Контур фігури:", 
                                   ["Варіант 12 (4-кутник)", "Трикутник (3 кути)", "П'ятикутник (5 кутів)"])

if poly_choice == "Варіант 12 (4-кутник)":
    raw_verts = ["1.0, 0.0", "2.0, 0.0", "0.0, 2.0", "0.0, 1.0"]
    b_types = [2, 1, 2, 3] # 1-Діріхле, 2-Неймана, 3-Робіна
elif poly_choice == "Трикутник (3 кути)":
    raw_verts = ["0.0, 0.0", "2.0, 0.0", "1.0, 1.73"]
    b_types = [1, 2, 3]
else:
    raw_verts = ["0.0, 0.0", "2.0, 0.0", "2.5, 1.5", "1.0, 2.5", "-0.5, 1.5"]
    b_types = [1, 2, 2, 3, 2]

verts = [parse_point(st.sidebar.text_input(f"V{i+1}:", val), (0.0, 0.0)) for i, val in enumerate(raw_verts)]

if not check_convexity(verts):
    st.sidebar.warning("⚠️ Фігура неопукла! Перевірте порядок обходу вершин.")
else:
    st.sidebar.success("✓ Фігура опукла.")

# Розрахунок сітки
nodes, elements, boundaries = generate_mesh_by_angle(verts, b_types, target_angle)

# Аналіз реальних кутів
element_angles = [math.degrees(triangle_min_angle(nodes[e[0]], nodes[e[1]], nodes[e[2]])) for e in elements]
actual_min_angle = min(element_angles) if element_angles else 0.0
bad_elements = [i for i, a in enumerate(element_angles) if a < target_angle]

# --- 4. Візуалізація та таблиці ---

col1, col2 = st.columns([1.1, 0.9])

with col1:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    
    for idx, e in enumerate(elements):
        pts = [nodes[n] for n in e]
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        
        if idx in bad_elements:
            ax.fill(xs, ys, color='red', alpha=0.35)
        ax.plot(xs, ys, color='black', linewidth=0.8)
        
        cx, cy = sum(p[0] for p in pts)/3, sum(p[1] for p in pts)/3
        ax.text(cx, cy, f"E{idx}", color='blue', fontsize=7, ha='center', va='center')

    colors = {0: 'gray', 1: 'red', 2: 'green', 3: 'purple'}
    for idx, (node, b) in enumerate(zip(nodes, boundaries)):
        ax.plot(node[0], node[1], marker='o', markersize=5, color=colors.get(b, 'black'))
        ax.text(node[0] + 0.02, node[1] + 0.02, str(idx), fontsize=8)

    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"Фактичний мін. кут: {actual_min_angle:.1f}° (Ціль: {target_angle}°)")
    st.pyplot(fig)

with col2:
    st.metric("Досягнутий мінімальний кут", f"{actual_min_angle:.2f}°", delta=f"{actual_min_angle - target_angle:.2f}°")
    st.write(f"**Вузлів:** {len(nodes)} | **Скінченних елементів:** {len(elements)}")
    
    if actual_min_angle >= target_angle:
        st.success(f"Цільовий кут >={target_angle}° повністю досягнуто!")
    else:
        st.info(f"Досягнуто геометричної межі для поточної форми: {actual_min_angle:.1f}°")

    st.write("**Таблиця зв'язності елементів (Topology)**")
    df_elements = pd.DataFrame({
        "Елемент": [f"E{i}" for i in range(len(elements))],
        "N1": [e[0] for e in elements],
        "N2": [e[1] for e in elements],
        "N3": [e[2] for e in elements],
        "Мін. кут (°)": [round(a, 1) for a in element_angles]
    })
    st.dataframe(df_elements, height=180, use_container_width=True)

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
