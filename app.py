import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.spatial import Delaunay
from matplotlib.path import Path

# --- 1. Геометричні розрахунки та перевірка опуклості ---

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_triangle_properties(pts):
    """
    Обчислює характеристики елемента: мінімальний кут (в градусах), 
    площу та координати центру описаного кола (cc).
    """
    A, B, C = pts[0], pts[1], pts[2]
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)

    # Площа за Героном
    s = (a + b + c) / 2.0
    area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))

    # Теорема косинусів
    angles = []
    for x, y, z in [(a, b, c), (b, a, c), (c, a, b)]:
        if y * z == 0:
            angles.append(0.0)
            continue
        val = np.clip((y**2 + z**2 - x**2) / (2 * y * z), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(val)))

    min_ang = min(angles)

    # Координати центру описаного кола
    D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    if abs(D) < 1e-9:
        return min_ang, area, None

    Ux = ((A[0]**2 + A[1]**2)*(B[1] - C[1]) + (B[0]**2 + B[1]**2)*(C[1] - A[1]) + (C[0]**2 + C[1]**2)*(A[1] - B[1])) / D
    Uy = ((A[0]**2 + A[1]**2)*(C[0] - B[0]) + (B[0]**2 + B[1]**2)*(A[0] - C[0]) + (C[0]**2 + C[1]**2)*(B[0] - A[0])) / D

    return min_ang, area, np.array([Ux, Uy])

def parse_point(text, default):
    try:
        parts = text.replace(",", " ").split()
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except ValueError:
        pass
    return default

def check_convexity(verts):
    n = len(verts)
    signs = []
    for i in range(n):
        p0, p1, p2 = verts[i], verts[(i + 1) % n], verts[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if abs(cross) > 1e-7:
            signs.append(cross > 0)
    return len(set(signs)) <= 1

def get_node_marker(pt, poly_pts, edge_types, tol=1e-4):
    """
    Визначає граничний маркер вузла (1, 2, 3) за належністю до ребра,
    або 0 для внутрішнього вузла.
    """
    n_v = len(poly_pts)
    p = np.array(pt)
    for i in range(n_v):
        p1 = np.array(poly_pts[i])
        p2 = np.array(poly_pts[(i + 1) % n_v])
        
        d1 = np.linalg.norm(p - p1)
        d2 = np.linalg.norm(p2 - p)
        d12 = np.linalg.norm(p2 - p1)
        
        if abs(d1 + d2 - d12) < tol:
            return edge_types[i] if i < len(edge_types) else 0
    return 0

# --- 2. Генератор сітки Рупперта (Delaunay Refinement) ---

def generate_mesh_by_angle(verts, edge_types, target_min_angle, max_iter=200):
    poly = np.array(verts, dtype=np.float64)
    n_v = len(poly)
    path = Path(poly)

    # 1. Початкове адаптивне розбиття контуру
    total_perim = sum(np.linalg.norm(poly[(i+1)%n_v] - poly[i]) for i in range(n_v))
    target_seg_len = total_perim / (n_v * 3.5)

    points = []
    for i in range(n_v):
        p1 = poly[i]
        p2 = poly[(i + 1) % n_v]
        edge_len = np.linalg.norm(p2 - p1)
        n_segs = max(1, int(np.ceil(edge_len / target_seg_len)))
        for j in range(n_segs):
            points.append(p1 + (p2 - p1) * (j / n_segs))

    points = np.array(points, dtype=np.float64)

    # 2. Ітераційний процес покращення сітки
    for _ in range(max_iter):
        tri = Delaunay(points)
        simplices = tri.simplices

        # Фільтрація трикутників за межами контуру
        centroids = np.mean(points[simplices], axis=1)
        inside_mask = path.contains_points(centroids, radius=-1e-5)
        simplices = simplices[inside_mask]

        bad_triangles = []
        for s in simplices:
            min_ang, area, cc = calculate_triangle_properties(points[s])
            if min_ang < target_min_angle and cc is not None:
                score = target_min_angle - min_ang
                centroid = np.mean(points[s], axis=0)
                bad_triangles.append((score, cc, centroid))

        # Якщо всі кути задовольняють умову — розбиття завершено
        if not bad_triangles:
            break

        # Сортування: виправляємо спершу найбільш гострі трикутники
        bad_triangles.sort(key=lambda x: x[0], reverse=True)
        added = False

        for _, cc, centroid in bad_triangles:
            # Fallback: якщо центр описаного кола вилітає за контур, беремо центроїд
            candidate = cc if path.contains_point(cc) else centroid
            
            # Захист від злипання точок (запобігає утворенню чорних плям)
            if np.min(np.linalg.norm(points - candidate, axis=1)) > 1e-3:
                points = np.vstack([points, candidate])
                added = True
                break

        if not added:
            break

    # Фінальна тріангуляція
    tri = Delaunay(points)
    simplices = tri.simplices
    centroids = np.mean(points[simplices], axis=1)
    inside_mask = path.contains_points(centroids, radius=-1e-5)
    final_elements = [tuple(s) for s in simplices[inside_mask]]

    # Маркування границь для кожного отриманого вузла
    nodes = [tuple(p) for p in points]
    boundaries = [get_node_marker(p, verts, edge_types) for p in nodes]

    return nodes, final_elements, boundaries

# --- 3. Інтерфейс Streamlit ---

st.set_page_config(page_title="Триангуляція Делоне (МСЕ)", layout="wide")
st.title("Триангуляція Делоне за критерієм мінімального кута")

st.sidebar.header("Контроль якості")
target_angle = st.sidebar.slider(
    "Мінімальний кут сітки (°):", 
    min_value=10.0, 
    max_value=33.0, 
    value=20.0, 
    step=1.0,
    help="Алгоритм динамічно вставляє вузли, доки всі кути не перевищать цей поріг."
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
    st.sidebar.warning("⚠️ Фігура неопукла! Перевірте порядок вершин.")
else:
    st.sidebar.success("✓ Фігура опукла.")

# Розрахунок сітки
nodes, elements, boundaries = generate_mesh_by_angle(verts, b_types, target_angle)

# Розрахунок реальних кутів утворених елементів
element_angles = []
bad_elements = []
for idx, e in enumerate(elements):
    p1, p2, p3 = nodes[e[0]], nodes[e[1]], nodes[e[2]]
    deg, _, _ = calculate_triangle_properties(np.array([p1, p2, p3]))
    element_angles.append(deg)
    if deg < target_angle:
        bad_elements.append(idx)

actual_min_angle = min(element_angles) if element_angles else 0.0

# --- 4. Візуалізація та структуровані таблиці ---

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

    # Вузли: 1-червоний, 2-зелений, 3-фіолетовий, 0-сірий
    colors = {0: 'gray', 1: 'red', 2: 'green', 3: 'purple'}
    for idx, (node, b) in enumerate(zip(nodes, boundaries)):
        ax.plot(node[0], node[1], marker='o', markersize=5, color=colors.get(b, 'black'))
        ax.text(node[0] + 0.02, node[1] + 0.02, str(idx), fontsize=8)

    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"Фактичний мін. кут: {actual_min_angle:.1f}° (Ціль: {target_angle:.0f}°)")
    st.pyplot(fig)

with col2:
    st.metric("Досягнутий мінімальний кут", f"{actual_min_angle:.2f}°", delta=f"{actual_min_angle - target_angle:.2f}°")
    st.write(f"**Вузлів:** {len(nodes)} | **Скінченних елементів:** {len(elements)}")

    if actual_min_angle >= target_angle:
        st.success(f"Цільовий критерій якості (>={target_angle:.0f}°) виконано!")
    else:
        st.info(f"Досягнуто геометричної межі для форми: {actual_min_angle:.1f}°")

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

    # 2. Таблиця координат і крайових умов
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
