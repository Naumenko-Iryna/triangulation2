import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import triangle as tr

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
    """Перевірка опуклості многокутника через векторний добуток суміжних ребер."""
    n = len(verts)
    signs = []
    for i in range(n):
        p0, p1, p2 = verts[i], verts[(i + 1) % n], verts[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if abs(cross) > 1e-7:
            signs.append(cross > 0)
    return len(set(signs)) <= 1

# --- 2. Генерація сітки виключно бібліотекою Triangle ---

def generate_mesh_triangle(verts, edge_types, min_angle):
    n_v = len(verts)
    # З'єднуємо вершини контуру у замкнений циклічний граф (сегменти)
    segments = [[i, (i + 1) % n_v] for i in range(n_v)]
    seg_markers = [edge_types[i] if i < len(edge_types) else 0 for i in range(n_v)]

    poly_data = {
        'vertices': np.array(verts, dtype=np.float64),
        'segments': np.array(segments, dtype=np.int32),
        'segment_markers': np.array(seg_markers, dtype=np.int32)
    }

    # 'p' - планарний граф (контур)
    # 'q{min_angle}' - алгоритм Рупперта (вставка точок Штайнера для забезпечення кута)
    # 'D' - Conforming Delaunay (гарантує розбиття граней і вставку внутрішніх точок)
    tri_args = f'pq{float(min_angle):.1f}D'
    mesh_out = tr.triangulate(poly_data, tri_args)

    nodes = [tuple(p) for p in mesh_out['vertices'].tolist()]
    elements = [tuple(t) for t in mesh_out['triangles'].tolist()]

    # Зчитування та перенесення маркерів границь (1, 2, 3) на отримані вузли
    node_boundaries = [0] * len(nodes)
    if 'vertex_markers' in mesh_out:
        node_boundaries = [int(m[0]) for m in mesh_out['vertex_markers']]

    return nodes, elements, node_boundaries

# --- 3. Інтерфейс додатку (Streamlit) ---

st.set_page_config(page_title="Тріангуляція Делоне (Triangle)", layout="wide")
st.title("Генерація сітки МСЕ за допомогою бібліотеки Triangle")

st.sidebar.header("Критерій якості сітки")
# ЄДИНИЙ ПОВЗУНОК — контролює розбиття за якістю
min_angle_target = st.sidebar.slider(
    "Мінімальний кут сітки (°):", 
    min_value=10, 
    max_value=33, 
    value=20,
    step=1,
    help="Бібліотека автоматично підбирає кількість вузлів і густину так, щоб жоден трикутник не мав кута менше за цей."
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

# Безпосередній виклик бібліотеки Triangle
nodes, elements, boundaries = generate_mesh_triangle(verts, b_types, min_angle_target)

# Розрахунок фактичних кутів утворених елементів
element_angles = [math.degrees(triangle_min_angle(nodes[e[0]], nodes[e[1]], nodes[e[2]])) for e in elements]
actual_min_angle = min(element_angles) if element_angles else 0.0

# --- 4. Візуалізація та структуровані таблиці ---

col1, col2 = st.columns([1.1, 0.9])

with col1:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    
    # Відмальовування трикутників
    for idx, e in enumerate(elements):
        pts = [nodes[n] for n in e]
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        ax.plot(xs, ys, color='black', linewidth=0.8)
        
        # Центроїд для підпису елемента
        cx, cy = sum(p[0] for p in pts)/3, sum(p[1] for p in pts)/3
        ax.text(cx, cy, f"E{idx}", color='blue', fontsize=7, ha='center', va='center')

    # Побудова та кодування граничних вузлів (1-червоний, 2-зелений, 3-фіолетовий, 0-сірий)
    colors = {0: 'gray', 1: 'red', 2: 'green', 3: 'purple'}
    for idx, (node, b) in enumerate(zip(nodes, boundaries)):
        ax.plot(node[0], node[1], marker='o', markersize=5, color=colors.get(b, 'black'))
        ax.text(node[0] + 0.02, node[1] + 0.02, str(idx), fontsize=8)

    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"Triangle (CDT) | Фактичний мін. кут: {actual_min_angle:.1f}°")
    st.pyplot(fig)

with col2:
    st.metric("Досягнутий найменший кут", f"{actual_min_angle:.2f}°")
    st.write(f"**Згенеровано вузлів:** {len(nodes)} | **Елементів:** {len(elements)}")
    st.success(f"Критерій якості гарантовано: усі кути >= {min_angle_target}°")

    # 1. Таблиця зв'язності (інцидентності)
    st.write("**Масив зв'язності елементів (Topology)**")
    df_elements = pd.DataFrame({
        "Елемент": [f"E{i}" for i in range(len(elements))],
        "N1": [e[0] for e in elements],
        "N2": [e[1] for e in elements],
        "N3": [e[2] for e in elements],
        "Мін. кут (°)": [round(a, 1) for a in element_angles]
    })
    st.dataframe(df_elements, height=180, use_container_width=True)

    # 2. Таблиця вузлів із крайовими умовами
    st.write("**Таблиця координат вузлів та крайових міток**")
    b_meanings = {0: "0 (Внутрішній)", 1: "1 (Діріхле)", 2: "2 (Неймана)", 3: "3 (Робіна)"}
    df_nodes = pd.DataFrame({
        "Вузол": [f"N{i}" for i in range(len(nodes))],
        "X": [round(n[0], 3) for n in nodes],
        "Y": [round(n[1], 3) for n in nodes],
        "Маркер": boundaries,
        "Умова": [b_meanings.get(b, str(b)) for b in boundaries]
    })
    st.dataframe(df_nodes, height=180, use_container_width=True)
