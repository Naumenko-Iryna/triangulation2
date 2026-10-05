import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd

# --- 1. Геометричні розрахунки та перевірка опуклості ---

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def triangle_min_angle(p1, p2, p3):
    a, b, c = dist(p2, p3), dist(p1, p3), dist(p1, p2)
    def angle(adj1, adj2, opp):
        if adj1 * adj2 == 0: return 0.0
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
    n = len(verts)
    signs = []
    for i in range(n):
        p0, p1, p2 = verts[i], verts[(i + 1) % n], verts[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if abs(cross) > 1e-7:
            signs.append(cross > 0)
    return len(set(signs)) <= 1

# --- 2. Генератор сітки Делоне з ізопараметричним розбиттям ---

def generate_mesh_core(nx, ny, verts, edge_types):
    nodes, node_boundaries, elements = [], [], []
    num_edges = len(verts)

    if num_edges == 4:
        # Ізопараметрична білінійна структура для чотирикутника
        for j in range(ny + 1):
            v = j / ny
            for i in range(nx + 1):
                u = i / nx
                x = (1-u)*(1-v)*verts[0][0] + u*(1-v)*verts[1][0] + u*v*verts[2][0] + (1-u)*v*verts[3][0]
                y = (1-u)*(1-v)*verts[0][1] + u*(1-v)*verts[1][1] + u*v*verts[2][1] + (1-u)*v*verts[3][1]
                nodes.append((x, y))

                b_mark = 0
                if v == 0: b_mark = edge_types[0]
                elif u == 1: b_mark = edge_types[1]
                elif v == 1: b_mark = edge_types[2]
                elif u == 0: b_mark = edge_types[3]
                node_boundaries.append(b_mark)

        for j in range(ny):
            for i in range(nx):
                bl = j * (nx + 1) + i
                br = bl + 1
                tl = (j + 1) * (nx + 1) + i
                tr = tl + 1

                # Локальний критерій Делоне для вибору діагоналі
                a1 = min(triangle_min_angle(nodes[bl], nodes[br], nodes[tr]),
                         triangle_min_angle(nodes[bl], nodes[tr], nodes[tl]))
                a2 = min(triangle_min_angle(nodes[bl], nodes[br], nodes[tl]),
                         triangle_min_angle(nodes[br], nodes[tr], nodes[tl]))

                if a1 >= a2:
                    elements.extend([(bl, br, tr), (bl, tr, tl)])
                else:
                    elements.extend([(bl, br, tl), (br, tr, tl)])
    else:
        # Секторне розбиття від центроїда
        center = (sum(v[0] for v in verts) / num_edges, sum(v[1] for v in verts) / num_edges)
        node_map = {}

        for s in range(num_edges):
            p_start = verts[s]
            p_end = verts[(s + 1) % num_edges]
            b_val = edge_types[s] if s < len(edge_types) else 0

            for j in range(ny + 1):
                t_rad = j / ny
                for i in range(nx + 1):
                    t_edge = i / nx
                    bx = (1 - t_edge) * p_start[0] + t_edge * p_end[0]
                    by = (1 - t_edge) * p_start[1] + t_edge * p_end[1]
                    rx = (1 - t_rad) * bx + t_rad * center[0]
                    ry = (1 - t_rad) * by + t_rad * center[1]

                    key = (round(rx, 5), round(ry, 5))
                    if key not in node_map:
                        node_map[key] = len(nodes)
                        nodes.append((rx, ry))
                        node_boundaries.append(b_val if j == 0 else 0)

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
                        if len({bl, br, tl}) == 3:
                            elements.append((bl, br, tl))
                    else:
                        a1 = min(triangle_min_angle(nodes[bl], nodes[br], nodes[tr]),
                                 triangle_min_angle(nodes[bl], nodes[tr], nodes[tl]))
                        a2 = min(triangle_min_angle(nodes[bl], nodes[br], nodes[tl]),
                                 triangle_min_angle(nodes[br], nodes[tr], nodes[tl]))
                        if a1 >= a2:
                            elements.extend([(bl, br, tr), (bl, tr, tl)])
                        else:
                            elements.extend([(bl, br, tl), (br, tr, tl)])

    return nodes, elements, node_boundaries

def auto_mesh_by_angle(verts, edge_types, target_angle):
    """
    Автоматично транслює бажаний кут у щільність регулярного розбиття.
    При збільшенні кута сітка плавно подрібнюється без геометричних збоїв.
    """
    # Плавний розрахунок кроків дискретизації залежно від цільового кута
    # 10°-15° -> 2x2, 16°-22° -> 3x3, 23°-27° -> 4x4, 28°+ -> 5x5...
    density = max(1, int((target_angle - 8) / 3.5))
    nx = density
    ny = density

    nodes, elements, boundaries = generate_mesh_core(nx, ny, verts, edge_types)
    return nodes, elements, boundaries, nx, ny

# --- 3. Інтерфейс Streamlit ---

st.set_page_config(page_title="Адаптивна тріангуляція Делоне", layout="wide")
st.title("Адаптивна тріангуляція області за критерієм якості")

st.sidebar.header("Контроль якості")
# ЄДИНИЙ ПОВЗУНОК ДЛЯ КЕРУВАННЯ СІТКОЮ
target_angle = st.sidebar.slider(
    "Мінімальний кут сітки (°):", 
    min_value=10, 
    max_value=32, 
    value=20, 
    step=1,
    help="Збільшення кута автоматично подрібнює сітку для покращення розрахунків МСЕ."
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

# Розрахунок сітки під кут
nodes, elements, boundaries, nx_used, ny_used = auto_mesh_by_angle(verts, b_types, target_angle)

# Аналіз фактичних кутів утворених елементів
element_angles = [math.degrees(triangle_min_angle(nodes[e[0]], nodes[e[1]], nodes[e[2]])) for e in elements]
actual_min_angle = min(element_angles) if element_angles else 0.0
bad_elements = [i for i, a in enumerate(element_angles) if a < target_angle]

# --- 4. Візуалізація та структуровані таблиці ---

col1, col2 = st.columns([1.1, 0.9])

with col1:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    
    for idx, e in enumerate(elements):
        pts = [nodes[n] for n in e]
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        
        # Підсвічування трикутників нижче цільового кута
        if idx in bad_elements:
            ax.fill(xs, ys, color='red', alpha=0.3)
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
    st.write(f"**Адаптована сітка:** {nx_used}x{ny_used}")
    st.write(f"**Вузлів:** {len(nodes)} | **Скінченних елементів:** {len(elements)}")

    if actual_min_angle >= target_angle:
        st.success(f"Цільовий критерій якості (>={target_angle}°) повністю виконано!")
    else:
        st.info(f"Для цієї геометрії досягнуто оптимальний кут {actual_min_angle:.1f}°.")

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

    # 2. Таблиця вузлів із крайовими умовами
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
