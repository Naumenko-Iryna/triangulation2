import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.path import Path

# --- 1. Геометрія та характеристики трикутника ---

def get_triangle_data(pts):
    """Рахує найменший кут (град) і центр описаного кола."""
    A, B, C = pts[0], pts[1], pts[2]
    a, b, c = np.linalg.norm(B - C), np.linalg.norm(A - C), np.linalg.norm(A - B)
    
    # Теорема косинусів
    cos_vals = [
        np.clip((b**2 + c**2 - a**2) / (2 * b * c), -1.0, 1.0),
        np.clip((a**2 + c**2 - b**2) / (2 * a * c), -1.0, 1.0),
        np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0)
    ]
    min_deg = float(np.min(np.degrees(np.arccos(cos_vals))))

    # Точка Штайнера (центр описаного кола)
    D = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
    if abs(D) < 1e-9:
        return min_deg, np.mean(pts, axis=0)

    ux = ((A[0]**2 + A[1]**2)*(B[1] - C[1]) + (B[0]**2 + B[1]**2)*(C[1] - A[1]) + (C[0]**2 + C[1]**2)*(A[1] - B[1])) / D
    uy = ((A[0]**2 + A[1]**2)*(C[0] - B[0]) + (B[0]**2 + B[1]**2)*(A[0] - C[0]) + (C[0]**2 + C[1]**2)*(B[0] - A[0])) / D
    return min_deg, np.array([ux, uy])

def check_convexity(pts):
    """Знакосталість 2D векторного (косого) добутку без виклику np.cross."""
    n = len(pts)
    signs = []
    for i in range(n):
        v1 = pts[(i + 1) % n] - pts[i]
        v2 = pts[(i + 2) % n] - pts[(i + 1) % n]
        # Двовимірний векторний добуток: x1*y2 - y1*x2
        cross_2d = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross_2d) > 1e-7:
            signs.append(cross_2d > 0)
    return len(set(signs)) <= 1

# --- 2. Генерація сітки Делоне з контролем кута ---

def generate_mesh(poly, edge_types, target_angle, max_iter=150):
    path = Path(poly)
    n_v = len(poly)
    
    # 1. Початкове розбиття контуру
    pts = []
    seg_len = sum(np.linalg.norm(poly[(i+1)%n_v] - poly[i]) for i in range(n_v)) / (n_v * 3)
    for i in range(n_v):
        p1, p2 = poly[i], poly[(i+1)%n_v]
        n_segs = max(1, int(np.ceil(np.linalg.norm(p2 - p1) / seg_len)))
        for j in range(n_segs):
            pts.append(p1 + (p2 - p1) * (j / n_segs))
    pts = np.array(pts)

    # 2. Адаптивне подрібнення (Рупперт)
    for _ in range(max_iter):
        tri = Delaunay(pts)
        simplices = tri.simplices[path.contains_points(np.mean(pts[tri.simplices], axis=1), radius=-1e-5)]
        
        bad_elems = []
        for s in simplices:
            deg, cc = get_triangle_data(pts[s])
            if deg < target_angle:
                bad_elems.append((target_angle - deg, cc, np.mean(pts[s], axis=0)))
        
        if not bad_elems:
            break
            
        bad_elems.sort(key=lambda x: x[0], reverse=True)
        added = False
        for _, cc, centroid in bad_elems:
            candidate = cc if path.contains_point(cc) else centroid
            if np.min(np.linalg.norm(pts - candidate, axis=1)) > 1e-3:  # захист від злипання
                pts = np.vstack([pts, candidate])
                added = True
                break
        if not added:
            break

    # 3. Фінальні елементи та маркування
    tri = Delaunay(pts)
    simplices = tri.simplices[path.contains_points(np.mean(pts[tri.simplices], axis=1), radius=-1e-5)]
    
    boundaries = []
    for p in pts:
        mark = 0
        for i in range(n_v):
            p1, p2 = poly[i], poly[(i+1)%n_v]
            if abs(np.linalg.norm(p - p1) + np.linalg.norm(p2 - p) - np.linalg.norm(p2 - p1)) < 1e-4:
                mark = edge_types[i]
                break
        boundaries.append(mark)
        
    return pts, simplices, boundaries

# --- 3. Інтерфейс Streamlit ---

st.set_page_config(page_title="Триангуляція Делоне", layout="wide")
st.title("Триангуляція Делоне за критерієм мінімального кута")

st.sidebar.header("Параметри")
target_angle = st.sidebar.slider("Мінімальний кут (°)", 10.0, 32.0, 20.0, 1.0)
preset = st.sidebar.selectbox("Область:", ["Варіант 12 (4-кутник)", "Трикутник", "П'ятикутник"])

presets = {
    "Варіант 12 (4-кутник)": ([[1.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 1.0]], [2, 1, 2, 3]),
    "Трикутник": ([[0.0, 0.0], [2.0, 0.0], [1.0, 1.73]], [1, 2, 3]),
    "П'ятикутник": ([[0.0, 0.0], [2.0, 0.0], [2.5, 1.5], [1.0, 2.5], [-0.5, 1.5]], [1, 2, 2, 3, 2])
}
poly_default, b_types = presets[preset]

verts = np.array([[float(x) for x in st.sidebar.text_input(f"V{i+1}:", f"{p[0]}, {p[1]}").replace(',', ' ').split()[:2]] 
                  for i, p in enumerate(poly_default)])

if not check_convexity(verts):
    st.sidebar.warning("⚠️ Фігура неопукла!")

# Генерація
nodes, elements, boundaries = generate_mesh(verts, b_types, target_angle)
angles = [get_triangle_data(nodes[s])[0] for s in elements]
min_deg_found = min(angles) if len(angles) > 0 else 0.0

# --- 4. Візуалізація та Таблиці ---

col1, col2 = st.columns([1.1, 0.9])
with col1:
    fig, ax = plt.subplots(figsize=(6, 6))
    for idx, (s, ang) in enumerate(zip(elements, angles)):
        pts_tri = np.vstack([nodes[s], nodes[s[0]]])
        if ang < target_angle:
            ax.fill(pts_tri[:,0], pts_tri[:,1], color='red', alpha=0.35)
        ax.plot(pts_tri[:,0], pts_tri[:,1], 'k-', lw=0.7)
        c = np.mean(nodes[s], axis=0)
        ax.text(c[0], c[1], f"E{idx}", color='blue', fontsize=7, ha='center', va='center')
        
    colors = {0: 'gray', 1: 'red', 2: 'green', 3: 'purple'}
    for idx, (p, b) in enumerate(zip(nodes, boundaries)):
        ax.plot(p[0], p[1], 'o', color=colors.get(b, 'black'), ms=4)
        ax.text(p[0]+0.02, p[1]+0.02, str(idx), fontsize=7)
        
    ax.set_aspect('equal')
    ax.grid(True, ls=':', alpha=0.5)
    ax.set_title(f"Фактичний мін. кут: {min_deg_found:.1f}° (Ціль: {target_angle:.0f}°)")
    st.pyplot(fig)

with col2:
    st.metric("Мінімальний кут", f"{min_deg_found:.2f}°", delta=f"{min_deg_found - target_angle:.2f}°")
    st.write(f"**Вузлів:** {len(nodes)} | **Елементів:** {len(elements)}")
    
    st.write("**Таблиця зв'язності (Topology)**")
    st.dataframe(pd.DataFrame(elements, columns=['N1', 'N2', 'N3']).assign(**{"Мін. кут (°)": np.round(angles, 1)}), height=170, use_container_width=True)
    
    st.write("**Таблиця вузлів та крайових міток**")
    b_map = {0: "0 (Внутрішній)", 1: "1 (Діріхле)", 2: "2 (Неймана)", 3: "3 (Робіна)"}
    st.dataframe(pd.DataFrame(nodes, columns=['X', 'Y']).assign(Маркер=boundaries, Умова=[b_map.get(b, str(b)) for b in boundaries]), height=170, use_container_width=True)
