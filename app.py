import streamlit as st
import math
import matplotlib.pyplot as plt

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1]) # ((x1-21)^2+(y1-y2)^2)^1/2. Допоміжна функція для знаходження довжини відрізків між вузлами.

def triangle_min_angle(p1, p2, p3):
    a, b, c = dist(p2, p3), dist(p1, p3), dist(p1, p2) # розрахунок довжин сторін трикутника
    def angle(adj1, adj2, opp):
        if adj1 * adj2 == 0: return 0
        val = max(-1.0, min(1.0, (adj1**2 + adj2**2 - opp**2) / (2 * adj1 * adj2))) # теорема косинусів але тут рахується саме кут. max і mix для попадання косинуса в діапазон від -1 до 1
        return math.acos(val) # аркосинус повертає радіани
    return min(angle(b, c, a), angle(a, c, b), angle(a, b, c))

def parse_point(text, default):
    try:
        parts = text.replace(",", " ").split() # коми на пробіли, розділяємо значення
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except ValueError:
        pass
    return default

def generate_mesh(nx, ny, verts, edge_types):
    nodes, node_boundaries, elements = [], [], []
    num_edges = len(verts)

    # Якщо передано стандартний чотирикутник — використовуємо білінійне відображення
    if num_edges == 4:
        # Генерація координат вузлів та їх маркування
        for j in range(ny + 1):
            v = j / ny
            for i in range(nx + 1):
                u = i / nx # створення еталонного квадрата [0, 1]x[0, 1]. Ваги дають в сумі 1. Натягування квадрата на наш чотирикутник
                x = (1-u)*(1-v)*verts[0][0] + u*(1-v)*verts[1][0] + u*v*verts[2][0] + (1-u)*v*verts[3][0] # ізопараметричне білінійне відображення одиничного квадрата
                y = (1-u)*(1-v)*verts[0][1] + u*(1-v)*verts[1][1] + u*v*verts[2][1] + (1-u)*v*verts[3][1] # ваги у своїх вершинах дадуть 1, в інших 0
                nodes.append((x, y))
                
                b_mark = 0 
                if v == 0: b_mark = edge_types[0]      # Нижня грань V1 -> V2. Умова 2-го роду
                elif u == 1: b_mark = edge_types[1]    # Права грань V2 -> V3. Умова 1-го роду
                elif v == 1: b_mark = edge_types[2]    # Верхня грань V3 -> V4. Умова 2-го роду.
                elif u == 0: b_mark = edge_types[3]    # Ліва грань V4 -> V1. Умова 3-го роду.
                node_boundaries.append(b_mark)

        # Триангуляція з максимізацією мінімального кута (критерій Делоне)
        for j in range(ny):
            for i in range(nx): # звернення до 4 вершин області. bottom-left, top-right
                bl = j * (nx + 1) + i
                br = bl + 1
                tl = (j + 1) * (nx + 1) + i
                tr = tl + 1
                
                a1 = min(triangle_min_angle(nodes[bl], nodes[br], nodes[tr]), # діагональ bl - tr. Беремо найменший кут 
                         triangle_min_angle(nodes[bl], nodes[tr], nodes[tl]))
                a2 = min(triangle_min_angle(nodes[bl], nodes[br], nodes[tl]), # діагональ br - tl
                         triangle_min_angle(nodes[br], nodes[tr], nodes[tl]))
                
                if a1 >= a2: # Вибираємо той розріз де мінімальний кут виявився більшим
                    elements.extend([(bl, br, tr), (bl, tr, tl)])
                else:
                    elements.extend([(bl, br, tl), (br, tr, tl)])
    else:
        # Універсальне секторне розбиття для довільного n-кутника. радіально-секторний генератор сітки
        cx = sum(v[0] for v in verts) / num_edges # Знаходимо центр фігури як середнє арифметичне
        cy = sum(v[1] for v in verts) / num_edges
        center = (cx, cy)

        # Створення шарів точок від контуру до центру
        node_map = {}
        for s in range(num_edges): # беремо по черзі кожну сторону фігури
            p_start = verts[s]
            p_end = verts[(s + 1) % num_edges]
            b_val = edge_types[s] if s < len(edge_types) else 0

            for j in range(ny + 1):
                t_rad = j / ny  # 2. кожну знайдену на краю точку ми починаємо стягувати прямою лінією до центру
                for i in range(nx + 1):
                    t_edge = i / nx # 1. ділимо зовнішню сторону многокутника на Nx маленьких кроків і ставимо там точки
                    # Точка на поточному ребрі
                    bx = (1 - t_edge) * p_start[0] + t_edge * p_end[0] # векторна формула лінійного відрізка
                    by = (1 - t_edge) * p_start[1] + t_edge * p_end[1]
                    # Стягування точки до центру області
                    rx = (1 - t_rad) * bx + t_rad * center[0] # з'єднує щойно знайдену точку ребра із центром фігури
                    ry = (1 - t_rad) * by + t_rad * center[1]

                    key = (round(rx, 5), round(ry, 5))
                    if key not in node_map: # перевіряємо чи вузл вже не був записаний в таблицю
                        node_map[key] = len(nodes) # довжина списку і є порядковим номером вузла
                        nodes.append((rx, ry))
                        b_mark = b_val if j == 0 else 0 # маркуємо крайовою умовою лише контурні вузли
                        node_boundaries.append(b_mark)

        # Формування комірок і тріангуляція Делоне в кожному секторі
        for s in range(num_edges):
            p_start = verts[s]
            p_end = verts[(s + 1) % num_edges]
            for j in range(ny):
                for i in range(nx):
                    def get_id(edge_idx, rad_idx):
                        t_r = rad_idx / ny # наскільки ми близько до центру (0 - межа, 1 - центр)
                        t_e = edge_idx / nx # де саме на поточному ребрі (від 0 до 1)
                        bx = (1 - t_e) * p_start[0] + t_e * p_end[0] # перераховуємо координати точок. Точки на контурі
                        by = (1 - t_e) * p_start[1] + t_e * p_end[1]
                        return node_map[(round((1 - t_r) * bx + t_r * center[0], 5),  # точки після зміщення до центру
                                         round((1 - t_r) * by + t_r * center[1], 5))]

                    bl = get_id(i, j)
                    br = get_id(i + 1, j)
                    tl = get_id(i, j + 1)
                    tr = get_id(i + 1, j + 1)

                    # розрізання кожної знайденої комірки на скінченні елементи
                    if tl == tr:
                        if len({bl, br, tl}) == 3: # запобігає появі вироджених елементів із нульовою площею й додає один трикутник із вершиною в центроїді
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

st.set_page_config(page_title="Триангуляція Делоне (МСЕ)", layout="wide")
st.title("Триангуляція Делоне для довільних фігур")

st.sidebar.header("Параметри розбиття")
nx = st.sidebar.number_input("Густина Nx", min_value=1, max_value=25, value=2)
ny = st.sidebar.number_input("Густина Ny", min_value=1, max_value=25, value=2)

st.sidebar.header("Контроль якості сітки")
angle_threshold = st.sidebar.slider("Критичний поріг кута (°)", min_value=5, max_value=45, value=20)

st.sidebar.header("Геометрія області")
poly_type = st.sidebar.selectbox("Оберіть форму фігури:", 
                                 ["Варіант 12 (4-кутник)", "Трикутник (3 кути)", "П'ятикутник (5 кутів)", "Власна кількість вершин"])

if poly_type == "Варіант 12 (4-кутник)":
    raw_verts = ["1.0, 0.0", "2.0, 0.0", "0.0, 2.0", "0.0, 1.0"]
    b_types = [2, 1, 2, 3]
elif poly_type == "Трикутник (3 кути)":
    raw_verts = ["0.0, 0.0", "2.0, 0.0", "1.0, 1.73"]
    b_types = [1, 2, 3]
elif poly_type == "П'ятикутник (5 кутів)":
    raw_verts = ["0.0, 0.0", "2.0, 0.0", "2.5, 1.5", "1.0, 2.5", "-0.5, 1.5"]
    b_types = [1, 2, 2, 3, 2]
else:
    num_pts = st.sidebar.number_input("Кількість вершин:", min_value=3, max_value=10, value=4)
    raw_verts = [f"{math.cos(2*math.pi*i/num_pts):.2f}, {math.sin(2*math.pi*i/num_pts):.2f}" for i in range(num_pts)]
    b_types = [(i % 3) + 1 for i in range(num_pts)]

verts = []
st.sidebar.caption("Координати вершин (обхід проти годинникової стрілки):")
for i, def_val in enumerate(raw_verts):
    t = st.sidebar.text_input(f"V{i+1}:", value=def_val)
    verts.append(parse_point(t, (0.0, 0.0)))

nodes, elements, boundaries = generate_mesh(nx, ny, verts, b_types)

element_angles = []
bad_elements = []
global_min_angle = 180.0

for idx, element in enumerate(elements): # idx номер трикутника, element трійка індексів вершин
    p1, p2, p3 = nodes[element[0]], nodes[element[1]], nodes[element[2]] # витягуємо координати вузлів
    min_deg = math.degrees(triangle_min_angle(p1, p2, p3)) # переводимо радіани в градуси
    element_angles.append(min_deg)
    
    if min_deg < global_min_angle: # просто знаходимо мінімальний кут всієї сітки
        global_min_angle = min_deg
    if min_deg < angle_threshold: # порівнюємо мінімальний кут з пороговим значенням
        bad_elements.append(idx)

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Побудова трикутників із підсвічуванням елементів, що порушують поріг
    for idx, element in enumerate(elements):
        pts = [nodes[n] for n in element]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        
        if idx in bad_elements: # якщо у трикутника є кут, який порушує поріг, то він стає червоним
            ax.fill(xs, ys, color='red', alpha=0.3)
            
        pts.append(pts[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color='black', linewidth=0.8, alpha=0.6)
        
        cx, cy = sum(xs)/3, sum(ys)/3 # знаходимо центроїд аби намалювати тут номер трикутника
        ax.text(cx, cy, f"E{idx}", color='blue', fontsize=8, ha='center', va='center', fontweight='bold')

    # Побудова та маркування вузлів
    colors = {0: 'gray', 1: 'red', 2: 'green', 3: 'purple'}
    for idx, (node, b_mark) in enumerate(zip(nodes, boundaries)):
        ax.plot(node[0], node[1], color=colors.get(b_mark, 'black'), marker='o', markersize=6)
        ax.text(node[0] + 0.02, node[1] + 0.02, str(idx), fontsize=8)

    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_title(f"Сітка: {nx}x{ny} | Мін. кут: {global_min_angle:.1f}°")
    st.pyplot(fig)

with col2:
    st.write(f"**Вузлів:** {len(nodes)} | **Елементів:** {len(elements)}")
    st.metric("Найменший кут сітки", f"{global_min_angle:.2f}°")
    
    # Індикатори контролю мінімального кута
    if len(bad_elements) == 0:
        st.success(f"Усі елементи задовольняють поріг >= {angle_threshold}°")
    else:
        st.warning(f"Елементів із кутом < {angle_threshold}°: {len(bad_elements)} шт. (підсвічено червоним)")
    
    elements_log = "\n".join([f"E{i}: {e} | Мін. кут: {element_angles[i]:.1f}°" for i, e in enumerate(elements)])
    st.text_area("Масив зв'язності (інцидентності)", elements_log, height=180)
    
    nodes_log = "\n".join([f"N{i}: ({n[0]:.2f}, {n[1]:.2f}) | Границя: {b}" for i, (n, b) in enumerate(zip(nodes, boundaries))])
    st.text_area("Вузли (ID: X, Y | Маркер)", nodes_log, height=180)

st.markdown("---")
with st.expander("📄 Звіт до виконання завдання №1 (Триангуляція області)", expanded=True):
    st.markdown(r"""
### 1. Формулювання завдання
Розглядається крайова задача для двовимірного диференціального рівняння:
$$-\sum_{i,j=1}^{2} \frac{\partial}{\partial x_i} a_{ij}(x) \frac{\partial u}{\partial x_j} + du = f, \quad x_1, x_2 \in \Omega$$

**Параметри Варіанту 12:**
* Коефіцієнти: $a_{11} = 8$, $a_{12} = a_{21} = 0$, $a_{22} = 2$, $d = 0$, $f = 1$.
* Область $\Omega$: чотирикутник з вершинами $V_1(1,0)$, $V_2(2,0)$, $V_3(0,2)$, $V_4(0,1)$.
* Крайові умови на гранях:
  * $V_2-V_3$: умова 1-го роду ($u = 0$, Діріхле).
  * $V_1-V_2$ та $V_3-V_4$: умова 2-го роду ($Nu = 0$, Неймана).
  * $V_4-V_1$: умова 3-го роду ($\beta Nu + \delta(u - u_c) = 0$, Робіна).

---

### 2. Покроковий алгоритм роботи програми
1. **Зчитування параметрів:** отримання координат вершин фігури, кількості кроків розбиття ($Nx, Ny$) та допустимого порогу кута.
2. **Генерація вузлів:** 
   * Для 4-кутника: розбиття еталонного квадрата $[0, 1] \times [0, 1]$ на кроки $Nx, Ny$ та білінійне перенесення точок у фізичну область.
   * Для довільного $n$-кутника: поділ на сектори від центроїда зі стягуванням концентричних шарів точок до центру та усуненням дублікатів.
3. **Маркування границь:** присвоєння кожному згенерованому вузлу мітки ($1, 2, 3$ для відповідних зовнішніх ребер та $0$ для всіх внутрішніх точок).
4. **Тріангуляція за Делоне:** обхід чотирикутних комірок, розрахунок кутів для двох можливих діагоналей через теорему косинусів та вибір розрізу з більшим мінімальним кутом.
5. **Контроль якості та візуалізація:** знаходження найменшого кута для кожного трикутника, виявлення елементів нижче встановленого порогу, малювання сітки з червоним підсвічуванням дефектів та формування таблиць даних.

---

### 3. Отримані результати
* Створено вебдодаток для інтерактивної побудови та налаштування сітки під задані параметри розбиття.
* Сформовано повний список координат вузлів із числовими маркерами крайових умов.
* Згенеровано масив зв'язності (інцидентності) трикутних елементів.
* Реалізовано графічний аудит якості сітки з числовим відображенням мінімального кута.

---

### 4. Висновок
Завдання з геометричної дискретизації області виконано в повному обсязі. Побудовано коректну трикутну сітку за критерієм Делоне, виключено вироджені елементи та сформовано всі вихідні масиви вершин, зв'язків і крайових міток.
""")
