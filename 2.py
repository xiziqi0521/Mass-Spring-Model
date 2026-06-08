import taichi as ti
import math

# 初始化 Taichi，使用 GPU 加速运算
ti.init(arch=ti.gpu)

# 物理与网格参数
N = 20             # 布料网格分辨率 N x N
mass = 1.0         # 质点质量
dt = 5e-4          # 时间步长
k_s = 10000.0      # 结构弹簧劲度系数
k_shear = 8000.0   # 剪切弹簧劲度系数 (稍低，避免过刚)
k_bend = 3000.0    # 弯曲弹簧劲度系数 (更低，弯曲较软)
k_d = 1.0          # 阻尼系数
gravity = ti.Vector([0.0, -9.8, 0.0])
max_velocity = 50.0  # 速度上限，防止数值爆炸

# 球体碰撞参数
# shape=(1,) 而非 shape=()，scene.particles 需要 shape[0] 存在
sphere_center = ti.Vector.field(3, dtype=float, shape=(1,))
sphere_radius = ti.field(dtype=float, shape=())

# 定义 Taichi 数据场
x = ti.Vector.field(3, dtype=float, shape=N * N)       # 位置
v = ti.Vector.field(3, dtype=float, shape=N * N)       # 速度
f = ti.Vector.field(3, dtype=float, shape=N * N)       # 受力
is_fixed = ti.field(dtype=int, shape=N * N)            # 是否为固定点

# 隐式欧拉专用的预测缓存场
x_next = ti.Vector.field(3, dtype=float, shape=N * N)
v_next = ti.Vector.field(3, dtype=float, shape=N * N)
f_next = ti.Vector.field(3, dtype=float, shape=N * N)

# 弹簧数据场
# 每个质点最多有：2 结构 + 2 剪切 + 2 弯曲 = 6 组，总量翻倍留余
max_springs = N * N * 8
spring_indices = ti.field(dtype=int, shape=max_springs * 2)
spring_pairs = ti.Vector.field(2, dtype=int, shape=max_springs)
spring_lengths = ti.field(dtype=float, shape=max_springs)
spring_stiffness = ti.field(dtype=float, shape=max_springs)  # 每根弹簧的劲度系数
num_springs = ti.field(dtype=int, shape=())

# 控制选做功能开关
enable_shear = ti.field(dtype=int, shape=())
enable_bending = ti.field(dtype=int, shape=())
enable_collision = ti.field(dtype=int, shape=())

# ============ 初始化 ============

@ti.kernel
def init_positions():
    """初始化质点位置与固定状态"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j
        x[idx] = ti.Vector([i * 0.05 - 0.5, 0.8, j * 0.05 - 0.5])
        v[idx] = ti.Vector([0.0, 0.0, 0.0])
        f[idx] = ti.Vector([0.0, 0.0, 0.0])
        # 固定第一排的两个角点
        if j == 0 and (i == 0 or i == N - 1):
            is_fixed[idx] = 1
        else:
            is_fixed[idx] = 0

@ti.kernel
def init_springs():
    """初始化弹簧：结构 + 剪切 + 弯曲 (根据开关决定是否添加)"""
    for i, j in ti.ndrange(N, N):
        idx = i * N + j

        # -------- 结构弹簧 (Structural)：始终启用 --------
        # 右侧相邻
        if i < N - 1:
            nb = (i + 1) * N + j
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()
            spring_stiffness[c] = k_s

        # 下方相邻
        if j < N - 1:
            nb = i * N + (j + 1)
            c = ti.atomic_add(num_springs[None], 1)
            spring_pairs[c] = ti.Vector([idx, nb])
            spring_lengths[c] = (x[idx] - x[nb]).norm()
            spring_stiffness[c] = k_s

        # -------- 剪切弹簧 (Shear)：对角线连接 --------
        if enable_shear[None] == 1:
            # 右下对角
            if i < N - 1 and j < N - 1:
                nb = (i + 1) * N + (j + 1)
                c = ti.atomic_add(num_springs[None], 1)
                spring_pairs[c] = ti.Vector([idx, nb])
                spring_lengths[c] = (x[idx] - x[nb]).norm()
                spring_stiffness[c] = k_shear

            # 左下对角
            if i > 0 and j < N - 1:
                nb = (i - 1) * N + (j + 1)
                c = ti.atomic_add(num_springs[None], 1)
                spring_pairs[c] = ti.Vector([idx, nb])
                spring_lengths[c] = (x[idx] - x[nb]).norm()
                spring_stiffness[c] = k_shear

        # -------- 弯曲弹簧 (Bending)：间隔一个质点连接 --------
        if enable_bending[None] == 1:
            # 水平方向间隔1
            if i < N - 2:
                nb = (i + 2) * N + j
                c = ti.atomic_add(num_springs[None], 1)
                spring_pairs[c] = ti.Vector([idx, nb])
                spring_lengths[c] = (x[idx] - x[nb]).norm()
                spring_stiffness[c] = k_bend

            # 垂直方向间隔1
            if j < N - 2:
                nb = i * N + (j + 2)
                c = ti.atomic_add(num_springs[None], 1)
                spring_pairs[c] = ti.Vector([idx, nb])
                spring_lengths[c] = (x[idx] - x[nb]).norm()
                spring_stiffness[c] = k_bend

@ti.kernel
def init_spring_indices():
    """同步渲染索引"""
    for i in range(num_springs[None]):
        spring_indices[i * 2] = spring_pairs[i][0]
        spring_indices[i * 2 + 1] = spring_pairs[i][1]

def init_cloth():
    """按顺序调用各初始化 kernel，确保 GPU 同步"""
    num_springs[None] = 0
    init_positions()
    init_springs()
    init_spring_indices()

# ============ 力学计算 ============

@ti.func
def compute_forces_on(pos: ti.template(), vel: ti.template(), force: ti.template()):
    """计算所有力 (重力 + 阻尼 + 弹簧力)"""
    # 第一阶段：清空受力，施加重力与阻尼
    for i in range(N * N):
        force[i] = gravity * mass - k_d * vel[i]

    # 第二阶段：累加弹簧力 (使用 atomic_add 保证多线程安全)
    for i in range(num_springs[None]):
        idx_a = spring_pairs[i][0]
        idx_b = spring_pairs[i][1]
        pos_a = pos[idx_a]
        pos_b = pos[idx_b]
        d = pos_a - pos_b
        dist = d.norm()
        if dist > 1e-6:
            d_normalized = d / dist
            f_spring = -spring_stiffness[i] * (dist - spring_lengths[i]) * d_normalized
            ti.atomic_add(force[idx_a], f_spring)
            ti.atomic_add(force[idx_b], -f_spring)

@ti.func
def clamp_velocity(vel: ti.template(), idx: int):
    """速度钳制，防止数值爆炸"""
    vel_norm = vel[idx].norm()
    if vel_norm > max_velocity:
        vel[idx] = vel[idx] / vel_norm * max_velocity

@ti.func
def resolve_sphere_collision(pos: ti.template(), vel: ti.template(), idx: int):
    """处理质点与球体的碰撞：将质点推出球体表面，并反弹速度法向分量"""
    if enable_collision[None] == 1:
        sc = sphere_center[0]
        sr = sphere_radius[None]
        d = pos[idx] - sc
        dist = d.norm()
        if dist < sr + 0.01:  # 0.01 为质点半径偏移量
            # 将质点推到球面外
            normal = d / (dist + 1e-6)
            pos[idx] = sc + normal * (sr + 0.01)
            # 消除法向速度分量（完全非弹性碰撞 + 轻微摩擦）
            vn = vel[idx].dot(normal)
            if vn < 0.0:  # 只处理向球心方向运动的情况
                vel[idx] -= vn * normal  # 移除法向速度
                vel[idx] *= 0.9          # 切向摩擦衰减

# ============ 积分求解器 ============

@ti.kernel
def step_explicit():
    """显式欧拉 (Explicit Euler)"""
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            x[i] += v[i] * dt
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            resolve_sphere_collision(x, v, i)

@ti.kernel
def step_semi_implicit():
    """半隐式欧拉 (Semi-Implicit Euler)"""
    compute_forces_on(x, v, f)
    for i in range(N * N):
        if is_fixed[i] == 0:
            v[i] += (f[i] / mass) * dt
            clamp_velocity(v, i)
            x[i] += v[i] * dt
            resolve_sphere_collision(x, v, i)

@ti.kernel
def step_implicit_iter():
    """隐式欧拉 (Implicit Euler) - 定点迭代"""
    for i in range(N * N):
        v_next[i] = v[i]
        x_next[i] = x[i]
    for _ in ti.static(range(3)):
        compute_forces_on(x_next, v_next, f_next)
        for i in range(N * N):
            if is_fixed[i] == 0:
                v_next[i] = v[i] + (f_next[i] / mass) * dt
                clamp_velocity(v_next, i)
                x_next[i] = x[i] + v_next[i] * dt
                resolve_sphere_collision(x_next, v_next, i)
    for i in range(N * N):
        v[i] = v_next[i]
        x[i] = x_next[i]

# ============ 主函数 ============

def main():
    # 初始化选做开关（默认全部关闭，与原始效果一致）
    enable_shear[None] = 0
    enable_bending[None] = 0
    enable_collision[None] = 0

    # 球体初始参数
    sphere_center[0] = ti.Vector([0.0, 0.2, 0.0])
    sphere_radius[None] = 0.25

    init_cloth()

    # 建立 GGUI 窗口
    window = ti.ui.Window("Games101 - Mass Spring System (Extended)", (800, 800))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.5, 2.0)
    camera.lookat(0.0, 0.0, 0.0)

    current_method = 1  # 0: 显式, 1: 半隐式, 2: 隐式
    paused = False

    # 用 Python 变量缓存开关状态，方便检测变化（切换弹簧类型需要重建弹簧）
    py_shear = False
    py_bending = False
    py_collision = False

    while window.running:
        # =========== GUI 控制面板 ===========
        window.GUI.begin("Control Panel", 0.02, 0.02, 0.40, 0.60)

        window.GUI.text("--- Integration Method ---")
        prefix_0 = "[*] " if current_method == 0 else "[ ] "
        prefix_1 = "[*] " if current_method == 1 else "[ ] "
        prefix_2 = "[*] " if current_method == 2 else "[ ] "

        if window.GUI.button(prefix_0 + "Explicit Euler (Explosive)"):
            current_method = 0
            init_cloth()
        if window.GUI.button(prefix_1 + "Semi-Implicit Euler (Stable)"):
            current_method = 1
            init_cloth()
        if window.GUI.button(prefix_2 + "Implicit Euler (Damped)"):
            current_method = 2
            init_cloth()

        window.GUI.text("")
        window.GUI.text("--- Spring Model (Bonus) ---")

        # 剪切弹簧开关
        shear_label = "[ON]  Shear Springs" if py_shear else "[OFF] Shear Springs"
        if window.GUI.button(shear_label):
            py_shear = not py_shear
            enable_shear[None] = 1 if py_shear else 0
            init_cloth()  # 重建弹簧拓扑

        # 弯曲弹簧开关
        bending_label = "[ON]  Bending Springs" if py_bending else "[OFF] Bending Springs"
        if window.GUI.button(bending_label):
            py_bending = not py_bending
            enable_bending[None] = 1 if py_bending else 0
            init_cloth()

        window.GUI.text("")
        window.GUI.text("--- Collision (Bonus) ---")

        # 碰撞球体开关
        collision_label = "[ON]  Sphere Collision" if py_collision else "[OFF] Sphere Collision"
        if window.GUI.button(collision_label):
            py_collision = not py_collision
            enable_collision[None] = 1 if py_collision else 0
            init_cloth()

        window.GUI.text("")
        window.GUI.text("--- Simulation Control ---")

        pause_label = "Resume Simulation" if paused else "Pause Simulation"
        if window.GUI.button(pause_label):
            paused = not paused

        if window.GUI.button("Reset Cloth"):
            init_cloth()

        window.GUI.end()
        # ====================================

        if not paused:
            for _ in range(40):
                if current_method == 0:
                    step_explicit()
                elif current_method == 1:
                    step_semi_implicit()
                elif current_method == 2:
                    step_implicit_iter()

        # 渲染场景
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        # 布料粒子与弹簧线框
        scene.particles(x, radius=0.015, color=(0.2, 0.6, 1.0))
        scene.lines(x, indices=spring_indices, width=1.5, color=(0.8, 0.8, 0.8))

        # 若碰撞启用，绘制球体（Taichi 用粒子近似）
        if py_collision:
            scene.particles(sphere_center, radius=sphere_radius[None], color=(0.9, 0.3, 0.2))

        canvas.scene(scene)
        window.show()

if __name__ == '__main__':
    main()