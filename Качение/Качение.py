# КАК КАТИТСЯ КОЛЕСО

# Поверхность задана графиком функции y=f(x) ("вид сбоку")
# Моделируется качение обруча, диска, сферы или шара по этой поверхности без проскальзывания под действием силы тяжести
# Сохраняются две анимации: одна "в реальном времени", другая замедленная

# ЗАДАНИЯ ПО ДОРАБОТКЕ

# 1) Разобраться в коде, его математических и физических основаниях
# 2) Реализовать учёт силы трения качения, а также условие перехода от качения к скольжению или падению
# 3) Добавить моделирование броска мяча с учётом сопротивления воздуха и вращения при полёте (эффект Магнуса),
# с возможностью отскока мяча от поверхности или качения по ней
# 4) Перейти от двумерной картины к трёхмерной, с произвольными поверхностями и траекториями движения
# 5) Смоделировать движение более сложных объектов, в том числе допускающих планирующий полёт

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import matplotlib as mpl
from scipy.integrate import solve_ivp

# Здесь можно задать поверхность
def surface_function(x: float) -> float:
    return 0.05 * x**2 + 0.15 * np.sin(2 * x)

x_min = -10 # размеры окна визуализации
x_max = 10
y_min = -1
y_max = 10

# Форма тела. Варианты: "solid_sphere" (сплошной шар), "hollow_sphere" (полая сфера), "solid_disk" (диск), "ring" (обруч)
BODY_TYPE = "solid_disk"

# Коэффициент beta в формуле момента инерции I = beta * m * R^2 в зависимости от формы тела
BETA_VALUES = {
    "solid_sphere": 2 / 5,   # сплошной шар
    "hollow_sphere": 2 / 3,  # полая сфера
    "solid_disk": 1 / 2,     # диск
    "ring": 1.0,             # кольцо
}

BODY_NAMES = {
    "solid_sphere": "Сплошной шар",
    "hollow_sphere": "Полая сфера",
    "solid_disk": "Диск",
    "ring": "Кольцо",
}

beta = float(BETA_VALUES[BODY_TYPE])
body_name = BODY_NAMES[BODY_TYPE]

# Начальные условия
x0 = -7.0   # м
v_c0 = 0.0  # м/с
phi0 = 0.0  # рад

# Приближённое вычисление первой и второй производных функции f. Можно заменить на точное нахождение, если f задана формулой
# Слишком маленький шаг H_DERIV или H_DERIV2 может привести к неустойчивости расчётов, слишком большой - сглаживает детали
H_DERIV = 1e-5
H_DERIV2 = 2e-4

def numerical_derivative(f, x: float, h: float = H_DERIV) -> float:
    return (f(x + h) - f(x - h)) / (2.0 * h)

def numerical_second_derivative(f, x: float, h: float = H_DERIV2) -> float:
    return (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)

g = 9.81  # Ускорение свободного падения
R = 0.5   # Радиус колеса

SIMULATION_TIME = 10.0 # время наблюдения в секундах
ANIMATION_FPS = 30 # плавность анимаций, количество кадров в секунду
PLAYBACK_SLOWDOWN = 4 # целочисленный коэффициент замедления для замедленной анимации

# Общее число кадров замедленной анимации (в реальной анимации будут те же кадры, но с пропусками)
frames_slow = int(SIMULATION_TIME * ANIMATION_FPS * PLAYBACK_SLOWDOWN) + 1
t_frames_slow = np.linspace(0.0, SIMULATION_TIME, frames_slow) # моменты времени, соответствующие кадрам замедленной анимации
# Индексы кадров для реальной и замедленной анимаций
idx_frames_real = range(0, len(t_frames_slow), PLAYBACK_SLOWDOWN)
idx_frames_slow = range(0, len(t_frames_slow), 1)

# Возвращает производную p(x)=f'(x), функцию L(x)=sqrt(1+p^2) (она входит в формулу элемента дуги кривой ds=L(x)dx)
# и кривизну kappa(x)=f''(x)/(1+p^2)^(3/2)
def slope_L_kappa(x: float):
    p = numerical_derivative(surface_function, x)
    L = float(np.sqrt(1.0 + p * p))
    f2 = numerical_second_derivative(surface_function, x)
    kappa = float(f2 / (L**3))
    return p, L, kappa

# Возвращает координаты точки контакта колеса с поверхностью, координаты центра колеса, f'(x) и L(x) в точке контакта
def contact_and_center(x: float):

    y = float(surface_function(x))
    p = numerical_derivative(surface_function, x)
    L = float(np.sqrt(1.0 + p * p))

    # Единичная нормаль к графику y=f(x)
    n_x = -p / L
    n_y = 1.0 / L

    x_c = float(x + R * n_x)
    y_c = float(y + R * n_y)

    return float(x), y, x_c, y_c, float(p), L


# Правая часть системы дифференциальных уравнений для состояния колеса (x, v_c, phi)
# x - абсцисса точки контакта, v_c - скорость центра колеса C, phi - угол, на который повернулось колесо
# Центр колеса C и точка контакта колеса с поверхностью r связаны равенством C=r+Rn, где R - радиус, n - вектор нормали
# Дифференцируем по времени и применяем формулы Френе для плоской кривой: v_c=(1-R*kappa)v_s,
# где kappa - кривизна кривой в точке контакта, v_s - скорость точки контакта
# Первое уравнение: dx/dt=(ds/dt)/L=v_s/L=v_c/((1-R*kappa)L), определение L см. выше
# (Это же равно v_c*cos(theta)/(1-R*kappa), где theta - угол наклона касательной,
# таким образом, в случае малой кривизны графика скорость изменения x примерно равна v_c*cos(theta) - почему так?)
# Если нет проскальзывания, то v_c=-R*omega, где omega - угловая скорость вращения
# Дифференцируем по времени: a_c=-R*eps, где a_c - тангенциальное ускорение центра (проекция вектора ускорения на касательную),
# eps - угловое ускорение. Закон динамики вращения тела, аналогичный второму закону Ньютона: M=I*eps,
# M - момент силы, I - момент инерции. Поворачивать колесо вокруг центра может только сила трения, так что M - её момент
# Для момента инерции шара/сферы/диска/кольца известны формулы, включающие параметр beta. Так находим выражение для силы трения
# Из второго закона Ньютона для a_c получаем второе уравнение: dv_c/dt = a_c = -(g/(1+beta))*sin(theta),  sin(theta)=p/L 
# Третье уравнение: d phi/dt = omega = -v_c/R
def rhs(t, y):
    x = float(y[0])
    v_c = float(y[1])
    phi = float(y[2])
    p, L, kappa = slope_L_kappa(x)
    dxdt = v_c / ((1.0 - R * kappa) * L)
    dvdt = -(g / (1.0 + beta)) * (p / L)
    dphidt = -(v_c / R)
    return [dxdt, dvdt, dphidt]

# Решить систему дифференциальных уравнений!
def integrate_trajectory(y0):

    sol = solve_ivp(
        rhs,
        t_span=(0.0, SIMULATION_TIME),
        y0=np.array(y0, dtype=float),
        t_eval=t_frames_slow,
        method='DOP853',
        rtol=1e-7,
        atol=1e-9,
    )

    if sol.status < 0:
        raise RuntimeError(f"solve_ivp завершился с ошибкой: {sol.message}")

    x_arr = sol.y[0].astype(float)
    vs_arr = sol.y[1].astype(float)
    phi_arr = sol.y[2].astype(float)

    if np.isnan(x_arr).any() or np.isnan(vs_arr).any() or np.isnan(phi_arr).any():
        raise RuntimeError("Траектория содержит NaN. Проверьте параметры и поверхность.")

    return x_arr, vs_arr, phi_arr

x_traj, vc_traj, phi_traj = integrate_trajectory([x0, v_c0, phi0])

# Создание анимаций
def create_animation(frame_indices, filename, title):

    plt.style.use("default")
    mpl.rcParams["font.family"] = "DejaVu Sans"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м")
    ax.set_title(title, pad=12)

    ax.grid(True, linestyle="--", alpha=0.5)

    # Оси координат
    ax.axhline(0, linewidth=1, alpha=0.6)
    ax.axvline(0, linewidth=1, alpha=0.6)

    # Текстовое поле для отображения времени
    time_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        ha="left", va="top"
    )

    # Поверхность
    x_vals = np.linspace(-10, 10, 600)
    y_vals = surface_function(x_vals)
    ax.plot(x_vals, y_vals, linewidth=2.0)

    # Колесо
    wheel = Circle((0, 0), R, fill=False, linewidth=2.2)
    ax.add_patch(wheel)

    # "Спицы" для визуализации вращения
    num_spokes = 8
    base_angles = np.linspace(0, 2 * np.pi, num_spokes, endpoint=False)
    spokes = []
    for _ in base_angles:
        line, = ax.plot([], [], linewidth=1.2)
        spokes.append(line)

    # Точка контакта
    contact_dot = Circle((0, 0), 0.06, fill=True, color='black')
    ax.add_patch(contact_dot)

    def update(frame_id):
        idx = frame_indices[frame_id]

        x = float(x_traj[idx])
        phi = float(phi_traj[idx])
        t_phys = float(t_frames_slow[idx])

        x_con, y_con, x_c, y_c, p, L = contact_and_center(x)

        wheel.center = (x_c, y_c)
        contact_dot.center = (x_con, y_con)
        time_text.set_text(f"t = {t_phys:.2f} с")

        for i, a0 in enumerate(base_angles):
            x1 = x_c + R * np.cos(a0 + phi)
            y1 = y_c + R * np.sin(a0 + phi)
            spokes[i].set_data([x_c, x1], [y_c, y1])

        return [wheel, contact_dot, time_text] + spokes

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000 / ANIMATION_FPS,
        blit=True,
    )
    
    ani.save(filename, writer="pillow", fps=ANIMATION_FPS)
    plt.close(fig)


real_title = f"Качение без проскальзывания (реальное время): {body_name}"
slow_title = f"Качение без проскальзывания (замедленно): {body_name}"

fname_real = f"rolling_{BODY_TYPE}_real_time.gif"
fname_slow = f"rolling_{BODY_TYPE}_slow_motion.gif"

create_animation(idx_frames_real, fname_real, real_title)
create_animation(idx_frames_slow, fname_slow, slow_title)

print(f"GIF (реальное время): {fname_real}")
print(f"GIF (замедленно): {fname_slow}")