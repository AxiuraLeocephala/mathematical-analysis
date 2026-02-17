# ЗАДАЧА ТРЁХ И БОЛЕЕ ТЕЛ

# Дана звёздно-планетная система с любым количеством звёзд и планет,
# задаются их параметры и начальные условия (подобраны несколько интересных конфигураций).
# Программа сначала вычисляет их движение под действием сил гравитации,
# а затем - анимацию происходящего с точки зрения наблюдателя, находящегося на одной из планет (на заданной широте).
# Реализовано вращение тел вокруг своей оси (но не реализовано влияние гравитации других тел на характер и скорость вращения).
# Для видимых планет вычисляется смена фаз, подобных фазам Луны, а также солнечные затмения.
# При этом звёзды обрабатываются как точечные источники света
# Анимация сохраняется в файл n-body-problem.gif в папке программы

# ЗАДАНИЯ ПО ДОРАБОТКЕ ПРОГРАММЫ

# 1) Разобраться в коде, его математических и физических основаниях
# 2) Добавить в программу несколько интересных конфигураций звёзд и планет.
# Использовать реальные физические единицы и физические константы вместо используемых сейчас условных
# 3) Сейчас в программе всё движение происходит в одной плоскости, а оси вращения тел перпендикулярны плоскости движения.
# Рассмотреть более общий случай - с произвольным движением и произвольными наклонами осей вращения тел
# 4) Добавить вычисление степени освещённости и температуры в разных точках поверхности планет
# согласно подходящей простейшей модели, добавить смену дня и ночи
# 5) Добавить графический интерфейс с интерактивной анимацией, с возможностью изменять направление взгляда клавиатурой или мышью
# Добавить возможность "поднимать и опускать голову", т.е. изменять направление взгляда по вертикали
# 6) Изучить и реализовать влияние гравитации со стороны других тел на вращение тела вокруг своей оси
# 7) При желании можно ещё обрабатывать звёзды как неточечные источники света, добавить лунные затмения

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from scipy.integrate import solve_ivp

G = 1.0  # гравитационная постоянная
CONFIGURATION = 1 # какую конфигурацию считаем из предложенных ниже, можно добавить свои
SIMULATION_TIME = 12.0 # "физическое" время симуляции
ANIMATION_FPS = 30 # количество кадров в секунду в GIF (плавность анимации)
PLAYBACK_SLOWDOWN = 4.0 # замедлить анимацию в данное количество раз

d_projection = 1.0 # расстояние проекционной плоскости от наблюдателя (влияет на координаты x,y в поле зрения наблюдателя)
fov_angle = 2.0 * np.pi / 3.0  # угол обзора наблюдателя. То что под горизонтом - не видно

# Качество рендера изображений планет
PLANET_RASTER_N = 160 # базовое разрешение текстуры
PLANET_SUPERSAMPLE = 2 # сглаживание: можно увеличить, если качество недостаточное

# Дана конфигурация тел - заданы их положения, начальные скорости, массы.
# Перейти в систему отсчёта центра масс - в ней конфигурация находится вокруг начала координат и никуда не сдвигается как целое
def center_and_zero_momentum(positions: np.ndarray, velocities: np.ndarray, masses: np.ndarray):

    M = np.sum(masses)
    com = np.sum(positions * masses[:, None], axis=0) / M
    positions = positions - com

    P = np.sum(velocities * masses[:, None], axis=0)
    v_cm = P / M
    velocities = velocities - v_cm

    return positions, velocities

# Предложить положения и скорости двух тел заданных масс на расстоянии separation друг от друга,
# так чтобы они двигались по круговой орбите вокруг общего центра масс
def make_circular_binary(m1: float, m2: float, separation: float):

    a = separation
    w = np.sqrt(G * (m1 + m2) / (a ** 3))

    r1 = a * m2 / (m1 + m2)
    r2 = a * m1 / (m1 + m2)

    pos1 = np.array([-r1, 0.0, 0.0])
    pos2 = np.array([+r2, 0.0, 0.0])

    vel1 = np.array([0.0, -w * r1, 0.0])
    vel2 = np.array([0.0, +w * r2, 0.0])

    return pos1, vel1, pos2, vel2

# Скорость тела малой массы, движущегося по круговой орбите заданного радиуса вокруг тела заданной массы
def circular_orbit_velocity(M_central: float, r: float) -> float:
    return np.sqrt(G * M_central / r)

# КОНФИГУРАЦИИ ЗВЁЗД И ПЛАНЕТ (можно добавить свои)
config_names = {
    1: 'Двойная звезда и Луна',
    2: 'Восьмёрка',
    3: 'Восьмёрка с плюсом'
}

# Конфигурация 1: Двойная звезда (тела 0 и 1), вокруг которой движется планета (тело 2) со спутником (тело 3)
if CONFIGURATION == 1:
    
    ROTATION_PERIOD = 6 # период вращения тела наблюдателя вокруг своей оси
    OBSERVER_LATITUDE = np.pi/4 # широта, на которой находится наблюдатель, от -np.pi/2 до np.pi/2; крайние значения
    # соответствуют полюсам, их лучше не выбирать - там не определено направление на север, используемое в программе
    OBSERVER_LONGITUDE = 0.0 # начальная долгота, на которой находится наблюдатель в самом начале (менее важный параметр)
    VIEWING_DIRECTION_ANGLE = np.pi / 2 # угол зрения наблюдателя: 0 - направление на север, np.pi/2 - на восток
    observer_body_idx = 2  # На каком теле находится наблюдатель
    
    m1, m2 = 90.0, 60.0
    mp = 1.4
    mm = 0.08

    masses = np.array([m1, m2, mp, mm], dtype=float) # массы тел
    physical_radii = np.array([0.35, 0.3, 0.1, 0.05], dtype=float) # радиусы тел
    is_star = [True, True, False, False] # какие тела звёзды, а какие планеты
    colors = ['gold', 'white', 'cornflowerblue', 'tan'] # цвета тел на изображении

    a_bin = 1.9
    pos1, vel1, pos2, vel2 = make_circular_binary(m1, m2, a_bin)

    a_p = 6.8
    v_p = circular_orbit_velocity(m1 + m2, a_p)
    pos_p = np.array([0.0, a_p, 0.0])
    vel_p = np.array([-v_p, 0.0, 0.0])

    a_pm = 0.42
    pos_p_rel, vel_p_rel, pos_m_rel, vel_m_rel = make_circular_binary(mp, mm, a_pm)

    pos_planet = pos_p + pos_p_rel
    vel_planet = vel_p + vel_p_rel

    pos_moon = pos_p + pos_m_rel
    vel_moon = vel_p + vel_m_rel

    positions0 = np.vstack([pos1, pos2, pos_planet, pos_moon]) # начальные положения тел
    velocities0 = np.vstack([vel1, vel2, vel_planet, vel_moon]) # начальные скорости тел

    positions0, velocities0 = center_and_zero_momentum(positions0, velocities0, masses) # корректировка положений и скоростей

# Конфигурация 2: фигура-восьмёрка (классическое решение задачи трёх тел). Одна звезда (тело 0) и две планеты (тела 1 и 2)
elif CONFIGURATION == 2:

    ROTATION_PERIOD = 2 # период вращения тела наблюдателя вокруг своей оси
    OBSERVER_LATITUDE = np.pi/6 # широта, на которой находится наблюдатель, от -np.pi/2 до np.pi/2; крайние значения
    # соответствуют полюсам, их лучше не выбирать - там не определено направление на север, используемое в программе
    OBSERVER_LONGITUDE = 0.0 # начальная долгота, на которой находится наблюдатель в самом начале (менее важный параметр)
    VIEWING_DIRECTION_ANGLE = np.pi / 2 # угол зрения наблюдателя: 0 - направление на север, np.pi/2 - на восток
    observer_body_idx = 2 # На каком теле находится наблюдатель
    
    masses = np.array([1.0, 1.0, 1.0], dtype=float) # массы тел
    physical_radii = np.array([0.08, 0.08, 0.08], dtype=float) # радиусы тел
    is_star = [True, False, False] # какие тела звёзды, а какие планеты
    colors = ['gold', 'deepskyblue', 'lightgray'] # цвета тел на изображении

    positions0 = np.array([
        [ 0.97000436, -0.24308753, 0.0],
        [-0.97000436,  0.24308753, 0.0],
        [ 0.0,         0.0,        0.0]
    ], dtype=float) # начальные положения тел

    velocities0 = np.array([
        [ 0.4662036850,  0.4323657300, 0.0],
        [ 0.4662036850,  0.4323657300, 0.0],
        [-0.9324073700, -0.8647314600, 0.0]
    ], dtype=float) # начальные скорости тел

    positions0, velocities0 = center_and_zero_momentum(positions0, velocities0, masses)

# Конфигурация 3: "восьмёрка" из двух звёзд (тела 0 и 1) и одной планеты (тело 2),
# а также ещё одна планета (тело 3), вращающаяся вокруг этой "восьмёрки" и дестабилизирующая её своей гравитацией
else:
    
    ROTATION_PERIOD = 2 # период вращения тела наблюдателя вокруг своей оси
    OBSERVER_LATITUDE = np.pi/6 # широта, на которой находится наблюдатель, от -np.pi/2 до np.pi/2; крайние значения
    # соответствуют полюсам, их лучше не выбирать - там не определено направление на север, используемое в программе
    OBSERVER_LONGITUDE = 0.0 # начальная долгота, на которой находится наблюдатель в самом начале (менее важный параметр)
    VIEWING_DIRECTION_ANGLE = np.pi / 2 # угол зрения наблюдателя: 0 - направление на север, np.pi/2 - на восток
    observer_body_idx = 3 # На каком теле находится наблюдатель
    
    m = 1.0
    m_probe = 0.05

    masses = np.array([m, m, m, m_probe], dtype=float) # массы тел
    physical_radii = np.array([0.12, 0.12, 0.12, 0.12], dtype=float) # радиусы тел
    is_star = [True, True, False, False]  # какие тела звёзды, а какие планеты
    colors = ['gold', 'orangered', 'deepskyblue', 'lightgray']

    positions0 = np.array([
        [ 0.97000436, -0.24308753, 0.0],
        [-0.97000436,  0.24308753, 0.0],
        [ 0.0,         0.0,        0.0],
        [ 2.4,         0.0,        0.0],
    ], dtype=float) # начальные положения тел

    velocities0 = np.array([
        [ 0.4662036850,  0.4323657300, 0.0],
        [ 0.4662036850,  0.4323657300, 0.0],
        [-0.9324073700, -0.8647314600, 0.0],
        [ 0.0,           1.25,          0.0],
    ], dtype=float) # начальные скорости тел

    positions0, velocities0 = center_and_zero_momentum(positions0, velocities0, masses) # корректировка положений и скоростей

# Начальное состояние системы в одном векторе
initial_state = np.zeros(6 * len(masses), dtype=float)
for i in range(len(masses)):
    initial_state[6*i:6*i+3] = positions0[i]
    initial_state[6*i+3:6*i+6] = velocities0[i]

# Функция, задающая систему дифференциальных уравнений для расчёта движения тел
# Вообще, это система, состоящая из дифференциальных уравнений второго порядка:
# для каждого тела с номером i свои три уравнения x_i''=a_{ix}, y_i''=a_{iy}, z''=a_{iz}
# где в левой части стоят вторые производные координат этого тела по времени, составляющие вектор ускорения
# (ускорение - это скорость изменения скорости изменения координат, поэтому вторая производная)
# а в правой части должны стоять эти же компоненты ускорения,
# рассчитанные по второму закону Ньютона и закону всемирного тяготения.
# Но для решения системы дифференциальных уравнений здесь будет использоваться процедура solve_ivp из библиотеки scipy
# а для неё нужно предварительно преобразовать систему, так чтобы в ней были только первые производные
# Это делается так: x_i'=v_{ix}, y_i'=v_{iy}, z_i'=v_{iz}, v_{ix}'=a_{ix}, v_{iy}'=a_{iy}, v_{iz}'=a_{iz}
# Функция n_body_rhs рассчитывает правые части этой системы, если состояния x_i, y_i, z_i, v_{ix}, v_{iy}, v_{iz}
# для каждого тела хранятся подряд в массиве state
# Затем, зная начальные состояния из массива initial_state и закон их изменения из функции n_body_rhs,
# можно будет рассчитать движение тел
def n_body_rhs(t, state, masses):

    n = len(masses)

    # Преобразуем в удобную форму (n строк, 6 столбцов)
    Y = state.reshape((n, 6))
    pos = Y[:, 0:3]  # координаты тел (n×3)
    vel = Y[:, 3:6]  # скорости тел (n×3)

    # Массив ускорений, который мы будем заполнять
    a = np.zeros((n, 3), dtype=float)

    for i in range(n):
        ax, ay, az = 0.0, 0.0, 0.0
        xi, yi, zi = pos[i]
        for j in range(n):
            if j == i:
                continue  # тело не действует само на себя
            xj, yj, zj = pos[j]
            # Вектор от тела i к телу j
            rx = xj - xi
            ry = yj - yi
            rz = zj - zi
            dist2 = rx*rx + ry*ry + rz*rz
            dist = np.sqrt(dist2) # расстояние между телами

            # Закон всемирного тяготения в векторной форме. Разберитесь, откуда здесь куб
            coef = G * masses[j] / (dist**3)
            ax += coef * rx
            ay += coef * ry
            az += coef * rz

        a[i, 0] = ax
        a[i, 1] = ay
        a[i, 2] = az

    dY = np.zeros_like(Y)
    dY[:, 0:3] = vel
    dY[:, 3:6] = a

    return dY.reshape(-1) # собираем все правые части в один вектор

# Общее число кадров. Интервал времени между кадрами будет SIMULATION_TIME/(frames-1).
frames = int(SIMULATION_TIME * ANIMATION_FPS * PLAYBACK_SLOWDOWN) + 1
frames = max(frames, 2)
t_frames = np.linspace(0.0, SIMULATION_TIME, frames) # моменты времени, соответствующие кадрам

sol = solve_ivp(
    fun=lambda t, y: n_body_rhs(t, y, masses),
    t_span=(0.0, SIMULATION_TIME),
    y0=initial_state,
    t_eval=t_frames,
    method='DOP853',
    rtol=1e-9,
    atol=1e-12,
) # Решить систему дифференциальных уравнений и рассчитать движение тел

if not sol.success:
    raise RuntimeError(f"Интегратор не справился: {sol.message}")

solution = sol.y.T  # решение в виде массива frames x 6n, где n - количество тел
# для каждого момента времени найдено состояние каждого из n тел, т.е. координаты и компоненты скорости
# T означает транспонирование, без него был бы массив 6n x frames

# Найти местоположение наблюдателя и основные направления его системы координат по следующим данным
# body_pos - координаты центра планеты наблюдателя, time - момент времени, latitude - широта, longitude0 - начальная долгота
def get_observer_system(body_pos, time, observer_idx, latitude, longitude0):

    angular_velocity = 2.0 * np.pi / ROTATION_PERIOD # угловая скорость планеты наблюдателя
    lon = longitude0 + angular_velocity * time # долгота в данный момент времени (с учётом вращения планеты)
    Rb = physical_radii[observer_idx] # радиус планеты

    cos_lat = np.cos(latitude)
    sin_lat = np.sin(latitude)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    up_dir = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]) # направление вверх
    east_dir = np.array([-sin_lon, cos_lon, 0.0]) # направление на восток, в сторону вращения планеты
    north_dir = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat]) # направление на север

    observer_pos = body_pos + Rb * up_dir # координаты наблюдателя

    # Направление взгляда наблюдателя: от севера поворачиваем к востоку на угол VIEWING_DIRECTION_ANGLE
    forward_dir = np.cos(VIEWING_DIRECTION_ANGLE) * north_dir + np.sin(VIEWING_DIRECTION_ANGLE) * east_dir
    right_dir = np.cross(forward_dir, up_dir) # третий вектор ортонормированного базиса - векторное произведение двух других
    # Теперь right_dir, forward_dir, up_dir образуют правый ортонормированный базис и определяют оси x,y,z в системе наблюдателя

    return observer_pos, up_dir, right_dir, forward_dir, north_dir

# Возвращает координаты заданной точки в системе координат наблюдателя, и расстояние до этой точки
def camera_coordinates(point_pos, obs_pos, right_dir, up_dir, forward_dir):
    rel = point_pos - obs_pos
    x_local = float(np.dot(rel, right_dir))
    y_local = float(np.dot(rel, up_dir))
    z_local = float(np.dot(rel, forward_dir))
    dist = float(np.linalg.norm(rel))
    return x_local, y_local, z_local, dist

# Спроектировать сферу с заданными центром и радиусом на плоскость зрения z=d в системе координат наблюдателя
# Используем центральную проекцию: в точку (x,y,d) проектируются все точки вида (kx,ky,kd) с положительными k:
# все эти точки находятся в одной и той же стороне от наблюдателя, поэтому проектируются в одну точку в его поле зрения
# Таким образом, точки (x,y,d) будут проекциями точек сферы, если (kx,ky,kd) лежат на сфере
# Чтобы понять, для каких (x,y,d) это так (они и составляют проекцию), подставим (kx,ky,kd) в уравнение сферы
# Получится квадратное уравнение, множество подходящих (x,y) определяется условием неотрицательности дискриминанта
# Ограничивающая это множество линия находится из равенства нулю дискриминанта
# Это эллипс. Шарообразные тела будут выглядеть круговыми, если наблюдатель смотрит на них,
# но будут выглядеть эллиптическими, если они находятся вдали от центра поля зрения
def project_sphere(x0, y0, z0, R, d):

    if z0 <= 1e-9: # если сфера находится сзади и поэтому не видна
        return None
    if (x0 * x0 + y0 * y0 + z0 * z0) <= (R * R + 1e-12): # наблюдатель находится внутри сферы - не рисуем такое
        return None

    # Уравнение эллипса на плоскости z=d: A x^2 + B xy + C y^2 + D x + E y + F = 0
    # Коэффициенты выведены из дискриминанта квадратного уравнения пересечения луча и сферы
    A = R * R - y0 * y0 - z0 * z0
    B = 2.0 * x0 * y0
    C = R * R - x0 * x0 - z0 * z0
    D = 2.0 * d * x0 * z0
    E = 2.0 * d * y0 * z0
    F = d * d * (R * R - x0 * x0 - y0 * y0)

    # Нахождение параметров эллипса по его уравнению в общем виде
    
    M = np.array([[A, B / 2.0], [B / 2.0, C]], dtype=float)
    v = np.array([D / 2.0, E / 2.0], dtype=float)
    det = np.linalg.det(M)
    if abs(det) < 1e-14:
        return None

    center = -np.linalg.solve(M, v)
    cx, cy = float(center[0]), float(center[1]) # координаты центра эллипса

    Fc = F + D * cx + E * cy + A * cx * cx + B * cx * cy + C * cy * cy # подставляем их в уравнение эллипса

    # Собственные значения и собственные векторы
    eigvals, eigvecs = np.linalg.eigh(M)

    # Для эллипса нужно, чтобы -Fc/eigvals > 0
    # Не выдавать ошибку, если вдруг деление на нуль
    with np.errstate(divide='ignore', invalid='ignore'):
        axes2 = -Fc / eigvals

    if not np.all(np.isfinite(axes2)):
        return None
    if np.any(axes2 <= 0.0):
        return None

    a1 = float(np.sqrt(axes2[0]))
    b1 = float(np.sqrt(axes2[1])) # полуоси эллипса

    # Угол наклона первой оси - направление eigenvector[:,0]
    theta1 = float(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # Сделаем так, чтобы a >= b (для удобства)
    if b1 > a1:
        a, b = b1, a1
        theta = float(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    else:
        a, b = a1, b1
        theta = theta1

    ct = np.cos(theta)
    st = np.sin(theta)
    x_ext = float(np.sqrt((a * ct) ** 2 + (b * st) ** 2))
    y_ext = float(np.sqrt((a * st) ** 2 + (b * ct) ** 2)) # Половины сторон прямоугольника, ограничивающего эллипс

    return cx, cy, a, b, theta, x_ext, y_ext

# Границы поля зрения наблюдателя: от -max_half_width до max_half_width по горизонтали,
# от 0 до max_height по вертикали (потому что под горизонтом ничего не видно)
def calculate_field_of_view_limits(d, fov_angle):
    max_half_width = d * np.tan(fov_angle / 2.0)
    max_height = d * np.tan(fov_angle / 2.0)
    return max_half_width, max_height

# В поле зрения наблюдателя, находящегося в точке obs_pos, сфера с центром в точке point_pos и радиусом physical_radius
# выглядит как эллипс (если сфера видима и не под линией горизонта). Здесь ищутся параметры этого эллипса
# а также глубина (depth) для сортировки изображений, если они перекрывают друг друга - какое спереди, какие сзади
def central_projection(point_pos, obs_pos, up_dir, right_dir, forward_dir, d, fov_angle, physical_radius):

    x_local, y_local, z_local, distance = camera_coordinates(
        point_pos=point_pos,
        obs_pos=obs_pos,
        right_dir=right_dir,
        up_dir=up_dir,
        forward_dir=forward_dir,
    )

    # если тело позади наблюдателя
    if z_local <= 1e-6:
        return None, None, None

    R = float(physical_radius)

    alpha = fov_angle / 2.0 # половина угла обзора

    ang_x = float(np.arctan2(abs(x_local), z_local))
    ang_y = float(np.arctan2(abs(y_local), z_local))
    phi_max = max(ang_x, ang_y)

    ratio = np.clip(R / distance, 0.0, 0.999999) # угол, под которым виден радиус тела
    gamma = float(np.arcsin(ratio))

    # Если сфера целиком вне угла обзора (по x или по y), не рисуем
    if (ang_x - gamma) > alpha or (ang_y - gamma) > alpha:
        return None, None, None

    # Центр проекции (проекция центра сферы)
    cx = (d / z_local) * x_local
    cy = (d / z_local) * y_local

    # половина широты и высота кадра (если бы под горизонтом была видимая область, то это была бы половина высоты всего кадра)
    max_half_width = d * np.tan(fov_angle / 2.0)
    max_height = d * np.tan(fov_angle / 2.0)

    ell = project_sphere(x_local, y_local, z_local, R, float(d))

    # Упрощенный способ рисования тела - круговое изображение вместо эллиптического, если эллипс почему-то не посчитался
    r_fallback = float(d * np.tan(gamma) / max(1e-6, np.cos(phi_max))) # радиус кругового изображения
    if ell is None:
        a = b = r_fallback
        theta = 0.0
        x_ext = y_ext = r_fallback
    else:
        cx, cy, a, b, theta, x_ext, y_ext = ell

    # Отсечение по прямоугольнику поля зрения
    if (cx + x_ext < -max_half_width) or (cx - x_ext > max_half_width):
        return None, None, None

    # Горизонт: ниже y=0 ничего не видно
    if (cy + y_ext) < 0.0:
        return None, None, None

    if (cy - y_ext) > max_height:
        return None, None, None

    # Если несколько тел находятся в одной стороне от наблюдателя, вначале надо рисовать те что сзади, а потом те что спереди
    # Тела сортируются по величине depth = z_local - R - это координата ближайшей к наблюдателю точки тела вдоль оси его взгляда
    depth = z_local - R
    return (cx, cy, a, b, theta, x_ext, y_ext), distance, depth

# Рендер изображений планет с учётом фаз - возвращает RGBA-текстуру для освещённой части планеты
# на участке плоскости [x0,x1]×[y0,y1]. Здесь N - базовое разрешение текстуры, supersample - сглаживание
# RGBA-текстура - трёхмерный массив, каждому пикселю из двумерной сетки сопоставлен цвет - три компоненты RGB и прозрачность A
# Для каждого пикселя на плоскости зрения наблюдателя, проводим луч до пересечения с рассматриваемой планетой
# и вычисляем, освещена ли эта точка на поверхности планеты хотя бы одной звездой (и значит - увидит ли её наблюдатель)
# Можно усовершенствовать эту функцию, чтобы учитывалась различная освещённость разных точек поверхности
def render_planet(
    planet_center: np.ndarray,
    planet_radius: float,
    star_positions: list,
    base_color,
    obs_pos: np.ndarray,
    right_dir: np.ndarray,
    up_dir: np.ndarray,
    forward_dir: np.ndarray,
    d: float,
    x0: float, x1: float, y0: float, y1: float,
    N: int = 160,
    supersample: int = 2,
):

    if len(star_positions) == 0: # нет звёзд - значит ничего не видно
        return None

    ss = max(1, int(supersample))
    M = int(N) * ss

    xs = np.linspace(x0, x1, M)
    ys = np.linspace(y0, y1, M)
    X, Y = np.meshgrid(xs, ys) # сетка пикселей MxM
   
    # dir_vec - массив MxMx3 направлений лучей во все пиксели сетки. Каждому пикселю соответствует
    # направление (xs, ys, d) в системе координат наблюдателя, заданное вектором xs*right_dir+ys*up_dir+d*forward_dir
    # forward_dir[None, None, :] - делает из трёхмерного вектора forward_dir массив 1x1x3,
    # чтобы его можно было быстро и удобно умножать на другие трёхмерные массивы
    dir_vec = (
        forward_dir[None, None, :] * d
        + right_dir[None, None, :] * X[:, :, None]
        + up_dir[None, None, :] * Y[:, :, None]
    )
    
    dir_norm = np.linalg.norm(dir_vec, axis=2) # длины векторов из массива dir_vec;
    # axis=2 (для трёхмерных массивов возможные значения 0,1,2) - значит, векторы в массиве MxMx3 располагаются по третьей оси
    dir_hat = dir_vec / dir_norm[:, :, None] # нормировать векторы направлений (сделать их единичной длины)

    # Ищем пересечение каждого луча со сферой. Для этого переходим из системы координат наблюдателя в исходную систему координат
    # В ней уравнение луча r(t) = obs_pos + t*dir_hat, t>0, где obs_pos - положение наблюдателя
    # Это значит, что при подставлении всевозможных t>0 будем получать радиус-векторы r(t) (=наборы координат) точек луча
    # Подставляем в уравнение сферы |r(t) - C|^2 = R^2, где C - центр сферы; здесь модуль - это модуль вектора
    # Получаем квадратное уравнение, находим ближайшую из двух точек пересечения со сферой
    oc = obs_pos - planet_center
    c = np.dot(oc, oc) - planet_radius * planet_radius
    b = (dir_hat[:, :, 0] * oc[0] + dir_hat[:, :, 1] * oc[1] + dir_hat[:, :, 2] * oc[2])
    disc = b * b - c # дискриминант (точнее, массив MxM дискриминантов)
    hit = disc >= 0.0 # нас интересуют только те пиксели, где дискриминант неотрицателен и пересечение со сферой есть
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    t = -b - sqrt_disc
    hit &= t > 0.0 # нас интересуют только те пиксели, где наименьший корень положителен (пересечения "вперёд по лучу")

    if not np.any(hit):
        return None

    # Массив точек пересечения (вычисляем везде, но использовать будем только там, где hit=true)
    P = obs_pos[None, None, :] + t[:, :, None] * dir_hat

    # Массив нормалей к поверхности планеты в точках пересечения
    Nsurf = (P - planet_center[None, None, :]) / planet_radius

    # Строим массив из пикселей, соответствующих освещённым точкам планеты
    lit = np.zeros((M, M), dtype=bool)
    for S in star_positions: # для каждой звезды S
        # Вектор на звезду из точки на поверхности
        L = S[None, None, :] - P
        dots = np.sum(Nsurf * L, axis=2) # Скалярное произведение на вектор нормали - положительно, если точка освещена звездой
        lit |= (dots > 0.0)

    lit &= hit

    alpha = 1 # прозрачность, у нас изображение полностью непрозрачно
    alpha_hi = (alpha * lit.astype(np.float32)) # alpha=0 там, где ничего не видно

    if ss > 1:
        alpha_ds = alpha_hi.reshape(N, ss, N, ss).mean(axis=(1, 3))
        # Превращаем массив MxM в N x ss x N x ss и усредняем по мелким клеткам
    else:
        alpha_ds = alpha_hi

    r, g, bcol, _ = to_rgba(base_color)
    rgba = np.zeros((N, N, 4), dtype=np.float32)
    rgba[..., 3] = alpha_ds

    # Цвет задаём только там, где альфа положительна, чтобы вне диска планеты пиксели были точно чёрными
    vis = alpha_ds > 1e-6
    rgba[vis, 0] = r
    rgba[vis, 1] = g
    rgba[vis, 2] = bcol

    return rgba

max_half_width, max_height = calculate_field_of_view_limits(d_projection, fov_angle)

# Рисование графиков
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(24, 10), facecolor='black')
_title = ("Задача n тел: вид наблюдателя (слева) и вид сверху (справа)")
fig.suptitle(_title, color='white', fontsize=16, y=0.98)

# Левый график (поле зрения)
ax_left.set_facecolor('black')
ax_left.set_xlim(-max_half_width, max_half_width)
ax_left.set_ylim(0.0, max_height)
ax_left.set_aspect('equal', adjustable='box')
ax_left.set_title('Поле зрения наблюдателя (центральная проекция)', color='white', fontsize=14)
ax_left.tick_params(colors='white')
for spine in ax_left.spines.values():
    spine.set_color('white')

# Линия горизонта - нижняя граница изображения
horizon_line_left, = ax_left.plot([], [], color='white', lw=1, alpha=0.8)

# Правый график (вид сверху)
ax_right.set_facecolor('black')
ax_right.set_aspect('equal', adjustable='box')
ax_right.set_title('Вид сверху', color='white', fontsize=14)
ax_right.tick_params(colors='white')
for spine in ax_right.spines.values():
    spine.set_color('white')

pos0 = positions0
max_r = np.max(np.linalg.norm(pos0[:, :2], axis=1)) # автоматический подбор масштаба для вида сверху
R_view = max(3.0, 1.6 * max_r)
ax_right.set_xlim(-R_view, R_view)
ax_right.set_ylim(-R_view, R_view)

trajectories_right = []
celestial_circles_right = []
trajectory_data_right = [[] for _ in range(len(masses))]

for i in range(len(masses)):
    (line,) = ax_right.plot([], [], lw=1.2, color=colors[i], alpha=0.75)
    trajectories_right.append(line)

    circle = patches.Circle((0, 0), physical_radii[i], facecolor=colors[i], edgecolor=None, alpha=0.9)
    ax_right.add_patch(circle)
    celestial_circles_right.append(circle)

# Маркер центра масс
com_marker_right, = ax_right.plot([], [], marker='x', color='white', markersize=8, alpha=0.7)

# Стрелка направления взгляда
view_direction_arrow = ax_right.quiver(0, 0, 0, 0, color='cyan', angles='xy', scale_units='xy', scale=1.0, width=0.004)

# Построение анимации

artists_left_dynamic = []

def init_animation():
    horizon_line_left.set_data([-max_half_width * 1.2, max_half_width * 1.2], [0.0, 0.0])

    for line in trajectories_right:
        line.set_data([], [])

    for circ in celestial_circles_right:
        circ.center = (1e9, 1e9)

    com_marker_right.set_data([], [])

    view_direction_arrow.set_offsets([0.0, 0.0])
    view_direction_arrow.set_UVC(0.0, 0.0)

    return [horizon_line_left] + trajectories_right + celestial_circles_right + [com_marker_right, view_direction_arrow]


def animate(frame_idx: int): # один кадр анимации

    global artists_left_dynamic

    state = solution[frame_idx]
    n = len(masses)

    Y = state.reshape((n, 6))
    positions = Y[:, :3]

    body_pos = positions[observer_body_idx]
    t = t_frames[frame_idx]

    obs_pos, up_dir, right_dir, forward_dir, north_dir = get_observer_system(
        body_pos=body_pos,
        time=t,
        observer_idx=observer_body_idx,
        latitude=OBSERVER_LATITUDE,
        longitude0=OBSERVER_LONGITUDE,
    )

    # правый график: траектории и тела
    max_history = 420
    for i in range(n):
        trajectory_data_right[i].append(positions[i].copy())
        if len(trajectory_data_right[i]) > max_history:
            trajectory_data_right[i].pop(0)

        tr = np.array(trajectory_data_right[i])
        trajectories_right[i].set_data(tr[:, 0], tr[:, 1])
        celestial_circles_right[i].center = (positions[i, 0], positions[i, 1])

    M = np.sum(masses)
    com = np.sum(positions * masses[:, None], axis=0) / M
    com_marker_right.set_data([com[0]], [com[1]])

    f_xy = np.array([forward_dir[0], forward_dir[1]])
    view_direction_arrow.set_offsets([obs_pos[0], obs_pos[1]])
    view_direction_arrow.set_UVC(f_xy[0], f_xy[1])

    # левый график: очищаем динамику прошлого кадра
    for art in artists_left_dynamic:
        try:
            art.remove()
        except Exception:
            pass
    artists_left_dynamic = []

    star_positions = [positions[i] for i in range(n) if is_star[i]]

    # сортировка изображений по дальности (сначала рисуем дальние)
    items = []
    for i in range(n):
        if i == observer_body_idx:
            continue

        proj = central_projection(
            point_pos=positions[i],
            obs_pos=obs_pos,
            up_dir=up_dir,
            right_dir=right_dir,
            forward_dir=forward_dir,
            d=d_projection,
            fov_angle=fov_angle,
            physical_radius=physical_radii[i],
        )

        if proj[0] is None:
            continue

        ell_params, dist, depth = proj
        items.append((depth, dist, i, ell_params))

    items.sort(key=lambda x: x[0], reverse=True)

    for k, (depth, dist, i, ell_params) in enumerate(items):
        cx, cy, a, b, theta, x_ext, y_ext = ell_params

        # zorder делаем зависящим от порядка (дальние -> ближние)
        zbase = 100 + 10 * k
        if is_star[i]:
            # Изображение звезды - просто эллипс с найденными параметрами
            star_ell = patches.Ellipse(
                (cx, cy),
                width=2.0 * a,
                height=2.0 * b,
                angle=np.degrees(theta),
                facecolor=colors[i],
                edgecolor=None,
                alpha=1,
                zorder=zbase,
            )
            ax_left.add_patch(star_ell)
            artists_left_dynamic.append(star_ell)
        else:
            # Изображение планеты: сначала рисуем полный силуэт планеты чёрным, затем поверх него рисуем освещённую часть
            # Это позволяет правильно изображать "солнечные затмения"
            dark_disk = patches.Ellipse(
                (cx, cy),
                width=2.0 * a,
                height=2.0 * b,
                angle=np.degrees(theta),
                facecolor='black',
                edgecolor=None,
                alpha=1.0,
                zorder=zbase,
            )
            ax_left.add_patch(dark_disk)
            artists_left_dynamic.append(dark_disk)

            # Используем ограничивающий прямоугольник эллипса
            x0 = cx - x_ext
            x1 = cx + x_ext
            y0 = cy - y_ext
            y1 = cy + y_ext

            # Ниже горизонта (y<0) ничего не рисуем
            y0_render = max(y0, 0.0)
            if y1 <= 0.0 or y0_render >= y1:
                continue

            rgba = render_planet(
                planet_center=positions[i],
                planet_radius=float(physical_radii[i]),
                star_positions=star_positions,
                base_color=colors[i],
                obs_pos=obs_pos,
                right_dir=right_dir,
                up_dir=up_dir,
                forward_dir=forward_dir,
                d=d_projection,
                x0=x0, x1=x1, y0=y0_render, y1=y1,
                N=PLANET_RASTER_N,
                supersample=PLANET_SUPERSAMPLE,
            )

            if rgba is not None:
                # Рисуем построенную текстуру
                im = ax_left.imshow(
                    rgba,
                    extent=(x0, x1, y0_render, y1),
                    origin='lower',
                    interpolation='nearest',  # сглаживание делаем supersampling'ом, без доп. интерполяции
                    zorder=zbase + 1,
                )

                # Обрезаем текстуру эллипсом на всякий случай
                clip_shape = patches.Ellipse(
                    (cx, cy),
                    width=2.0 * a,
                    height=2.0 * b,
                    angle=np.degrees(theta),
                    transform=ax_left.transData,
                )
                im.set_clip_path(clip_shape)

                artists_left_dynamic.append(im)

    return [horizon_line_left] + artists_left_dynamic + trajectories_right + celestial_circles_right + [com_marker_right, view_direction_arrow]


anim = FuncAnimation(
    fig,
    animate,
    init_func=init_animation,
    frames=len(t_frames),
    interval=1000 / ANIMATION_FPS,
    blit=False,
)

out_name = 'n-body-problem.gif'
anim.save(out_name, writer="pillow", fps=ANIMATION_FPS)

print(f" Анимация сохранена: {out_name}")

plt.close('all')