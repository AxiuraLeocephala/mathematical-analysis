# КОЛЕБАНИЯ СТРУНЫ

# Моделируются колебания струны с закреплёнными концами, выведенной из положения равновесия и отпущенной без начальной скорости
# Используется метод разделения переменных, решение разлагается в ряд по гармоническим колебаниям

# ЗАДАНИЯ ПО ДОРАБОТКЕ

# 1) Разобраться в коде, его математических и физических основаниях. Поэкспериментировать с различными начальными формами струны
# 2) Реализовать случай ненулевой начальной скорости струны, а также вынужденных колебаний под действием внешней силы
# 3) Смоделировать звук, который вызывает струна, с помощью подходящей физической модели
# 4) Перейти от одномерной струны к двумерному барабану

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.integrate import quad

# Параметры задачи
L = 1.0 # Длина струны
c = 1.0 # Скорость распространения волн
T = 6 # Физическое время анимации
ANIMATION_FPS = 15 # Кадры в секунду
frames = int(T * ANIMATION_FPS) # Количество кадров
N_terms = 12 # Количество членов ряда Фурье
N_x = 700 # Разрешение по пространству
x = np.linspace(0, L, N_x)

# Начальная форма струны - функция должна принимать нулевые значения на концах струны, в точках 0 и L (концы закреплены)
def initial_shape(x):
    return x*x*(L-x)

# Вычисление коэффициентов Фурье A_n
def compute_fourier_coefficients():
    A_n = np.zeros(N_terms)
    for n in range(1, N_terms + 1):
        def integrand(x_val, n=n):
            return initial_shape(x_val) * np.sin(n * np.pi * x_val / L)    
        result, _ = quad(integrand, 0, L, limit=200)
        A_n[n-1] = (2/L) * result
    return A_n

A_n = compute_fourier_coefficients()

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, L)
ax.set_ylim(-0.2, 0.2)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', alpha=0.2, linewidth=0.8)
ax.axvline(x=0, color='k', alpha=0.5, linewidth=1.5)
ax.axvline(x=L, color='k', alpha=0.5, linewidth=1.5)

# Устанавливаем одинаковый масштаб по осям (можно отключить)
ax.set_aspect('equal')

# Создаем линии для визуализации
lines = []
colors = plt.cm.viridis(np.linspace(0, 1, N_terms))

# Тонкие линии для отдельных гармоник
for i in range(N_terms):
    line, = ax.plot([], [], color=colors[i], alpha=0.5, linewidth=0.9)
    lines.append(line)

# Жирная линия для полного решения
total_line, = ax.plot([], [], 'k-', linewidth=4.0)
lines.append(total_line)

# Функция для вычисления отклонения струны в момент времени t
def string_position(x, t):
    u = np.zeros_like(x)
    for n in range(1, N_terms + 1):
        omega_n = n * np.pi * c / L
        u += A_n[n-1] * np.cos(omega_n * t) * np.sin(n * np.pi * x / L)
    return u

# Функция для вычисления отдельной гармоники
def harmonic_position(x, t, n):
    omega_n = n * np.pi * c / L
    return A_n[n-1] * np.cos(omega_n * t) * np.sin(n * np.pi * x / L)

# Функция обновления кадра
def update(frame):
    t = frame / ANIMATION_FPS  # Текущее время
    
    # Обновляем отдельные гармоники
    for i in range(N_terms):
        y_harmonic = harmonic_position(x, t, i+1)
        lines[i].set_data(x, y_harmonic)
    
    # Обновляем полное решение
    y_total = string_position(x, t)
    lines[N_terms].set_data(x, y_total)
    
    # Обновляем заголовок с текущим временем
    ax.set_title(f'Колебания струны: t = {t:.3f} с', fontsize=14)
    
    return lines

# Создаём анимацию
animation = FuncAnimation(fig, update, frames=frames, interval=1000 / ANIMATION_FPS, blit=True, )

# Сохраняем анимацию в GIF файл
gif_path="string_vibrations.gif"
animation.save(gif_path, writer="pillow", fps=ANIMATION_FPS)
print(f"Анимация сохранена: {gif_path}")

plt.close(fig)