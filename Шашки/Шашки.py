# ИГРА В ШАШКИ С ИИ

# Запрограммированы два варианта искусственного интеллекта, умеющего играть в шашки
# Это классический алгоритм альфа-бета отсечения и простейший вариант метода Монте-Карло
# Реализован графический интерфейс, позволяющий пользователю играть в шашки с ИИ
# Силу искусственного интеллекта можно настраивать в программе
# (осторожно: если поставить слишком большие значения, программа зависнет)

# ЗАДАНИЯ ПО ДОРАБОТКЕ

# 1) Разобраться в коде
# 2) Сделать более сложный искусственный интеллект (например, алгоритм Monte Carlo Tree Search,
# лежащий в основе большинства современных подходов), и изучить его математические основы
# 3) Cделать такую же программу для шахмат или каких-нибудь других настольных игр
# 4) Добавить режим "без ИИ", позволяющий пользователям играть друг с другом
# 5) Реализовать игры искусственных интеллектов друг с другом и сохранение партий
# 6) Создать и обучить нейросеть выбору оптимального хода

import tkinter as tk # для графического интерфейса
from tkinter import messagebox
import random
import copy

BOARD_SIZE = 8
PLAYER_1_COLOR = 'white'
PLAYER_2_COLOR = 'black'
KING_SUFFIX = '_king'
CELL_SIZE = 60

class CheckersGame:

    # Действия при запуске программы: создание графического интерфейса и запуск игры
    def __init__(self, root):

        self.root = root
        self.root.title("Шашки")

        self.main_container = tk.Frame(root)
        self.main_container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.main_container, width=BOARD_SIZE * CELL_SIZE, height=BOARD_SIZE * CELL_SIZE)
        self.canvas.pack(pady=10)

        self.controls_frame = tk.Frame(self.main_container, pady=5)
        self.controls_frame.pack(side="bottom", fill="x")

        self.status_text = tk.StringVar()
        self.ai_mode = tk.IntVar(value=1)

        self.depth_var = tk.StringVar(value="3")
        self.sims_var = tk.StringVar(value="25")

        self.after_id = None

        self.setup_ui()
        self.start_new_game()

    # Создание элементов управления
    def setup_ui(self):

        lbl_status = tk.Label(self.controls_frame, textvariable=self.status_text, font=("Arial", 12, "bold"), fg="#333")
        lbl_status.pack(pady=(0, 10))

        algo_container = tk.Frame(self.controls_frame)
        algo_container.pack(pady=5)

        ab_frame = tk.LabelFrame(algo_container, text="Алгоритм Alpha-Beta", padx=10, pady=5)
        ab_frame.grid(row=0, column=0, padx=10, sticky="n")

        tk.Radiobutton(ab_frame, text="Использовать Alpha-Beta", variable=self.ai_mode, value=1).pack(anchor="w")

        param_ab = tk.Frame(ab_frame)
        param_ab.pack(fill="x", pady=5)
        tk.Label(param_ab, text="Глубина (полуходов):").pack(side="left")
        ent_depth = tk.Entry(param_ab, textvariable=self.depth_var, width=5)
        ent_depth.pack(side="left", padx=5)

        mc_frame = tk.LabelFrame(algo_container, text="Метод Monte Carlo", padx=10, pady=5)
        mc_frame.grid(row=0, column=1, padx=10, sticky="n")

        tk.Radiobutton(mc_frame, text="Использовать Monte Carlo", variable=self.ai_mode, value=2).pack(anchor="w")

        param_mc = tk.Frame(mc_frame)
        param_mc.pack(fill="x", pady=5)
        tk.Label(param_mc, text="Симуляций:").pack(side="left")
        ent_sims = tk.Entry(param_mc, textvariable=self.sims_var, width=5)
        ent_sims.pack(side="left", padx=5)

        btn_reset = tk.Button(
            self.controls_frame,
            text="НОВАЯ ИГРА",
            command=self.start_new_game,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=20,
        )
        btn_reset.pack(pady=15)

        self.canvas.bind("<Button-1>", self.on_click) # при щелчке на кнопке будет вызываться on_click

    # Читает целое число из счётчика. Если введено что-то не то, берётся значение по умолчанию
    def get_safe_int(self, var, default):
        try:
            val = int(var.get())
            return max(1, val)
        except ValueError:
            return default

    # Начать новую игру, случайно выбрать цвета для пользователя и ИИ
    def start_new_game(self):

        if self.after_id:
            self.root.after_cancel(self.after_id)
            self.after_id = None

        self.human_color = random.choice([PLAYER_1_COLOR, PLAYER_2_COLOR])
        self.ai_color = PLAYER_2_COLOR if self.human_color == PLAYER_1_COLOR else PLAYER_1_COLOR

        ru_color = "ЧЁРНЫЕ" if self.human_color == 'black' else "БЕЛЫЕ"
        self.status_text.set(f"Вы: {ru_color}")

        self.board = self.init_board()
        self.selected_piece = None
        self.possible_moves = []
        self.turn = PLAYER_1_COLOR
        self.must_continue_jump = False

        self.draw_board()

        if self.ai_color == PLAYER_1_COLOR:
            self.after_id = self.root.after(1000, self.ai_move)

    # Создать матрицу 8x8 со стартовой расстановкой шашек.
    # На пустых клетках хранится пустая строка, на занятых - строка 'black' или 'white'
    # с суффиксом '_king' если это дамка
    def init_board(self):
        board = [['' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r + c) % 2 != 0:
                    if r < 3:
                        board[r][c] = PLAYER_2_COLOR
                    elif r > 4:
                        board[r][c] = PLAYER_1_COLOR
        return board

    # Возвращает список всех допустимых ходов (slides). Если есть возможность взятия, то допустимы только взятия (jumps)
    def get_all_moves(self, board, player_color):
        all_jumps = []
        all_slides = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = board[r][c]
                if piece.startswith(player_color): # например "black" и "black_king" начинаются (startswith) с "black"
                    jumps = self.find_jump_sequences(board, r, c)
                    if jumps:
                        all_jumps.extend(jumps)
                    elif not all_jumps:
                        all_slides.extend(self.find_slides(board, r, c))
        return all_jumps if all_jumps else all_slides

    # Ищет все ходы без взятия для шашки (или дамки) на поле (r,c) в игровой позиции board
    def find_slides(self, board, r, c):
        slides = []
        piece = board[r][c]
        is_king = piece.endswith(KING_SUFFIX)
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        move_dir = -1 if piece.startswith(PLAYER_1_COLOR) else 1
        for dr, dc in directions:
            if not is_king and dr != move_dir:
                continue
            nr, nc = r + dr, c + dc
            if is_king:
                while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == '':
                    slides.append([(r, c, nr, nc, None)])
                    nr += dr
                    nc += dc
            else:
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == '':
                    slides.append([(r, c, nr, nc, None)])
        return slides

    # Строит все цепочки взятий для шашки (или дамки) на поле (r,c) в игровой позиции board
    def find_jump_sequences(self, board, r, c, current_sequence=None):
        if current_sequence is None:
            current_sequence = []
        piece = board[r][c]
        if not piece:
            return []
        is_king = piece.endswith(KING_SUFFIX)
        color = PLAYER_1_COLOR if piece.startswith(PLAYER_1_COLOR) else PLAYER_2_COLOR
        opp = PLAYER_2_COLOR if color == PLAYER_1_COLOR else PLAYER_1_COLOR
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        all_chains = []
        for dr, dc in directions:
            if is_king:
                for dist in range(1, BOARD_SIZE):
                    nr, nc = r + dr * dist, c + dc * dist
                    if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                        break
                    if board[nr][nc].startswith(color):
                        break
                    if board[nr][nc].startswith(opp):
                        jr, jc = nr + dr, nc + dc
                        if 0 <= jr < BOARD_SIZE and 0 <= jc < BOARD_SIZE and board[jr][jc] == '':
                            temp_board = copy.deepcopy(board)
                            temp_board[jr][jc] = piece
                            temp_board[r][c] = ''
                            temp_board[nr][nc] = ''
                            step = (r, c, jr, jc, (nr, nc))
                            sub_chains = self.find_jump_sequences(temp_board, jr, jc, current_sequence + [step])
                            if sub_chains:
                                all_chains.extend(sub_chains)
                            else:
                                all_chains.append(current_sequence + [step])
                        break
                    if board[nr][nc] != '':
                        break
            else:
                nr, nc = r + dr, c + dc
                jr, jc = r + 2 * dr, c + 2 * dc
                if 0 <= jr < BOARD_SIZE and 0 <= jc < BOARD_SIZE:
                    if board[nr][nc].startswith(opp) and board[jr][jc] == '':
                        temp_board = copy.deepcopy(board)
                        final_piece = piece
                        if (color == PLAYER_1_COLOR and jr == 0) or (color == PLAYER_2_COLOR and jr == BOARD_SIZE - 1):
                            final_piece = color + KING_SUFFIX
                        temp_board[jr][jc] = final_piece
                        temp_board[r][c] = ''
                        temp_board[nr][nc] = ''
                        step = (r, c, jr, jc, (nr, nc))
                        sub_chains = self.find_jump_sequences(temp_board, jr, jc, current_sequence + [step])
                        if sub_chains:
                            all_chains.extend(sub_chains)
                        else:
                            all_chains.append(current_sequence + [step])
        return all_chains

    # Применяет к доске всю цепочку взятий, ход за ходом (для хода ИИ)
    def apply_move_chain(self, board, chain):
        for step in chain:
            self.apply_move_step_to_board(board, step)

    # Применяет к доске один ход. Если captured не None, то снимает шашку соперника. Проверяет превращение в дамки
    def apply_move_step_to_board(self, board, move_step):
        r1, c1, r2, c2, captured = move_step
        piece = board[r1][c1]
        board[r2][c2] = piece
        board[r1][c1] = ''
        if captured:
            board[captured[0]][captured[1]] = ''
        if piece == PLAYER_1_COLOR and r2 == 0:
            board[r2][c2] = PLAYER_1_COLOR + KING_SUFFIX
        if piece == PLAYER_2_COLOR and r2 == BOARD_SIZE - 1:
            board[r2][c2] = PLAYER_2_COLOR + KING_SUFFIX

    # Перерисовывает доску, шашки, дамки, подсветку выбранной фигуры и её возможных ходов        
    def draw_board(self):
        self.canvas.delete("all")
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                color = "#D18B47" if (r + c) % 2 == 1 else "#FFCE9E"
                self.canvas.create_rectangle(x1, y1, x1 + CELL_SIZE, y1 + CELL_SIZE, fill=color, outline="")
                p = self.board[r][c]
                if p:
                    p_clr = "black" if p.startswith("black") else "white"
                    is_king = p.endswith(KING_SUFFIX)
                    self.canvas.create_oval(
                        x1 + 8,
                        y1 + 8,
                        x1 + CELL_SIZE - 8,
                        y1 + CELL_SIZE - 8,
                        fill=p_clr,
                        outline="gray",
                    )
                    if is_king:
                        ring_color = "gold" if p_clr == "black" else "dodger blue"
                        self.canvas.create_oval(
                            x1 + 12, y1 + 12, x1 + CELL_SIZE - 12, y1 + CELL_SIZE - 12,
                            outline=ring_color, width=3
                        )

        if self.selected_piece:
            r, c = self.selected_piece
            self.canvas.create_rectangle(
                c * CELL_SIZE + 2,
                r * CELL_SIZE + 2,
                (c + 1) * CELL_SIZE - 3,
                (r + 1) * CELL_SIZE - 3,
                outline="blue",
                width=3,
            )
            for chain in self.possible_moves:
                target = chain[0]
                self.canvas.create_oval(
                    target[3] * CELL_SIZE + 20,
                    target[2] * CELL_SIZE + 20,
                    (target[3] + 1) * CELL_SIZE - 20,
                    (target[2] + 1) * CELL_SIZE - 20,
                    fill="green",
                    stipple="gray50",
                )

    # Простейшая функция для оценки позиции (с точки зрения ИИ)
    def evaluate(self, board):
        score = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                p = board[r][c]
                if not p:
                    continue
                val = 5 if p.endswith(KING_SUFFIX) else 1
                if p.startswith(self.ai_color):
                    score += val
                else:
                    score -= val
        return score

    # Алгоритм альфа-бета отсечения для выбора хода с поиском на заданную глубину полуходов
    # is_max=True означает ход ИИ (максимизация оценки)
    def alpha_beta(self, board, depth, alpha, beta, is_max):
        moves = self.get_all_moves(board, self.ai_color if is_max else self.human_color)
        if depth == 0 or not moves:
            return self.evaluate(board)
        if is_max:
            max_val = float('-inf')
            for chain in moves:
                temp_board = copy.deepcopy(board)
                self.apply_move_chain(temp_board, chain)
                val = self.alpha_beta(temp_board, depth - 1, alpha, beta, False)
                max_val = max(max_val, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return max_val
        else:
            min_val = float('inf')
            for chain in moves:
                temp_board = copy.deepcopy(board)
                self.apply_move_chain(temp_board, chain)
                val = self.alpha_beta(temp_board, depth - 1, alpha, beta, True)
                min_val = min(min_val, val)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return min_val

    # Это не настоящий сильный MCTS - это простейший вариант метода Монте-Карло
    # Для каждого возможного хода запускает заданное количество случайных партий и смотрит на процент выигрышей
    def get_mc_move(self, simulations):

        root_moves = self.get_all_moves(self.board, self.ai_color)
        if not root_moves:
            return None
        if len(root_moves) == 1:
            return root_moves[0]

        best_score = float('-inf')
        best_move = root_moves[0]

        for move_chain in root_moves:
            wins = 0
            for _ in range(simulations):
                sim_board = copy.deepcopy(self.board)
                self.apply_move_chain(sim_board, move_chain)

                # Короткая симуляция
                curr = self.human_color
                for _ in range(25):  # Лимит ходов в симуляции
                    m = self.get_all_moves(sim_board, curr)
                    if not m:
                        if curr == self.human_color:
                            wins += 1
                        else:
                            wins -= 1
                        break
                    self.apply_move_chain(sim_board, random.choice(m))
                    curr = PLAYER_2_COLOR if curr == PLAYER_1_COLOR else PLAYER_1_COLOR

            if wins > best_score:
                best_score = wins
                best_move = move_chain
        return best_move

    # Обработка щелчка пользователя на поле доски
    def on_click(self, event):

        if self.turn != self.human_color:
            return

        c, r = event.x // CELL_SIZE, event.y // CELL_SIZE

        if not (0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE):
            return

        # Если мы не находимся в режиме обязательного продолжения взятия, разрешаем перевыбрать шашку для хода
        # Вычисляем доступные ходы для выбранной шашки
        if (not self.must_continue_jump) and self.board[r][c].startswith(self.human_color):
            self.selected_piece = (r, c)
            all_m = self.get_all_moves(self.board, self.human_color)
            self.possible_moves = [m for m in all_m if m and m[0][0] == r and m[0][1] == c]
            self.draw_board()
            return

        if not self.selected_piece:
            return

        # Ищем все цепочки, у которых первый шаг ведёт в кликнутую клетку.
        matching = [
            chain for chain in self.possible_moves
            if chain and chain[0][2] == r and chain[0][3] == c
        ]
        if not matching:
            return

        step = matching[0][0]
        self.apply_move_step_to_board(self.board, step)
        self.draw_board()

        # Если было взятие, проверяем, есть ли продолжения. Если есть, то обязательно продолжать взятия, иначе ход завершён
        if step[4] is not None:
            continuations = self.find_jump_sequences(self.board, r, c)
            if continuations:
                self.selected_piece = (r, c)
                self.possible_moves = continuations
                self.must_continue_jump = True
                self.draw_board()
                return

        self.selected_piece = None
        self.possible_moves = []
        self.must_continue_jump = False
        self.draw_board()
        
        self.turn = self.ai_color
        if not self.check_game_over():
            self.after_id = self.root.after(500, self.ai_move)

    # Ход ИИ выбранным режимом (Alpha-Beta или Monte Carlo)
    def ai_move(self):

        mode = self.ai_mode.get()
        best_chain = None

        if mode == 1:
            depth = self.get_safe_int(self.depth_var, 3)
            moves = self.get_all_moves(self.board, self.ai_color)
            if moves:
                best_val = float('-inf')
                best_chain = moves[0]
                for chain in moves:
                    temp_board = copy.deepcopy(self.board)
                    self.apply_move_chain(temp_board, chain)
                    val = self.alpha_beta(temp_board, depth - 1, float('-inf'), float('inf'), False)
                    if val > best_val:
                        best_val = val
                        best_chain = chain
        else:
            sims = self.get_safe_int(self.sims_var, 25)
            self.root.update()
            best_chain = self.get_mc_move(sims)

        if not best_chain:
            self.check_game_over()
            return    
        
        for step in best_chain:
            self.apply_move_step_to_board(self.board, step)
            self.draw_board()
            self.root.update()
            self.root.after(300)

        self.turn = self.human_color
        self.check_game_over()

    # Проверка окончания игры: если игрок не может сделать ход в свою очередь хода, он проиграл
    def check_game_over(self):
        if self.turn is None:
            return True
        moves = self.get_all_moves(self.board, self.turn)
        if moves:
            return False
        if self.turn == self.human_color:
            messagebox.showinfo("Конец игры", "ИИ победил")
        else:
            messagebox.showinfo("Конец игры", "Вы победили!")
        self.turn = None
        return True


if __name__ == "__main__":
    root = tk.Tk()
    # Автоподбор высоты окна под доску и элементы управления
    root.geometry(f"{BOARD_SIZE * CELL_SIZE}x{BOARD_SIZE * CELL_SIZE + 240}")
    game = CheckersGame(root)
    root.mainloop()