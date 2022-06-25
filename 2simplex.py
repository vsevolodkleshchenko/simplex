import numpy as np
import itertools as it
import copy

DEC = 2

# Задание параметров задачи #
print("\n##### УСЛОВИЕ #####\n")
# матрица условий
A = np.array([np.array([3, 4, 6, 1, -1]),
              np.array([2, 0, -3, 2, 0]),
              np.array([-1, 2, 0, 1, 2])])
# вектор ограничений
B = np.array([2, 10, 14])
# вектор коэффициентов целевой функции
C = np.array([7, -2, 0, 3, 1])

print("Коэффициенты целевой функции:", C, "Матрица условий:", A, "Вектор ограничений:", B, sep="\n")


def find_accept_basis(A, C):
    """ Ищем допустимый базис
    Возвращает: массив номеров базисных столбцов,
                матрицу обратную к "дельта(бэта)",
                массив "с(бэта)",
                массив "u * A - c" """
    rank = np.linalg.matrix_rank(A)
    combs = list(it.combinations(np.arange(A.shape[1]), rank))
    for comb in combs:
        delta = A[:, comb]
        c = np.array([C[i] for i in comb])
        if np.linalg.det(delta) == 0:
            continue
        delta_1 = np.linalg.inv(delta)
        uA_c = (c @ delta_1) @ A - C
        if np.all(np.around(uA_c, decimals=DEC) >= 0):
            N = np.array(list(comb))
            print("Допустимый базис может быть составлен из:", *(N + 1), "столбцов\n")
            return N, delta_1, c, uA_c


def build_table(a, a0, b, b0, N):
    """ Строит первую симплекс таблицу из блоков
    (названия блоков совпадают с названиям в конспекте)"""
    simplex_table = np.zeros((a.shape[0] + 1, a.shape[1] + 2))
    simplex_table[0, 0] = None
    simplex_table[1:, 0] = N + 1
    simplex_table[0, 1] = b0
    simplex_table[0, 2:] = a0
    simplex_table[1:, 1] = b
    simplex_table[1:, 2:] = a
    return simplex_table


def update_table(table):
    """ Обновляет симплекс таблицу
     Возвращается новая симплекс таблица"""
    new_table = copy.deepcopy(table)
    n_in, n_out = find_in_out(table)
    for c in range(1, new_table.shape[1]):
        for r in range(new_table.shape[0]):
            new_table[r, c] = table[r, c] - table[r, n_in] / table[n_out, n_in] * table[n_out, c]
        new_table[n_out, c] = table[n_out, c] / table[n_out, n_in]
    new_table[n_out, 0] = n_in - 1
    return new_table


def find_in_out(table):
    """ Находит номера вводимого (in) и выводимого (out) столбцов"""
    a0 = table[0, 2:]
    b = table[1:, 1]
    a = table[1:, 2:]
    n_out = np.where(np.around(b, decimals=DEC) < 0)[0][0]
    n_ = np.where(np.around(a, decimals=DEC)[n_out] < 0)[0]
    minimal = np.min(- a0[n_] / a[n_out, n_])
    n_in = np.intersect1d(np.where(- a0 / a[n_out] == minimal)[0], n_)[0]
    return n_in + 2, n_out + 1


def print_results(simplex_table):
    """ Выводит ответ КЗЛП по итоговой симплекс таблице"""
    print("Итоговое значение целевой функции:", np.around(simplex_table[0, 1], decimals=2))
    for i in range(1, simplex_table.shape[1] - 1):
        if i in simplex_table[1:, 0]:
            print("x_"+str(i), "=", np.around(simplex_table[np.where(simplex_table[1:, 0] == i)[0][0] + 1, 1], decimals=2))
        else:
            print("x_"+str(i), "=", 0)


def do_simplex(A, B, C):
    """ Основная функция двойственного симплекс метода """
    print("\n##### РЕШЕНИЕ #####\n")
    # ищем допустимый базис и получаем блоки симплекс таблицы
    N, delta_1, c, a0 = find_accept_basis(A, C)
    b = delta_1 @ B
    b0 = c @ b
    a = delta_1 @ A

    # составляем из блоков первую симплекс таблицу
    simplex_table = build_table(a, a0, b, b0, N)
    print("1-й шаг:", np.around(simplex_table, decimals=2), sep="\n")

    # делаем шаги симплекс метода
    i = 1
    while np.any(np.around(simplex_table[1:, 1], decimals=DEC) < 0):
        i += 1
        simplex_table = update_table(simplex_table)
        print("\n"+str(i)+"-й шаг:", np.around(simplex_table, decimals=2), sep="\n")
    print("\n##### ОТВЕТ #####\n")
    print_results(simplex_table)


do_simplex(A, B, C)
