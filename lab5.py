from _pydecimal import Decimal
from scipy.stats import f, t
import numpy as np
import scipy.stats
from random import randint
from copy import deepcopy

check = False

x1min = -8
x1max = 9
x2min = -8
x2max = 6
x3min = -5
x3max = 6
y_min = 200 + int((x1min + x2min + x3min) / 3)
y_max = 200 + int((x1max + x2max + x3max) / 3)

N = 8
q = 0.05
print("y=b0+b1*x1+b2*x2+b3*x3+b12*x1*x2+b13*x1*x3+b23*x2*x3+b123*x1*x2*x3+b11*x1^2+b22*x2^2+b33*x3^2")
x_array = np.array([[x1max, x2max, x3max, x1max * x2max, x1max * x3max, x2max * x3max, x1max * x2max * x3max],
                    [x1max, x2max, x3min, x1max * x2max, x1max * x3min, x2max * x3min, x1max * x2max * x3min],
                    [x1max, x2min, x3max, x1max * x2min, x1max * x3max, x2min * x3max, x1max * x2min * x3max],
                    [x1max, x2min, x3min, x1max * x2min, x1max * x3min, x2min * x3min, x1max * x2min * x3min],
                    [x1min, x2max, x3max, x1min * x2max, x1min * x3max, x2max * x3max, x1min * x2max * x3max],
                    [x1min, x2max, x3min, x1min * x2max, x1min * x3min, x2max * x3min, x1min * x2max * x3min],
                    [x1min, x2min, x3max, x1min * x2min, x1min * x3max, x2min * x3max, x1min * x2min * x3max],
                    [x1min, x2min, x3min, x1min * x2min, x1min * x3min, x2min * x3min, x1min * x2min * x3min]])

y_array = np.array([[210, 215, 215],
                    [210, 215, 215],
                    [210, 224, 215],
                    [225, 215, 219],
                    [210, 215, 215],
                    [210, 235, 218],
                    [210, 230, 215],
                    [231, 215, 232]])


def average_y(_i, _m):
    total = 0
    for j in range(0, len(y_array[0])):
        total += y_array[_i][j]
    return total / _m


def dispersion(line, average, _m):
    total = 0
    for _i in range(0, len(y_array[0])):
        total += (y_array[line][_i] - average) ** 2
    return total / _m


def kohren():
    _m = 3
    global x_array, y_array
    kohrens_list = [(1, 0.6798), (2, 0.5157), (3, 0.4377), (4, 0.391), (5, 0.3595), (6, 0.3362), (7, 0.3185),
                    (8, 0.3043),
                    (9, 0.2926), (10, 0.2829), (16, 0.2462), (36, 0.2022),
                    (144, 0.1616)]  # хардкод при f2 = 8 т.к. f2 статично
    while True:
        my_average_array = []
        for _i in range(0, N):
            my_average_array.append(average_y(_i, _m))
        my_dispersion_array = []
        for _i in range(0, N):
            my_dispersion_array.append(dispersion(_i, my_average_array[_i], _m))
        gp = np.max(np.array(my_dispersion_array)) / np.sum(np.array(my_dispersion_array))
        _f1 = _m - 1
        _f2 = N
        if gp <= kohrens_list[_f1 - 1][1]:
            return gp, _m, my_average_array, my_dispersion_array, _f1, _f2
        _m += 1  # increase in repetition
        additional = np.array([[randint(y_min, y_max)],
                               [randint(y_min, y_max)],
                               [randint(y_min, y_max)],
                               [randint(y_min, y_max)],
                               [randint(y_min, y_max)],
                               [randint(y_min, y_max)],
                               [randint(y_min, y_max)],
                               [randint(y_min, y_max)]])
        y_array = np.append(y_array, additional, axis=1)  # add column to planning matrix


Gp, m, my_Y_averageArray, myDispersionArray, f1, f2 = kohren()
print(f"m={m}, при Gр={Gp}")

my_X_averageArray = []


def average_x_in_line(position):
    total = 0
    for _i in range(N):
        total += x_array[_i][position]
    return total / N


for i in range(3):
    my_X_averageArray.append(average_x_in_line(i))
my = np.sum(my_Y_averageArray) / m


def get_sum(*args):  # take int X or array Xi_values for multiplication
    summa = 0
    try:
        if args[0] == "y":
            if len(args) == 1:
                summa = sum(my_Y_averageArray)
            else:
                for j in range(N):
                    sum_i_temp = 1
                    for _i in range(len(args) - 1):  # loop for all arguments except "у"
                        sum_i_temp *= x_array[j][args[_i + 1] - 1]  # all "X" multiplication
                    sum_i_temp *= my_Y_averageArray[j]  # multiply by "у"
                    summa += sum_i_temp

        elif len(args) == 1:
            args = args[0] - 1
            for obj in x_array:
                summa += obj[args]
        else:  # if function has cortege as input
            for obj in x_array:
                sum_i_temp = 1
                for _i in range(len(args)):
                    sum_i_temp *= obj[
                        args[_i] - 1]  # multiply all X from cortege, add twice X to cortege for square
                summa += sum_i_temp

    except:
        print("def error")
    return summa


coeffList_1 = [N, get_sum(1), get_sum(2), get_sum(3), get_sum(1, 2), get_sum(1, 3),
               get_sum(2, 3), get_sum(1, 2, 3)]
coeffList_2 = [get_sum(1), get_sum(1, 1), get_sum(1, 2), get_sum(1, 3), get_sum(1, 1, 2),
               get_sum(1, 1, 3), get_sum(1, 2, 3), get_sum(1, 1, 2, 3)]
coeffList_3 = [get_sum(2), get_sum(1, 2), get_sum(2, 2), get_sum(2, 3), get_sum(1, 2, 2),
               get_sum(1, 2, 3), get_sum(2, 2, 3), get_sum(1, 2, 2, 3)]
coeffList_4 = [get_sum(3), get_sum(1, 3), get_sum(2, 3), get_sum(3, 3), get_sum(1, 2, 3),
               get_sum(1, 3, 3), get_sum(2, 3, 3), get_sum(1, 2, 3, 3)]
coeffList_5 = [get_sum(1, 2), get_sum(1, 1, 2), get_sum(1, 2, 2), get_sum(1, 2, 3),
               get_sum(1, 1, 2, 2), get_sum(1, 1, 2, 3), get_sum(1, 2, 2, 3),
               get_sum(1, 1, 2, 2, 3)]
coeffList_6 = [get_sum(1, 3), get_sum(1, 1, 3), get_sum(1, 2, 3), get_sum(1, 3, 3),
               get_sum(1, 1, 2, 3), get_sum(1, 1, 3, 3), get_sum(1, 2, 3, 3),
               get_sum(1, 1, 2, 3, 3)]
coeffList_7 = [get_sum(2, 3), get_sum(1, 2, 3), get_sum(2, 2, 3), get_sum(2, 3, 3),
               get_sum(1, 2, 2, 3), get_sum(1, 2, 3, 3), get_sum(2, 2, 3, 3),
               get_sum(1, 2, 2, 3, 3)]
coeffList_8 = [get_sum(1, 2, 3), get_sum(1, 1, 2, 3), get_sum(1, 2, 2, 3), get_sum(1, 2, 3, 3),
               get_sum(1, 1, 2, 2, 3), get_sum(1, 1, 2, 3, 3), get_sum(1, 2, 2, 3, 3),
               get_sum(1, 1, 2, 2, 3, 3)]

coeffListNinth = [k0, k1, k2, k3, k4, k5, k6, k7] = [get_sum("y"), get_sum("y", 1), get_sum("y", 2), get_sum(
    "y", 3), get_sum("y", 1, 2), get_sum("y", 1, 3), get_sum("y", 2, 3), get_sum("y", 1, 2, 3)]

fullList = [coeffList_1, coeffList_2, coeffList_3, coeffList_4, coeffList_5, coeffList_6, coeffList_7, coeffList_8]


def positioning(position):
    new_fulllist = deepcopy(fullList)
    count = 0
    for each in new_fulllist:
        each.insert(position, coeffListNinth[count])
        each.pop(position + 1)
        count += 1
    return new_fulllist


fullDet = np.linalg.det(np.array(fullList))

b_array = [np.linalg.det(np.array(positioning(0))) / fullDet,
           np.linalg.det(np.array(positioning(1))) / fullDet,
           np.linalg.det(np.array(positioning(2))) / fullDet,
           np.linalg.det(np.array(positioning(3))) / fullDet]
b12 = np.linalg.det(np.array(positioning(4))) / fullDet
b13 = np.linalg.det(np.array(positioning(5))) / fullDet
b23 = np.linalg.det(np.array(positioning(6))) / fullDet
b123 = np.linalg.det(np.array(positioning(7))) / fullDet
"""
print(str(round(b0, 2)) + str(round(b1,2)) +" * x + "+ str(round(b2,2)) + " * x + "+str(round(b3,20))+ " * x + 
    "+str(round(b12,2))+" * x + "+str(round(b13,2))+" * x +"+str(round(b23,2))+ " * x +"+str(round(b123,2))+" * x")
"""
# Стьюдент
S2B = np.sum(myDispersionArray) / N
S2Bs = S2B / m / N
Sbs = np.sqrt(S2Bs)

x_array_normal = [
    [1, -1, -1, -1],
    [1, -1, -1, 1],
    [1, -1, 1, -1],
    [1, -1, 1, 1],
    [1, 1, -1, -1],
    [1, 1, -1, 1],
    [1, 1, 1, -1],
    [1, 1, 1, 1],
]


def get_beta(_i):
    summa = 0
    for j in range(N):
        summa += my_Y_averageArray[j] * x_array_normal[j][_i]
    summa /= N
    return summa


beta0 = get_beta(0)
beta1 = get_beta(1)
beta2 = get_beta(2)
beta3 = get_beta(3)

t_array = [abs(beta0) / Sbs, abs(beta1) / Sbs, abs(beta2) / Sbs, abs(beta3) / Sbs]

f3 = f1 * f2

t_tab = scipy.stats.t.ppf((1 + (1 - q)) / 2, f3)
print("t from table:", t_tab)

for i in range(len(t_array)):
    if t_array[i] < t_tab:
        b_array[i] = 0
        print("t" + str(i) + ":", t_array[i], " t" + str(i) + "<t_tab; b" + str(i) + "=0")

y_hat = []
for i in range(N):
    y_hat.append(
        b_array[0] + b_array[1] * x_array[i][0] + b_array[2] * x_array[i][1] + b_array[3] * x_array[i][2] +
        b12 * x_array[i][0] * x_array[i][1] +
        b13 * x_array[i][0] * x_array[i][2] + b123 * x_array[i][0] * x_array[i][1] * x_array[i][2])
"""
    print ( f"y{i + 1}_hat = {b0:.2f}{b1:+.2f}*x{i + 1}1{b2:+.2f}*x{i + 1}2{b3:+.2f}*x{i + 1}3{b12:+.2f}*x{i + 1}1"
            f"*x{i + 1}2{b13:+.2f}*x{i + 1}1*x{i + 1}3{b123:+.2f}*x{i + 1}1*x{i + 1}2*x{i + 1}3 "
            f"= {y_hat[ i ]:.2f}" )
"""
d = 2
f4 = N - d
S2_ad = 0
for i in range(N):
    S2_ad += (m / (N - d) * ((y_hat[i] - my_Y_averageArray[i]) ** 2))

Fp = S2_ad / S2B
Ft = scipy.stats.f.ppf(1 - q, f4, f3)
print("Fp:", Fp)
print("Ft:", Ft)
if Fp > Ft:
    print("Adequate precisely at 0,05")
    check = True
else:
    print("Not adequate precisely at 0,05")

if check == False:

    m = 0
    d = 0
    N = 15

    x1_min = x1min
    x1_max = x1max
    x2_min = x2min
    x2_max = x2max
    x3_min = x3min
    x3_max = x3max

    x01 = (x1_max - x1_min) / 2
    x02 = (x2_max - x2_min) / 2
    x03 = (x3_max - x3_min) / 2
    delta_x1 = x1_max - x01
    delta_x2 = x2_max - x02
    delta_x3 = x3_max - x03
    y_min = 200 + int((x1_min + x2_min + x3_min) / 3)
    y_max = 200 + int((x1_max + x2_max + x3_max) / 3)


    def sqrt(element):
        from math import sqrt
        return sqrt(element)


    def fab(element):
        from math import fabs
        return fabs(element)


    correct_input = False
    while not correct_input:
        try:
            m = int(input("Enter number of reiteration: "))
            p = float(input("Enter confidence probability: "))
            correct_input = True
        except ValueError:
            pass

    # starts cycle where the main code performs
    correct = False
    while not correct:
        try:
            array = [  # initial array
                [-1, -1, -1, +1, +1, +1, -1, +1, +1, +1],
                [-1, -1, +1, +1, -1, -1, +1, +1, +1, +1],
                [-1, +1, -1, -1, +1, -1, +1, +1, +1, +1],
                [-1, +1, +1, -1, -1, +1, -1, +1, +1, +1],
                [+1, -1, -1, -1, -1, +1, +1, +1, +1, +1],
                [+1, -1, +1, -1, +1, -1, -1, +1, +1, +1],
                [+1, +1, -1, +1, -1, -1, -1, +1, +1, +1],
                [+1, +1, +1, +1, +1, +1, +1, +1, +1, +1],
                [-1.215, 0, 0, 0, 0, 0, 0, 1.4623, 0, 0],
                [+1.215, 0, 0, 0, 0, 0, 0, 1.4623, 0, 0],
                [0, -1.215, 0, 0, 0, 0, 0, 0, 1.4623, 0],
                [0, +1.215, 0, 0, 0, 0, 0, 0, 1.4623, 0],
                [0, 0, -1.215, 0, 0, 0, 0, 0, 0, 1.4623],
                [0, 0, +1.215, 0, 0, 0, 0, 0, 0, 1.4623],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]


            class Critical_values:
                @staticmethod
                def get_cohren_value(size_of_selections, qty_of_selections, significance):
                    size_of_selections += 1
                    partResult1 = significance / (size_of_selections - 1)
                    params = [partResult1, qty_of_selections, (size_of_selections - 1 - 1) * qty_of_selections]
                    fisher = f.isf(*params)
                    result = fisher / (fisher + (size_of_selections - 1 - 1))
                    return Decimal(result).quantize(Decimal('.0001')).__float__()

                @staticmethod
                def get_student_value(f3, significance):
                    return Decimal(abs(t.ppf(significance / 2, f3))).quantize(Decimal('.0001')).__float__()

                @staticmethod
                def get_fisher_value(f3, f4, significance):
                    return Decimal(abs(f.isf(significance, f4, f3))).quantize(Decimal('.0001')).__float__()


            def x(l1, l2, l3):
                x_1 = l1 * delta_x1 + x01
                x_2 = l2 * delta_x2 + x02
                x_3 = l3 * delta_x3 + x03
                return [x_1, x_2, x_3]


            def generate_matrix():
                from random import randrange
                matrix_with_y = [[randrange(y_min, y_max) for y in range(m)] for x in range(N)]
                return matrix_with_y


            def find_average(lst, orientation):
                average = []
                if orientation == 1:
                    for rows in range(len(lst)):
                        average.append(sum(lst[rows]) / len(lst[rows]))
                else:
                    for column in range(len(lst[0])):
                        number_lst = []
                        for rows in range(len(lst)):
                            number_lst.append(lst[rows][column])
                        average.append(sum(number_lst) / len(number_lst))
                return average


            def count(first, second):
                need_a = 0
                for j in range(N):
                    need_a += x_matrix[j][first - 1] * x_matrix[j][second - 1] / N
                return need_a


            def find_known(number):
                need_a = 0
                for j in range(N):
                    need_a += average_y[j] * x_matrix[j][number - 1] / 15
                return need_a


            def solve(lst_1, lst_2):
                from numpy.linalg import solve
                solver = solve(lst_1, lst_2)
                return solver


            def check_result(b_lst, k):
                y_i = b_lst[0] + b_lst[1] * matrix[k][0] + b_lst[2] * matrix[k][1] + b_lst[3] * matrix[k][2] + \
                      b_lst[4] * matrix[k][3] + b_lst[5] * matrix[k][4] + b_lst[6] * matrix[k][5] + b_lst[7] * \
                      matrix[k][6] + \
                      b_lst[8] * matrix[k][7] + b_lst[9] * matrix[k][8] + b_lst[10] * matrix[k][9]
                return y_i


            def student_test(b_lst, number_x=10):
                b_dispersion = sqrt(dispersion_b2)
                for column in range(number_x):
                    t_practice = 0
                    t_theoretical = Critical_values.get_student_value(f3, q)
                    for row in range(N):
                        if column == 0:
                            t_practice += average_y[row] / N
                        else:
                            t_practice += average_y[row] * array[row][column - 1]
                    if (t_practice / b_dispersion) < t_theoretical:
                        b_lst[column] = 0
                return b_lst


            def fisher_test():
                dispersion_ad = 0
                f4 = N - d
                for row in range(len(average_y)):
                    dispersion_ad += (m * (average_y[row] - check_result(student_lst, row))) / (N - d)
                F_practice = dispersion_ad / dispersion_b2
                F_theoretical = Critical_values.get_fisher_value(f3, f4, q)
                return F_practice < F_theoretical


            x_matrix = [[] for x in range(N)]
            for i in range(len(x_matrix)):
                if i < 8:
                    x1 = x1_min if array[i][0] == -1 else x1_max
                    x2 = x2_min if array[i][1] == -1 else x2_max
                    x3 = x3_min if array[i][2] == -1 else x3_max
                else:
                    x_lst = x(array[i][0], array[i][1], array[i][2])
                    x1, x2, x3 = x_lst
                x_matrix[i] = [x1, x2, x3, x1 * x2, x1 * x3, x2 * x3, x1 * x2 * x3, x1 ** 2, x2 ** 2, x3 ** 2]

            matrix_y = generate_matrix()
            average_x = find_average(x_matrix, 0)
            average_y = find_average(matrix_y, 1)
            matrix = [(x_matrix[i] + matrix_y[i]) for i in range(N)]
            mx_i = average_x
            my = sum(average_y) / 15

            values_arr = [
                [1, mx_i[0], mx_i[1], mx_i[2], mx_i[3], mx_i[4], mx_i[5], mx_i[6], mx_i[7], mx_i[8], mx_i[9]],
                [mx_i[0], count(1, 1), count(1, 2), count(1, 3), count(1, 4), count(1, 5), count(1, 6), count(1, 7),
                 count(1, 8), count(1, 9), count(1, 10)],
                [mx_i[1], count(2, 1), count(2, 2), count(2, 3), count(2, 4), count(2, 5), count(2, 6), count(2, 7),
                 count(2, 8), count(2, 9), count(2, 10)],
                [mx_i[2], count(3, 1), count(3, 2), count(3, 3), count(3, 4), count(3, 5), count(3, 6), count(3, 7),
                 count(3, 8), count(3, 9), count(3, 10)],
                [mx_i[3], count(4, 1), count(4, 2), count(4, 3), count(4, 4), count(4, 5), count(4, 6), count(4, 7),
                 count(4, 8), count(4, 9), count(4, 10)],
                [mx_i[4], count(5, 1), count(5, 2), count(5, 3), count(5, 4), count(5, 5), count(5, 6), count(5, 7),
                 count(5, 8), count(5, 9), count(5, 10)],
                [mx_i[5], count(6, 1), count(6, 2), count(6, 3), count(6, 4), count(6, 5), count(6, 6), count(6, 7),
                 count(6, 8), count(6, 9), count(6, 10)],
                [mx_i[6], count(7, 1), count(7, 2), count(7, 3), count(7, 4), count(7, 5), count(7, 6), count(7, 7),
                 count(7, 8), count(7, 9), count(7, 10)],
                [mx_i[7], count(8, 1), count(8, 2), count(8, 3), count(8, 4), count(8, 5), count(8, 6), count(8, 7),
                 count(8, 8), count(8, 9), count(8, 10)],
                [mx_i[8], count(9, 1), count(9, 2), count(9, 3), count(9, 4), count(9, 5), count(9, 6), count(9, 7),
                 count(9, 8), count(9, 9), count(9, 10)],
                [mx_i[9], count(10, 1), count(10, 2), count(10, 3), count(10, 4), count(10, 5), count(10, 6),
                 count(10, 7), count(10, 8), count(10, 9), count(10, 10)]
            ]
            outputArray = [my, find_known(1), find_known(2), find_known(3), find_known(4), find_known(5), find_known(6),
                           find_known(7),
                           find_known(8), find_known(9), find_known(10)]

            beta = solve(values_arr, outputArray)
            print("\tRegression equation =>")
            print("{:.3f} + {:.3f} * X1 + {:.3f} * X2 + {:.3f} * X3 + {:.3f} * Х1X2 + {:.3f} * Х1X3 + {:.3f} * Х2X3"
                  "+ {:.3f} * Х1Х2X3 + {:.3f} * X11^2 + {:.3f} * X22^2 + {:.3f} * X33^2 = ŷ\n\t check"
                  .format(beta[0], beta[1], beta[2], beta[3], beta[4], beta[5], beta[6], beta[7], beta[8], beta[9],
                          beta[10]))
            for i in range(N):
                print("ŷ{} = {:.3f} ≈ {:.3f}".format((i + 1), check_result(beta, i), average_y[i]))

            y_dispersion = [0.0 for x in range(N)]
            homogeneity = False
            while not homogeneity:
                y_dispersion = [0.0 for x in range(N)]
                for i in range(N):
                    dispersion_i = 0
                    for j in range(m):
                        dispersion_i += (matrix_y[i][j] - average_y[i]) ** 2
                    y_dispersion.append(dispersion_i / (m - 1))

                f1 = m - 1
                f2 = N
                f3 = f1 * f2
                q = 1 - p

                Gp = max(y_dispersion) / sum(y_dispersion)

                print("\n")
                print("\tKohrens critetion")
                Gt = Critical_values.get_cohren_value(f2, f1, q)
                if Gt > Gp or m >= 25:  # because 25 is max number of reiterations
                    print("\t\tDispersion is homogeneous {:.2f}!".format(q))
                    homogeneity = True
                else:
                    print("\t\tDispersion is not homogeneous {:.2f}!".format(q))
                    m += 1  # new Y column will be added
            dispersion_b2 = sum(y_dispersion) / (N * N * m)
            student_lst = list(student_test(beta))
            print("\tRegression equation after Students criterion")
            print("{:.3f} + {:.3f} * X1 + {:.3f} * X2 + {:.3f} * X3 + {:.3f} * Х1X2 + {:.3f} * Х1X3 + {:.3f} * Х2X3"
                  "+ {:.3f} * Х1Х2X3 + {:.3f} * X11^2 + {:.3f} * X22^2 + {:.3f} * X33^2 = ŷ\n\tcheck"
                  .format(student_lst[0], student_lst[1], student_lst[2], student_lst[3], student_lst[4],
                          student_lst[5],
                          student_lst[6], student_lst[7], student_lst[8], student_lst[9], student_lst[10]))
            for i in range(N):
                print("ŷ{} = {:.3f} ≈ {:.3f}".format((i + 1), check_result(student_lst, i), average_y[i]))

            print("\n")
            print("\tFishers criterion =>")

            d = 11 - student_lst.count(0)
            if fisher_test():
                correct = True
                print("\t\tRegression equation is adequate")  # end code
            else:
                print(
                    "\t\tRegression equation is not adequate")  # cycle starts with new Y array until got adequate regresssion
        except ValueError:
            pass
