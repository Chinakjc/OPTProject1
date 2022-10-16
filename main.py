import numpy as np
import matplotlib.pyplot as plt
import time
import sys

eps = sys.float_info.epsilon
norme = np.linalg.norm
solve = np.linalg.solve


def newton_method(v0, df, hf, k_max):
    v = v0
    k = 0
    while (eps < norme(df(v))) and (k < k_max):
        # Xk+1 = Xk - inv(H)*df
        # H(Xk+1 -Xk) = -df = H(y)
        # Xk+1 = y + Xk
        y = solve(hf(v), -df(v))
        v = y + v
        k = k + 1
    return v


def newton_method_analyse_convergence(v0, df, hf, k_max):
    v = v0
    k = 0
    err = list()
    err0 = norme(df(v))
    t0 = time.time()
    while (eps < norme(df(v))) and (k < k_max):
        # Xk+1 = Xk - inv(H)*df
        # H(Xk+1 -Xk) = -df = H(y)
        # Xk+1 = y + Xk
        err.append(norme(df(v)) / err0)
        y = solve(hf(v), -df(v))
        v = y + v
        k = k + 1
    tt = time.time()
    print("nombre d'itération : " + str(k)+", avec temps de calculation : "+ str(tt-t0)+" seconds.")
    plt.plot(np.log([x + 1 for x in range(k)]), np.log(err))
    plt.show()
    return v


def f(x):
    x1 = x[0]
    x2 = x[1]
    return np.exp(np.prod(x)) - ((x1 ** 3 + x2 ** 3 + 1) ** 2) / 2.0


def param_grad_f(x, i):
    y1 = np.prod(x[:i])
    y2 = np.prod(x[i + 1:])
    return y1 * y2


def param_hf(x, i, j):
    m = np.block([[0, 2, 2, 2, 2], [1, 1, 2, 2, 2], [1, 2, 1, 2, 2], [1, 2, 2, 1, 2], [1, 2, 2, 2, 1]])
    k = np.block([[np.zeros(5)], [np.zeros(2), np.ones(3)], [0, 1, 0, 1, 1], [0, 1, 1, 0, 1], [0, 1, 1, 1, 0]])
    p = np.eye(5)
    p[0][0] = 0
    p[i][i] = 0
    p[0][i] = 1
    p[i][0] = 1
    m = np.dot(p.T, np.dot(m, p))
    k = np.dot(p.T, np.dot(k, p))
    '''print("m = ")
    print(m)
    print("k = ")
    print(k)'''
    param1 = 1
    param2 = 1
    for t in range(5):
        xt = x[t]
        coef1 = m[j][t]
        coef2 = k[j][t]
        param1 *= (xt ** coef1)
        param2 *= (xt ** coef2)

    return param1 + param2


def grad_f(x):
    res = list()
    for i in range(5):
        res.append(param_grad_f(x, i) * np.exp(np.prod(x)))
    res = np.expand_dims(res, axis=0).T
    x1 = x[0]
    x2 = x[1]
    r1 = - 3 * (x1 ** 3 + x2 ** 3 + 1) * x1 ** 2
    r2 = - 3 * (x1 ** 3 + x2 ** 3 + 1) * x2 ** 2
    b1 = np.block([[r1, np.zeros(4)]]).T

    b2 = np.block([[np.zeros(1), r2, np.zeros(3)]]).T

    return (res + b1 + b2)


def hessian_f(x):
    res = list()
    for i in range(5):
        for j in range(5):
            res.append(param_hf(x, i, j) * np.exp(np.prod(x)))
    res = np.reshape(res, [5, 5])
    x1 = x[0]
    x2 = x[1]
    res[0][0] = res[0][0] - 9 * x1 ** 4 - 6 * (x1 ** 3 + x2 ** 3 + 1) * x1
    res[0][1] = res[0][1] - 9 * x1 ** 2 * x2 ** 2
    res[1][0] = res[1][0] - 9 * x1 ** 2 * x2 ** 2
    res[1][1] = res[1][1] - 9 * x2 ** 4 - 6 * (x1 ** 3 + x2 ** 3 + 1) * x2
    return res


def g1(x):
    return np.dot(x.T, x) - 10


def g2(x):
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    return x2 * x3 - 5 * x4 * x5


def g3(x):
    x1 = x[0]
    x2 = x[1]
    return x1 ** 3 + x2 ** 3 + 1


def g(x):
    return np.block([[g1(x), g2(x), g3(x)]]).T


def jacob_g(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    line1 = np.block([2 * x1, 2 * x2, 2 * x3, 2 * x4, 2 * x5])
    line2 = np.block([0, x3, x2, -5 * x5, -5 * x4])
    line3 = np.block([3 * x1 ** 2, 3 * x2 ** 2, 0, 0, 0])
    return np.block([[line1], [line2], [line3]])


def d_g(x, i, j):  # di_gj(x)
    jac = jacob_g(x)
    return jac[i][j]


def h_g(x, i):  # h_gi(x)
    if (i == 1):
        return 2 * np.eye(5)
    if (i == 2):
        line1 = np.zeros(5)
        line2 = np.block([0, 0, 1, 0, 0])
        line3 = np.block([0, 1, 0, 0, 0])
        line4 = np.block([np.zeros(4), np.array([-5])])
        line5 = np.block([0, 0, 0, -5, 0])
        return np.block([[line1], [line2], [line3], [line4], [line5]])
    x1 = x[0]
    x2 = x[1]
    line1 = np.block([6 * x1, np.zeros(4)])
    line2 = np.block([0, 6 * x2, np.zeros(3)])
    return np.block([[line1], [line2], [np.zeros([3, 5])]])


def hessian_g_product_lambda(l, x):
    l1 = l[0]
    l2 = l[1]
    l3 = l[2]
    x1 = x[0]
    x2 = x[1]
    line1 = np.block([2 * l1 + 6 * l3 * x1, np.zeros(4)])
    line2 = np.block([0, 2 * l1 + 6 * l3 * x2, l2, 0, 0])
    line3 = np.block([0, l2, 2 * l1, 0, 0])
    line4 = np.block([0, 0, 0, 2 * l1, -5 * l2])
    line5 = np.block([0, 0, 0, -5 * l2, 2 * l1])
    return np.block([[line1], [line2], [line3], [line4], [line5]])
    '''res = np.array([])
    for i in range(5):
        ligne = np.array([])
        for j in range(5):
            s = 0
            for k in range(3):
                s += l[k] * h_g(x, k + 1)[i][j]
            ligne = np.block([ligne, s])
        res = np.append(res, ligne)
    return np.reshape(res, [5, 5])'''
    '''x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    l1 = l[0]
    l2 = l[1]
    l3 = l[2]
    line1 = np.block([2 * x1, 0, 3 * x1 ** 2, 6 * l3 * x1 + 2 * l1, np.zeros(4)])
    line2 = np.block([2 * x2, x3, 3 * x2 ** 2, 0, 6 * l3 * x2 + 2 * l1, l2, np.zeros(2)])
    line3 = np.block([2 * x3, x2, 0, 0, l2, 2 * l1, 0, 0])
    line4 = np.block([2 * x4, -5 * x5, np.zeros(4), 2 * l1, -5 * l2])
    line5 = np.block([2 * x5, -5 * x4, np.zeros(4), -5 * l2, 2 * l1])
    return np.block([[line1], [line2], [line3], [line4], [line5]])'''


def lagrange_f_g(l, x):
    return f(x) - np.dot(l.T, g(x))


def grad_lagrange_f_g(l, x):
    return np.block([[g(x)], [grad_f(x) + np.dot(jacob_g(x).T, l)]])


def hessian_lagrange_f_g(l, x):
    o3 = np.zeros([3, 3])
    jac = jacob_g(x)
    return np.block([[o3, jac], [jac.T, hessian_f(x) + hessian_g_product_lambda(l, x)]])


'''b = np.array([[1, 2, 3]])
print(b)
print(b.T)
print(np.dot(b, b.T))
x = np.array([0.1,0.2,1,1.5,0.5])
print("x = "+str(x))
#print(grad_f(x))
#print(hessian_f(x))
l = np.array([1,1,1])
#print(hessian_g_product_lambda(l,x))
#print(lagrange_f_g(l,x))
#print(np.zeros([3,3]))
print(hessian_lagrange_f_g(l,x))'''


def d_lagrange(v):
    l = v[:3]
    x = v[3:]
    return grad_lagrange_f_g(l, x)


def h_lagrange(v):
    l = v[:3]
    x = v[3:]
    return hessian_lagrange_f_g(l, x)


def newton(l0, x0, k_max, mode: bool = False):
    print("Appliquier Newton pour condition initiale suivante : ")
    print("x0 : ")
    print(x0)
    print("l0 : ")
    print(l0)
    print("En cours de calculation !")
    v0 = np.block([[l0], [x0]])
    if (mode):
        sol = newton_method_analyse_convergence(v0, d_lagrange, h_lagrange, k_max)
    else:
        sol = newton_method(v0, d_lagrange, h_lagrange, k_max)
    l_sol = sol[:3]
    x_sol = sol[3:]
    print("Calculation est terminée !")
    print("min de f est : " + str(f(x_sol)))
    print("x_sol est : ")
    print(str(x_sol))
    print("lambda_sol est : ")
    print(l_sol)
    print("Pour vérification de la condition contrainte, g(x_sol) = ")
    print(g(x_sol))


def sqp_method(v0, df, dl, hl, dg, g, k_max):
    v = v0
    k = 0
    while (eps < norme(dl(v))) and (k < k_max):
        l = v[:3]
        x = v[3:]
        hlk = hl(v)[-5:, -5:]
        dfk = df(x)
        jk = dg(x)
        gk = g(x)
        m = np.block([[hlk, jk.T], [jk, np.zeros([3, 3])]])
        b = np.block([[-dfk], [-gk]])
        ae = solve(m, b)
        a = ae[:5]
        e = ae[5:]
        b = e - l
        y = np.block([[b], [a]])
        v = y + v
        k = k + 1
    return v

def sqp_method_analyse_convergence(v0, df, dl, hl, dg, g, k_max):
    v = v0
    k = 0
    err = list()
    err0 = norme(dl(v))
    t0 = time.time()
    while (eps < norme(dl(v))) and (k < k_max):
        l = v[:3]
        x = v[3:]
        err.append(norme(dl(v)) / err0)
        hlk = hl(v)[-5:, -5:]
        dfk = df(x)
        jk = dg(x)
        gk = g(x)
        m = np.block([[hlk, jk.T], [jk, np.zeros([3, 3])]])
        b = np.block([[-dfk], [-gk]])
        ae = solve(m, b)
        a = ae[:5]
        e = ae[5:]
        b = e - l
        y = np.block([[b], [a]])
        v = y + v
        k = k + 1
    tt = time.time()
    print("nombre d'itération : " + str(k) + ", avec temps de calculation : " + str(tt - t0) + " seconds.")
    plt.plot(np.log([x + 1 for x in range(k)]), np.log(err))
    plt.show()
    return v

def sqp(l0, x0, k_max, mode: bool = False):
    print("Appliquier SQP pour condition initiale suivante : ")
    print("x0 : ")
    print(x0)
    print("l0 : ")
    print(l0)
    print("En cours de calculation !")
    v0 = np.block([[l0], [x0]])
    if (mode):
        sol = sqp_method_analyse_convergence(v0= v0, df = grad_f, dl = d_lagrange, hl = h_lagrange, dg = jacob_g, g = g, k_max = k_max)
    else:
        sol = sqp_method(v0= v0, df = grad_f, dl = d_lagrange, hl = h_lagrange, dg = jacob_g, g = g, k_max = k_max)
    l_sol = sol[:3]
    x_sol = sol[3:]
    print("Calculation est terminée !")
    print("min de f est : " + str(f(x_sol)))
    print("x_sol est : ")
    print(str(x_sol))
    print("lambda_sol est : ")
    print(l_sol)
    print("Pour vérification de la condition contrainte, g(x_sol) = ")
    print(g(x_sol))

x0 = np.expand_dims([-1.71, 1.59, 1.82, -0.763, -0.763], axis=0).T
x1 = np.expand_dims([-1.9, 1.82, 2.02, -0.9, -0.9], axis=0).T
x2 = np.expand_dims([1, 0, 3, 0, 0], axis=0).T
l0 = np.expand_dims([0, 0, 0], axis=0).T
k_max = 15000

newton(l0, x0, k_max, True)
sqp(l0, x0, k_max, True)
newton(l0, x1, k_max, True)
sqp(l0, x1, k_max, True)
newton(l0, x2, k_max, True)
sqp(l0, x1, k_max, True)

x10 = np.expand_dims([-1.7040916100093402,1.5802605567239958,2.4244592010611106,-1.0164078958300446,-1.0164078958348974], axis=0).T
newton(l0, x10, k_max, True)
sqp(l0, x10, k_max, True)

# print(grad_f(x0))
# print(hessian_f(x0))
# param_hf(x0, 0, 1)
# param_hf(x0, 4, 1)
# print(d_prod(x0,0,1))
# print(f(x10))
# print(g(x10))
# print(d_g(x0,1,1))
