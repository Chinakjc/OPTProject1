import numpy as np
import sys

eps = sys.float_info.epsilon
norme = np.linalg.norm
solve = np.linalg.solve
print(eps)


def newton(v0, df, hf, k_max):
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


def f(x):
    x1 = x[0]
    x2 = x[1]
    return np.exp(np.prod(x)) - ((x1 ** 3 + x2 ** 3 + 1) ** 2) / 2.0


def s_prod(x, i):
    y0 = np.prod(x)
    y1 = np.prod(x[:i])
    y2 = np.prod(x[i:])
    return (y1 * y2) * np.exp(y0)


def d_prod(x, i, j):
    y0 = np.prod(x)
    y1 = np.prod(x[:i]) * np.prod(x[i:])
    y2 = np.prod(x[:j]) * np.prod(x[j:])
    return (y1 * y2) * np.exp(y0)


def grad_f(x):
    res = list()
    for i in range(5):
        res.append(s_prod(x, i))
    res = np.expand_dims(res,axis=0).T
    x1 = x[0]
    x2 = x[1]
    r1 = - 3 * x1 ** 5 - 3 * (x2 ** 3 + 1) ** 2
    r2 = - 3 * x2 ** 5 - 3 * (x1 ** 3 + 1) ** 2
    b1 = np.block([[r1,np.zeros(4)]]).T
    b2 = np.block([[np.zeros(1),r2,np.zeros(3)]]).T
    return res + b1 + b2


def hessian_f(x):
    res = list()
    for i in range(5):
        l = list()
        for j in range(5):
            l.append(d_prod(x, i, j))
        res.append(l)
    res = np.array(res)
    x1 = x[0]
    x2 = x[1]
    res[0][0] = res[0][0] - 15 * x1 ** 4
    res[0][1] = res[0][1] - 18 * x2 ** 5 - 18 * x2 ** 2
    res[1][0] = res[1][0] - 18 * x1 ** 5 - 18 * x1 ** 2
    res[1][1] = res[1][1] - 15 * x2 ** 4
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
    return np.block([[line1],[line2],[line3]])


def hessian_g_product_lambda(l, x):
    l1 = l[0]
    l2 = l[1]
    l3 = l[2]
    x1 = x[0]
    x2 = x[1]
    line1 = np.block([2 * l1 + 6 * l3 * x1, 0, 0, 0, 0])
    line2 = np.block([0, 2 * l1 + 6 * l3 * x2, l2, 0, 0])
    line3 = np.block([0, l2, 2 * l1, 0, 0])
    line4 = np.block([0, 0, 0, 2 * l1, -5 * l2])
    line5 = np.block([0, 0, 0, -5 * l2, 2 * l1])
    return np.block([[line1], [line2], [line3], [line4], [line5]])


def lagrange_f_g(l, x):
    return f(x) - np.dot(l.T, g(x))


def grad_lagrange_f_g(l, x):
    return np.block([[g(x)],[grad_f(x) + np.dot(jacob_g(x).T, l)]])


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

def get_f_valeur(v):
    return f(v[3:])

x0 = np.expand_dims([-1.71,1.59,1.82,-0.763,-0.763],axis=0).T
x1 = np.expand_dims([-1.9,1.82,2.02,-0.9,-0.9],axis=0).T
x2 = np.expand_dims([1,0,3,0,0],axis=0).T
l0 = np.expand_dims([0,1,0],axis=0).T
v0 = np.block([[l0], [x0]])
v1 = np.block([[l0], [x1]])
v2 = np.block([[l0], [x2]])
k_max = 10000
print(get_f_valeur(newton(v0, d_lagrange, h_lagrange, k_max)))
print(get_f_valeur(newton(v1, d_lagrange, h_lagrange, k_max)))
print(get_f_valeur(newton(v2, d_lagrange, h_lagrange, k_max)))
'''print(jacob_g(x0))
print(norme(jacob_g(x0)))'''
