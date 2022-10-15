import numpy as np
import sys

eps = sys.float_info.epsilon
norme = np.linalg.norm
solve = np.linalg.solve
print(eps)


def newtown(x0, df, hf, kMax):
    x = x0
    k = 0
    while (eps < norme(df(x))) and (k < kMax):
        # Xk+1 = Xk - inv(H)*df
        # H(Xk+1 -Xk) = -df = H(y)
        # Xk+1 = y + Xk
        y = solve(hf(x), -df(x))
        x = y + x
        k = k + 1
    return x
