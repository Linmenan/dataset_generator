import numpy as np

def composite_simpson(f, a, b, n):
    """
    使用复化辛普森方法近似计算函数 f 在区间 [a, b] 上的积分。
    参数 n 必须为偶数，此处若 n 为奇数则自动加 1。
    """
    if n % 2 == 1:
        n += 1
    h = (b - a) / n
    x_vals = np.linspace(a, b, n+1)
    y_vals = f(x_vals)
    S = y_vals[0] + y_vals[-1] + 4 * np.sum(y_vals[1:-1:2]) + 2 * np.sum(y_vals[2:-2:2])
    return h/3 * S

def offset_poly(s,s_offset,a,b,c,d):
    return a + b*(s-s_offset) + c*(s-s_offset)**2 + d*(s-s_offset)**3