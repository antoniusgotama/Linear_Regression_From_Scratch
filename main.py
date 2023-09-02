import pandas as pd
from tqdm import tqdm


def loadData(data):
    data = pd.read_csv(data)
    x, y = data.iloc[:, 0], data.iloc[:, 1]
    return x, y


def optimimizeGradien(x, y, lc=0.00001, epoch=100000):
    m = 0
    c = 0
    for i in tqdm(range(epoch)):
        sum_grad_m, sum_grad_c = 0, 0
        for k in range(len(x)):
            sum_grad_m += 2*x[k]*(y[k]-m*x[k]-c)/len(x)
            sum_grad_c += 2*(y[k] - m*x[k] - c)/len(x)
        m = m + lc * sum_grad_m
        c = c + lc * sum_grad_c
    return m, c


def printingError(m, c, x, y):
    sumoferror = 0
    for i in range(len(x)):
        sumoferror += (y[i] - (m*x[i] + c))**2
    return sumoferror

if __name__ == "__main__":
    x, y = loadData("data.csv")
    m_optimize, c_optimize = optimimizeGradien(x, y)
    print(f"the equation will be y = {m_optimize}x+{c_optimize}")
    print(printingError(m_optimize, c_optimize, x, y))
