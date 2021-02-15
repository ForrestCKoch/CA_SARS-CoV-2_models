import numpy as np

def get_reproduction_number(beta=0.115,S=1):

    F = np.zeros(shape=(28,28))
    F[0,0:16] = 0.2*beta*S
    F[0,16:18] = 1.2*beta*S
    F[0,18:] = beta * S


    V = np.eye(28)
    for i in range(1,28):
        v = -1
        if i == 4:
            v = -0.2
            V[i,i-1] = v
        elif i == 16:
            v = -0.8
            V[i,3] = v
        else:
            V[i,i-1] = v

    w,v = np.linalg.eigh(F @ np.linalg.inv(V))
    return w.max()

if __name__ == '__main__':
    print(get_reproduction_number())
