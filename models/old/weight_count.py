import numpy as np
import torch

def main():
    path = input('path... ')

    dic = torch.load(path, map_location='cpu')
    lis = [(n, np.prod(list(p.shape))) for n, p in dic.items()]
    lis.sort(key=lambda x: x[1])
    for n, p in lis:
        print(n, p)

if __name__ == "__main__":
    main()
