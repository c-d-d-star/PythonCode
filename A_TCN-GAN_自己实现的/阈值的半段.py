import pandas as pd
import numpy as np
import torch

cnt=[]
thead=torch.tensor(0)
data=torch.randn(10,4,1)
for i,y in enumerate(data,0):
    # print(y[-1,:])
    if y[-1,:]>thead:
        cnt.append(y)
print(len(cnt))
