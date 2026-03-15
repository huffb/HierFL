import numpy as np
import torch
def Gaussian_Simple(epsilon, delta, sensitivity, size):
    noise_scale = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
    return np.random.normal(0, noise_scale, size=size)

def Model_Noise_Add(delta,sepsilon,depsilon,model,w,sensitivity):
    if(model == 'lenet' or model == 'cnn_complex'):
        for name, param in w:
            if 'conv' in name:
                epsilon = sepsilon
            else:
                epsilon = depsilon
            noise = Gaussian_Simple(epsilon, delta, sensitivity, size=param.data.size())
            param.data += torch.tensor(noise)



