import scipy
import scipy.signal
import numpy as np

def stft(x, framesamp=256, hopsamp=64):
    w = scipy.signal.hanning(framesamp, False)
    w = np.sqrt(w)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp+1, hopsamp)])
    return X

def istft(X,  T, hopsamp=64):
    x = scipy.zeros(T)
    weights = scipy.zeros(T)
    framesamp = X.shape[1]
    w = scipy.signal.hanning(framesamp, False)
    w = np.sqrt(w)

    for n,i in enumerate(range(0, len(x)-framesamp+1, hopsamp)):
        x[i:i+framesamp] += w*scipy.real(scipy.ifft(X[n]))
        weights[i:i+framesamp] += w**2

    weights[weights==0] = 1
    x = x/weights

    return x
