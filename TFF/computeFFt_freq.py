#https://www.youtube.com/watch?v=O0Y8FChBaFU&ab_channel=1Mviews
import numpy as np
import matplotlib.pyplot as plt

# 1- Compute FFT using Numpy 

# Contruct a time signal
Fs = 5_000 # Hz, sampling freq
tstep = 1/Fs # sample time interval

f0 = 100 # signal freq (cada t=1 entra esta fercuencia = data estas vueltas)
N = int(10 * Fs / f0) # # of sample

t = np.linspace(0, (N-1)*tstep, N) # time steps
fstep = Fs / N #freq interval
f = np.linspace(0, (N-1)*fstep, N) # freq steps

y = 1 * np.sin(2 * np.pi * f0 * t) + 0.5 * np.sin(2 * np.pi * (f0+543) * t)+ np.random.rand(N)#our signal

# Perform ff
X = np.fft.fft(y) # serie num. complejos
X_mag = np.abs(X) / N # normalizamos con N


f_plot = f[0:int(N/2+1)]
X_mag_plot = 2* X_mag[0:int(N/2+1)]
X_mag_plot[0] = X_mag_plot[0] / 2 #Note: DC compon

# 2- Plot the fr equency specturm using matplotlib
fig, [ax1,ax2] = plt.subplots(2,1)
ax1.plot(t,y,'-')
ax1.set_xlabel('Time')
ax1.set_ylabel('Intensity')

ax2.plot(f_plot,X_mag_plot,'-')
ax2.set_xlabel('Frequancy')
ax2.set_ylabel('Maginitud')

plt.show()

# Dicover the frequencies  = the f0 ? and the 1* or 0.5* in front !!! 