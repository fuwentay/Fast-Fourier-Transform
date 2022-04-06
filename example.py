import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

time_step = 0.05
time_vec = np.arange(0, 10, time_step)
period = 5

sig = (np.sin(2*np.pi*time_vec/period)) + 0.25*np.random.randn(time_vec.size)

plt.plot(time_vec, sig)
#plt.show()

# Round off
#print(np.round(sig,2))

sig_fft = fftpack.fft(sig)

### Returns complex "list"
#print(sig_fft)

Amplitude = np.abs(sig_fft)
Power = Amplitude**2
Angle = np.angle(sig_fft)

sample_freq = fftpack.fftfreq(sig.size, d=time_step)

Amp_Freq = np.array([Amplitude, sample_freq])

Amp_position = Amp_Freq[0,:].argmax()
peak_freq = Amp_Freq[1, Amp_position]

high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq) > peak_freq] = 0
filtered_sig = fftpack.ifft(high_freq_fft)

# Unfiltered FFT
# plt.plot(sample_freq, Amplitude)
# plt.show()

# Filtered FFT
#print(high_freq_fft)
filtered_Amplitude = np.abs(high_freq_fft)
#plt.plot(sample_freq, filtered_Amplitude)
#plt.show()

# Filtered Signal
plt.plot(time_vec, filtered_sig)
plt.show()