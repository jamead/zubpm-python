import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirdesign, freqz

# Given parameters
fs = 117.3491e6  # Sampling frequency in Hz
nyquist = fs / 2  # Nyquist frequency

# Bandpass specifications (normalized to Nyquist frequency)
passband = [30.2e6 / nyquist, 30.5e6 / nyquist]  # Passband in normalized frequency
stopband = [(30.2e6 - 100e3) / nyquist, (30.5e6 + 100e3) / nyquist]  # Stopband with transition band

# Attenuation and ripple specifications
gpass = 1  # Maximum ripple in passband (dB)
gstop = 60  # Minimum attenuation in stopband (dB)

# Design the IIR filter using elliptic filter (other options: Butterworth, Chebyshev)
b, a = iirdesign(wp=passband, ws=stopband, gpass=gpass, gstop=gstop, ftype='ellip')

# Frequency response of the designed filter
w, h = freqz(b, a, worN=8000, fs=fs)

# Plot the frequency response
plt.figure(figsize=(10, 6))
plt.plot(w / 1e6, 20 * np.log10(np.abs(h)), label="Frequency Response")
plt.axvline(30.2, color='green', linestyle='--', label="Passband Start (30.2 MHz)")
plt.axvline(30.5, color='green', linestyle='--', label="Passband End (30.5 MHz)")
plt.axvline((30.2 - 0.1), color='red', linestyle='--', label="Stopband Start")
plt.axvline((30.5 + 0.1), color='red', linestyle='--', label="Stopband End")
plt.title("IIR Bandpass Filter Frequency Response")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude (dB)")
plt.ylim(-80, 5)
plt.grid()
plt.legend()
plt.show()

# Print filter coefficients
print("Filter Coefficients:")
print(f"Numerator (b): {b}")
print(f"Denominator (a): {a}")

