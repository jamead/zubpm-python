import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, kaiserord

# Given parameters
fs = 117.3491e6  # Sampling frequency in Hz
passband = [30.2e6, 30.5e6]  # Passband frequencies in Hz
transition_band = 200e3  # Transition band in Hz
attenuation = 30  # Desired attenuation in dB

# Normalized frequencies (relative to Nyquist frequency)
nyquist = fs / 2
low_cutoff = (passband[0] - transition_band) / nyquist
high_cutoff = (passband[1] + transition_band) / nyquist

# Calculate filter order using the Kaiser window method
width = transition_band / nyquist  # Transition width in normalized frequency
numtaps, beta = kaiserord(attenuation, width)

# Design the FIR filter
filter_taps = firwin(numtaps, [low_cutoff, high_cutoff], pass_zero=False, window=('kaiser', beta))

# Frequency response of the filter
w, h = freqz(filter_taps, worN=8000, fs=fs)

# Plot the frequency response
plt.figure(figsize=(10, 6))
plt.plot(w / 1e6, 20 * np.log10(np.abs(h)), label="Frequency Response")
plt.axvline(passband[0] / 1e6, color='green', linestyle='--', label="Passband Start")
plt.axvline(passband[1] / 1e6, color='green', linestyle='--', label="Passband End")
plt.axvline((passband[0] - transition_band) / 1e6, color='red', linestyle='--', label="Transition Start")
plt.axvline((passband[1] + transition_band) / 1e6, color='red', linestyle='--', label="Transition End")
plt.title("Bandpass FIR Filter Frequency Response")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude (dB)")
plt.ylim(-80, 5)
plt.grid()
plt.legend()
plt.show()

# Print filter details
print(f"Filter Order: {numtaps - 1}")
print(f"Cutoff Frequencies (normalized): [{low_cutoff:.6f}, {high_cutoff:.6f}]")

