import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, kaiserord, resample_poly

# Given parameters
fs = 117.3491e6  # Sampling frequency in Hz
passband = [30.2e6, 30.5e6]  # Passband frequencies in Hz
transition_band = 100e3  # Transition band in Hz
attenuation = 60  # Desired attenuation in dB

# Normalized frequencies (relative to Nyquist frequency)
nyquist = fs / 2
low_cutoff = (passband[0] - transition_band) / nyquist
high_cutoff = (passband[1] + transition_band) / nyquist

# Transition width and filter order
width = transition_band / nyquist  # Transition width in normalized frequency
numtaps, beta = kaiserord(attenuation, width)

# Design the FIR filter
filter_taps = firwin(numtaps, [low_cutoff, high_cutoff], pass_zero=False, window=('kaiser', beta))

# Define the number of polyphase sub-filters (phases)
num_phases = 4  # You can tune this based on your system's requirements
poly_filters = np.reshape(filter_taps, (num_phases, -1), order='C')

# Plot frequency response of each phase
plt.figure(figsize=(12, 8))
for i, phase in enumerate(poly_filters):
    w, h = freqz(phase, worN=8000, fs=fs / num_phases)
    plt.plot(w / 1e6, 20 * np.log10(np.abs(h)), label=f"Phase {i + 1}")

# Aggregate frequency response of the entire filter
combined_response = np.zeros(8000, dtype=complex)
for i, phase in enumerate(poly_filters):
    combined_response += freqz(phase, worN=8000, fs=fs / num_phases)[1]

plt.plot(w / 1e6, 20 * np.log10(np.abs(combined_response)), 'k--', label="Overall Response", linewidth=2)

# Add labels and show the plot
plt.title("Polyphase Bandpass FIR Filter Frequency Response")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude (dB)")
plt.ylim(-80, 5)
plt.grid()
plt.legend()
plt.show()

# Print filter details
print(f"Filter Order: {numtaps - 1}")
print(f"Cutoff Frequencies (normalized): [{low_cutoff:.6f}, {high_cutoff:.6f}]")
print(f"Number of Polyphase Filters: {num_phases}")

