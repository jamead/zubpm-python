import numpy as np
from scipy.signal import firwin, freqz
import matplotlib.pyplot as plt

# Filter specifications
fs = 117.3491e6       # Sample rate (Hz)
cutoff = 800e3   # Desired cutoff frequency (Hz)
numtaps = 101     # Number of filter coefficients (taps)

# Design the FIR filter
coefficients = firwin(numtaps, cutoff, fs=fs)

# Compute the frequency response
w, h = freqz(coefficients, worN=8000)

# Plot the frequency response
plt.figure(figsize=(10, 6))
plt.plot(w * fs / (2 * np.pi), 20 * np.log10(np.abs(h)), 'b')
plt.title('FIR Filter Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()
plt.show()


# Export coefficients to a Xilinx COE file
coe_header = (
    "; FIR Filter Coefficients\n"
    "radix=10;\n"
    "coefdata=\n"
)
coe_body = ',\n'.join(f"{coef:.6f}" for coef in coefficients) + ';'
coe_content = coe_header + coe_body

with open("fir_filter.coe", "w") as coe_file:
    coe_file.write(coe_content)



