
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import sys




def read_fadata(fname):
   a = []
   b = []
   c = []
   d = []
   xpos = []
   ypos = []

   linenum = 0;
   maxpts = 500000;

   with open(fname,'r') as fin:
     for line in fin:
        linenum = linenum + 1;
        data = line.strip('\n').split()
        if (len(data)!=6): 
            print("Error in line %d" % linenum)
            break;
        elif (linenum > maxpts):
            break;
        else:        
           a.append(float(data[0]))
           b.append(float(data[1]))
           c.append(float(data[2]))
           d.append(float(data[3]))
           xpos.append(float(data[4]))
           ypos.append(float(data[5]))


   a = np.array(a, dtype=np.float32) / 2**31
   b = np.array(b, dtype=np.float32) / 2**31
   c = np.array(c, dtype=np.float32) / 2**31
   d = np.array(d, dtype=np.float32) / 2**31
   xpos = np.array(xpos, dtype=np.float32) / 1000  #scale to um
   ypos = np.array(ypos, dtype=np.float32) / 1000  #scale to um
  
   
   # Assuming a, b, c, d, xpos, ypos are 1D arrays of the same length
   n = len(a)
   fa_data = np.zeros((n, 6), dtype=np.float32)  # create empty 2D array with 6 columns
   print("Number of Samples = %d" % n)
   fa_data[:,0] = a
   fa_data[:,1] = b
   fa_data[:,2] = c
   fa_data[:,3] = d
   fa_data[:,4] = xpos
   fa_data[:,5] = ypos 
   
   return fa_data 


def calcPSD(x, fs=None, nperseg=None): # input format: rows for number of sample, columns for number of bpm
    x = np.transpose(x)
    if nperseg is None:
        if x.ndim == 1: nperseg = x.shape[0]
        elif x.ndim == 2: _, nperseg = x.shape

    if fs is None:
        # contants for SR
        f_rf_sr = 499.68e6 # rf frequency
        h_sr = 1320
        f_rev_sr = f_rf_sr/h_sr
        dec_fa_sr = 38
        fs = f_rev_sr/dec_fa_sr # FA data ~10 kHz

    f, psd = welch(x, fs=fs, nperseg=nperseg)

    return f, np.transpose(psd)


def calc_psd_and_int_psd(x, y, fs=None, nperseg=None, fstart=0, fstop=300000): # input x, y in [um]
    f0 = fstart # [Hz] initial freq for plot
    f1 = fstop # [Hz] final freq for plot

    x = np.array(x)
    y = np.array(y)

    f, psd_x = calcPSD(x, fs=fs, nperseg=nperseg)
    _, psd_y = calcPSD(y, fs=fs, nperseg=nperseg)

    # calc int. psd
    df = np.mean(np.diff(f))

    i0 = np.where(f>=f0)[0][0]
    if f1 > f[-1]: f1 = f[-1]
    i1 = np.where(f>=f1)[0][0]

    # recon data
    f = f[i0:i1]
    psd_x = psd_x[i0:i1]
    psd_y = psd_y[i0:i1]

    # integrated PSD
    if psd_x.ndim == 1:
        int_psd_x = np.sqrt(np.cumsum(psd_x) * df)
        int_psd_y = np.sqrt(np.cumsum(psd_y) * df)
    else:
        int_psd_x = np.sqrt(np.cumsum(psd_x, axis=0) * df)
        int_psd_y = np.sqrt(np.cumsum(psd_y, axis=0) * df)

    return f, psd_x, psd_y, int_psd_x, int_psd_y

def plot_psd_int_psd(f, psd_x, psd_y, int_psd_x, int_psd_y,
                     *args, label=None, axes=None, units='units', yscale='lin', 
                     sqrt_psd=False, plot_int_psd=True,
                     **plot_kwargs):
    if axes is None:
        if plot_int_psd:
            fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex='col')
        else:
            fig, axes = plt.subplots(2, 1, figsize=(10, 5))
    axes = axes.ravel()
    
    psd_y_label = f'PSD Y [{units}$^2$/Hz]'
    psd_x_label = f'PSD X [{units}$^2$/Hz]'
    if sqrt_psd:
        psd_x = np.sqrt(psd_x)
        psd_y = np.sqrt(psd_y)
        psd_x_label = rf'$\sqrt{{\mathrm{{PSD}} X}}$ [{units}/$\sqrt{{\mathrm{{Hz}}}}$]'
        psd_y_label = rf'$\sqrt{{\mathrm{{PSD}}Y}}$ [{units}/$\sqrt{{\mathrm{{Hz}}}}$]'

    # PSD X
    lines = axes[0].plot(f, psd_x, *args, **plot_kwargs)
    if label:
        lines[0].set_label(label)
    for line in lines[1:]:
        line.set_label('_nolegend_')
    if label is not None:
        axes[0].legend()
    axes[0].set_xscale('log')
    axes[0].set_ylabel(psd_x_label)
    axes[0].grid(which='both', linestyle='--', linewidth=0.5)
    if yscale == 'log':
        axes[0].set_yscale('log')
    elif yscale == 'lin':   
        axes[0].set_yscale('linear')
    else:
        raise ValueError(f"Invalid yscale: {yscale}. Use 'log' or 'lin'.")
    axes[0].set_title('Horizontal')

    # PSD Y
    lines = axes[1].plot(f, psd_y, *args, **plot_kwargs)
    if label:
        lines[0].set_label(label)
    for line in lines[1:]:
        line.set_label('_nolegend_')
    axes[1].set_xscale('log')
    axes[1].set_ylabel(psd_y_label)
    axes[1].grid(which='both', linestyle='--', linewidth=0.5)
    if yscale == 'log':
        axes[1].set_yscale('log')
    elif yscale == 'lin':   
        axes[1].set_yscale('linear')
    else:
        raise ValueError(f"Invalid yscale: {yscale}. Use 'log' or 'lin'.")
    axes[1].set_title('Vertical')

    if plot_int_psd:
        # Integrated PSD X
        axes[2].plot(f, int_psd_x, *args, **plot_kwargs)
        axes[2].set_xscale('log')
        axes[2].set_ylabel(f'Int. PSD X [{units}]')
        axes[2].set_xlabel('Frequency [Hz]')
        axes[2].grid(which='both', linestyle='--', linewidth=0.5)

        # Integrated PSD Y
        axes[3].plot(f, int_psd_y, *args, **plot_kwargs)
        axes[3].set_xscale('log')
        axes[3].set_ylabel(f'Int. PSD Y [{units}]')
        axes[3].set_xlabel('Frequency [Hz]')
        axes[3].grid(which='both', linestyle='--', linewidth=0.5)

    return axes

if __name__ == "__main__":
    # Example usage
    Frf = 499.68e6
    h = 1320
    Ftbt = Frf/h
    fs = Ftbt/38  #FA Sampling Rate

    
    if len(sys.argv) != 2:
       print ("Missing input file...")
       sys.exit() 
    else:
       fname = sys.argv[1]


    fa_data=read_fadata(fname)
    x = fa_data[:,4]
    y = fa_data[:,5]
    

    print("Std X: %.3f um" % (np.std(x)))
    print("Std Y: %.3f um" % (np.std(y)))
    

    f, psd_x, psd_y, int_psd_x, int_psd_y = calc_psd_and_int_psd(x, y, fs)
    plot_psd_int_psd(f, psd_x, psd_y, int_psd_x, int_psd_y, label='Test', units='um')
    
    # Print total integrated power
    print(f"Total integrated power X: {int_psd_x[-1]:.3f}")
    print(f"Total integrated power Y: {int_psd_y[-1]:.3f}")
    
    
    plt.show()
