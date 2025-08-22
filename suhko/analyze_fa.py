import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal



Frf = 499.68e6
h = 1320
Ftbt = Frf/h
kx=10e-3
ky=10e-3
Ffa = Ftbt/38




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
  
   fasum = a+b+c+d
   print("Read %d FA pts" % linenum)
   calcxpos = kx * 1e6 * (((a+d)-(b+c))/fasum)
   calcypos = ky * 1e6 * (((a+b)-(c+d))/fasum)
   print
   print("python Xpos std: %f um" % (np.std(calcxpos)))
   print("python Xpos mean: %f um" % (np.mean(calcxpos)))
   print("python Ypos std: %f um" % (np.std(calcypos)))
   print("python Ypos mean: %f um" % (np.mean(calcypos)))
   print
   print("fpga Xpos std: %f um" % (np.std(xpos)))
   print("fpga Xpos mean: %f um" % (np.mean(xpos)))
   print("fpga Ypos std: %f um" % (np.std(ypos)))
   print("fpga Ypos mean: %f um" % (np.mean(ypos)))
   
   # Assuming a, b, c, d, xpos, ypos are 1D arrays of the same length
   n = len(a)
   fa_data = np.zeros((n, 6), dtype=np.float32)  # create empty 2D array with 6 columns
   print("Length = %d" % n)
   fa_data[:,0] = a
   fa_data[:,1] = b
   fa_data[:,2] = c
   fa_data[:,3] = d
   fa_data[:,4] = xpos
   fa_data[:,5] = ypos
   
   
   
   return fa_data 





def calc_psd(y):
   N = len(y)  
   f = np.linspace(0,Fs/2,N/2+1)
   #print ("len(t)=%f" % len(adc_data))
   #print ("len(f)=%f" % len(f))
   #print ("Total Time=%f" % (Ts*N))

   y = y - np.mean(y)
   print ("len(y)=%f" % len(y))
   w = np.hanning(N)
   x = w * y
   xfft = np.abs(np.fft.rfft(x)) / (N/2.0)

   p = 20*np.log10(xfft)
   return y,p


def plot_psd(a,b,c,d):
   N = len(a)*2-1
   f = np.linspace(0,Fs/2,N/2+1)
   f = f/1e6 #scale to MHz  
   ylim = [-160,0]
   fig,axes = plt.subplots(nrows=2,ncols=2)
   ax1=plt.subplot(221)
   plt.plot(f,a,'b')
   plt.ylabel('dBFS')
   ax1.set_ylim(ylim)
   #plt.xlabel('freq (MHz)')
   plt.title('PSD ChA')
   plt.grid()
   ax2=plt.subplot(222, sharex=ax1)
   plt.plot(f,b,'b')
   plt.ylabel('dBFS')
   ax2.set_ylim(ylim)
   #plt.xlabel('freq (MHz)')
   plt.title('PSD ChB')
   plt.grid()
   ax3=plt.subplot(223, sharex=ax1)
   plt.plot(f,c,'b')
   plt.ylabel('dBFS')
   plt.xlabel('freq (MHz)')
   plt.title('PSD ChC')
   ax3.set_ylim(ylim)
   plt.grid()
   ax4=plt.subplot(224, sharex=ax1)
   plt.plot(f,d,'b')
   plt.ylabel('dbFS')
   plt.xlabel('freq (MHz)')
   plt.title('PSD ChD')
   plt.grid()
   ax4.set_ylim(ylim)
   titlestr = "ADC PSD   " + "numpts=" + str(2*(len(f)-1))
   fig.suptitle(titlestr)




def plot_fa(fa_data):

    a = fa_data[:,0] / (2**29)
    b = fa_data[:,1] / (2**29)
    c = fa_data[:,2] / (2**29)
    d = fa_data[:,3] / (2**29)
    x = fa_data[:,4] # in um
    y = fa_data[:,5] # in um
    print("Len of a: %d" % len(a))

    # 3 rows × 2 cols = 6 plots
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 8))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # Channel A
    ax1.plot(a, 'b-')
    ax1.set_ylabel('FA ChA')
    #ax1.set_title(r'%FS=%4.3f   $\sigma$=%4.3f um' % (np.mean(a), np.std(a)))
    ax1.grid(True)

    # Channel B
    ax2.plot(b, 'b-')
    ax2.set_ylabel('FA ChB')
    #ax2.set_title(r'$\mu$=%4.3f um  $\sigma$=%4.3f um' % (np.mean(b), np.std(b)))
    ax2.grid(True)

    # Channel C
    ax3.plot(c, 'b-')
    ax3.set_ylabel('FA ChC')
    #ax3.set_title(r'$\mu$=%4.3f um  $\sigma$=%4.3f um' % (np.mean(c), np.std(c)))
    ax3.grid(True)

    # Channel D
    ax4.plot(d, 'b-')
    ax4.set_ylabel('FA ChD')
    #ax4.set_title(r'$\mu$=%4.3f um  $\sigma$=%4.3f um' % (np.mean(d), np.std(d)))
    ax4.grid(True)
    
    # X Position
    ax5.plot(x, 'b-')
    ax5.set_ylabel('FA XPos (um)')
    ax5.set_xlabel('Sample #')
    ax5.set_title(r'$\mu$=%4.3f um  $\sigma$=%4.3f um' % (np.mean(x), np.std(x)))
    ax5.grid(True)

    # Y Position
    ax6.plot(y, 'b-')
    ax6.set_ylabel('FA YPos (um)')
    ax6.set_xlabel('Sample #')
    ax6.set_title(r'$\mu$=%4.3f um  $\sigma$=%4.3f um' % (np.mean(y), np.std(y)))
    ax6.grid(True)

    # Overall title
    fig.suptitle("FA Position", fontsize=14)

    # Fix layout so titles/labels don’t overlap
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Print summary
    print("Xpos mean: %f um \tstd: %f um" % (np.mean(x), np.std(x)))
    print("Ypos mean: %f um \tstd: %f um" % (np.mean(y), np.std(y)))

    plt.show(block=False)

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def plot_psd(fa_data, Ffa):
    """
    Plot PSD in dB and integrated PSD for X and Y channels from fa_data.

    Parameters:
        fa_data : 2D array (num_samples x 6)
                   Columns: a,b,c,d,xpos,ypos
        Ffa     : sampling frequency (Hz)
    """
    numpts = len(fa_data[:,0])
    print("NumPts: %d" % numpts)
    df = Ffa / numpts
    xpos = fa_data[:,4]
    ypos = fa_data[:,5]

    # Compute PSD
    f, psdX = signal.periodogram(xpos, Ffa, 'flattop', scaling='density')
    f, psdY = signal.periodogram(ypos, Ffa, 'flattop', scaling='density')

    # Convert PSD to dB for better visualization
    psdX_db = 10 * np.log10(psdX)
    psdY_db = 10 * np.log10(psdY)

    # Compute integrated PSD (cumulative RMS)
    cs_psdX = np.sqrt(np.cumsum(psdX * df))
    cs_psdY = np.sqrt(np.cumsum(psdY * df))

    # Create figure and axes
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10,8), sharex=True)

    # PSD plot (dB) with log-x axis
    axs[0].semilogx(f, psdX_db, label='X')
    axs[0].semilogx(f, psdY_db, label='Y')
    axs[0].set_ylabel('PSD [dB $\mu m^2$/Hz]')
    axs[0].set_title('FA PSD')
    axs[0].grid(True)
    axs[0].legend(loc=2)

    # Integrated PSD plot
    axs[1].semilogx(f, cs_psdX, '.-', label='X')
    axs[1].semilogx(f, cs_psdY, '.-', label='Y')
    axs[1].set_xlabel('Frequency [Hz]')
    axs[1].set_ylabel('Integrated PSD [$\mu m$]')
    axs[1].set_title('FA Integrated PSD')
    axs[1].grid(True)
    axs[1].legend(loc=2)

    plt.tight_layout()
    plt.show(block=False)

    # Print total integrated power
    print(f"Total integrated power X: {cs_psdX[-1]:.6e}")
    print(f"Total integrated power Y: {cs_psdY[-1]:.6e}")





def main():

   plt.ion()
   if len(sys.argv) != 2:
       print ("Missing input file...")
       sys.exit() 
   else:
       fname = sys.argv[1]


   fa_data=read_fadata(fname)
   plot_fa(fa_data)
   
   plot_psd(fa_data,Ffa)
   

   
   plt.show()
   input('Press any key to quit...')




if __name__ == "__main__":
    main()

