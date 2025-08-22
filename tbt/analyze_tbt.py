import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal



Frf = 499.68e6
Fin = 499.68e6
h = 1320
hADC = 310
hIf = 80
Fs = Frf/h*hADC
Ftbt = Frf/h
Ts =  1/Fs
adcFS = 32768.0
#N = 100000 
numTurns = 500
kx=10e-3
ky=10e-3
Ftbt = Frf/h




def read_tbtdata(fname):
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
  
   tbtsum = a+b+c+d
   print("Read %d turns" % linenum)
   calcxpos = kx * 1e6 * (((a+d)-(b+c))/tbtsum)
   calcypos = ky * 1e6 * (((a+b)-(c+d))/tbtsum)
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
   tbt_data = np.zeros((n, 6), dtype=np.float32)  # create empty 2D array with 6 columns
   print("Length = %d" % n)
   tbt_data[:,0] = a
   tbt_data[:,1] = b
   tbt_data[:,2] = c
   tbt_data[:,3] = d
   tbt_data[:,4] = xpos
   tbt_data[:,5] = ypos
   
   
   
   return tbt_data 





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




def plot_tbt(tbt_data):

    a = tbt_data[:,0] / (2**29)
    b = tbt_data[:,1] / (2**29)
    c = tbt_data[:,2] / (2**29)
    d = tbt_data[:,3] / (2**29)
    x = tbt_data[:,4] # in um
    y = tbt_data[:,5] # in um
    print("Len of a: %d" % len(a))

    # 3 rows × 2 cols = 6 plots
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(10, 8))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # Channel A
    ax1.plot(a, 'b-')
    ax1.set_ylabel('TbT ChA')
    #ax1.set_title(r'%FS=%4.3f   $\sigma$=%4.3f um' % (np.mean(a), np.std(a)))
    ax1.grid(True)

    # Channel B
    ax2.plot(b, 'b-')
    ax2.set_ylabel('TbT ChB')
    #ax2.set_title(r'$\mu$=%4.3f um  $\sigma$=%4.3f um' % (np.mean(b), np.std(b)))
    ax2.grid(True)

    # Channel C
    ax3.plot(c, 'b-')
    ax3.set_ylabel('TbT ChC')
    #ax3.set_title(r'$\mu$=%4.3f um  $\sigma$=%4.3f um' % (np.mean(c), np.std(c)))
    ax3.grid(True)

    # Channel D
    ax4.plot(d, 'b-')
    ax4.set_ylabel('TbT ChD')
    #ax4.set_title(r'$\mu$=%4.3f um  $\sigma$=%4.3f um' % (np.mean(d), np.std(d)))
    ax4.grid(True)
    
    # X Position
    ax5.plot(x, 'b-')
    ax5.set_ylabel('TbT XPos (um)')
    ax5.set_xlabel('Turn #')
    ax5.set_title(r'$\mu$=%4.3f um  $\sigma$=%4.3f um' % (np.mean(x), np.std(x)))
    ax5.grid(True)

    # Y Position
    ax6.plot(y, 'b-')
    ax6.set_ylabel('TbT YPos (um)')
    ax6.set_xlabel('Turn #')
    ax6.set_title(r'$\mu$=%4.3f um  $\sigma$=%4.3f um' % (np.mean(y), np.std(y)))
    ax6.grid(True)

    # Overall title
    fig.suptitle("TbT Position", fontsize=14)

    # Fix layout so titles/labels don’t overlap
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Print summary
    print("Xpos mean: %f um \tstd: %f um" % (np.mean(x), np.std(x)))
    print("Ypos mean: %f um \tstd: %f um" % (np.mean(y), np.std(y)))

    plt.show(block=False)

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def plot_psd(tbt_data, Ftbt):
    """
    Plot PSD in dB and integrated PSD for X and Y channels from tbt_data.

    Parameters:
        tbt_data : 2D array (num_samples x 6)
                   Columns: a,b,c,d,xpos,ypos
        Ftbt     : sampling frequency (Hz)
    """
    numpts = len(tbt_data[:,0])
    print("NumPts: %d" % numpts)
    df = Ftbt / numpts
    xpos = tbt_data[:,4]
    ypos = tbt_data[:,5]

    # Compute PSD
    f, psdX = signal.periodogram(xpos, Ftbt, 'flattop', scaling='density')
    f, psdY = signal.periodogram(ypos, Ftbt, 'flattop', scaling='density')

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
    axs[0].set_title('TbT PSD')
    axs[0].grid(True)
    axs[0].legend(loc=2)

    # Integrated PSD plot
    axs[1].semilogx(f, cs_psdX, '.-', label='X')
    axs[1].semilogx(f, cs_psdY, '.-', label='Y')
    axs[1].set_xlabel('Frequency [Hz]')
    axs[1].set_ylabel('Integrated PSD [$\mu m$]')
    axs[1].set_title('TbT Integrated PSD')
    axs[1].grid(True)
    axs[1].legend(loc=2)

    plt.tight_layout()
    plt.show(block=False)

    # Print total integrated power
    print(f"Total integrated power X: {cs_psdX[-1]:.6e}")
    print(f"Total integrated power Y: {cs_psdY[-1]:.6e}")




'''
def plot_psd(tbt_data, Ftbt):
    numpts = len(tbt_data[:,0])
    print("NumPts: %d" % numpts)
    df = Ftbt / numpts
    xpos = tbt_data[:,4]
    ypos = tbt_data[:,5]

    # compute PSD
    f, psdX = signal.periodogram(xpos, Ftbt, 'flattop', scaling='density')
    cs_psdX = np.sqrt(np.cumsum(psdX*df))
    f, psdY = signal.periodogram(ypos, Ftbt, 'flattop', scaling='density')
    cs_psdY = np.sqrt(np.cumsum(psdY*df))

    # PSD and integrated PSD plots
    fig2, axs2 = plt.subplots(nrows=2, ncols=1, figsize=(8,8), sharex=True)

    # PSD plot on log-x scale
    axs2[0].semilogx(f, psdX, label='X')
    axs2[0].semilogx(f, psdY, label='Y')
    axs2[0].set_title('TbT PSD')
    axs2[0].set_ylabel(r'PSD [$\mu m^2$/Hz]')
    axs2[0].set_ylim([1e-10, 1e-2])
    axs2[0].grid(True)
    axs2[0].legend(loc=2)

    # Integrated PSD plot
    axs2[1].semilogx(f, cs_psdX, '.-', label='X')
    axs2[1].semilogx(f, cs_psdY, '.-', label='Y')
    axs2[1].set_title('TbT Integrated PSD')
    axs2[1].set_xlabel('Freq [Hz]')
    axs2[1].set_ylabel(r'Int PSD [$\mu m$]')
    axs2[1].grid(True)
    axs2[1].legend(loc=2)

    plt.tight_layout()
    plt.show(block=False)




def plot_psd(tbt_data, Ftbt):
    numpts = len(tbt_data[:,0])
    print("NumPts: %d" % numpts)
    df = Ftbt / numpts
    xpos = tbt_data[:,4]
    ypos = tbt_data[:,5]

    # compute PSD
    f, psdX = signal.periodogram(xpos, Ftbt, 'flattop', scaling='density')
    cs_psdX = np.sqrt(np.cumsum(psdX*df))
    f, psdY = signal.periodogram(ypos, Ftbt, 'flattop', scaling='density')
    cs_psdY = np.sqrt(np.cumsum(psdY*df))

    # PSD and integrated PSD plots
    fig2, axs2 = plt.subplots(nrows=2, ncols=1, figsize=(8,8))

    # PSD plot
    axs2[0].semilogy(f, psdX, label='X')
    axs2[0].semilogy(f, psdY, label='Y')
    axs2[0].set_title('TbT PSD')
    axs2[0].set_ylabel(r'PSD [$\mu m^2$/Hz]')
    axs2[0].set_ylim([1e-10, 1e-2])
    axs2[0].set_xlim([0, Ftbt/2])
    axs2[0].grid(True)
    axs2[0].legend(loc=2)

    # Integrated PSD plot
    axs2[1].semilogx(f, cs_psdX, '.-', label='X')
    axs2[1].semilogx(f, cs_psdY, '.-', label='Y')
    axs2[1].set_title('TbT Integrated PSD')
    axs2[1].set_xlabel('Freq [Hz]')
    axs2[1].set_ylabel(r'Int PSD [$\mu m$]')
    axs2[1].grid(True)
    axs2[1].legend(loc=2)

    plt.tight_layout()
    plt.show(block=False)





def plot_psd(tbt_data):
   numpts = len(tbt_data[:,0])
   print("NumPts: %d" % numpts)
   df = Ftbt/numpts
   xpos = tbt_data[:,4]
   ypos = tbt_data[:,5]
 
   # compute and plot power spectrum
   f,psdX = signal.periodogram(xpos,Ftbt,'flattop',scaling='density')
   cs_psdX = np.sqrt(np.cumsum(psdX*df))
   f,psdY = signal.periodogram(ypos,Ftbt,'flattop',scaling='density')
   cs_psdY = np.sqrt(np.cumsum(psdY*df))

   # PSD and Int. PSD Plots
   fig2,axs2 = plt.subplots(nrows=2, ncols=1, figsize=(8,8))

   plt.figure(2)
   #plt.semilogy(f,np.sqrt(psdX))
   axs2[0] = plt.subplot(211)
   #plt.loglog(f,psdX,f,psdY)
   plt.semilogy(f,psdX,f,psdY)
   plt.gca().legend(('X','Y'),loc=2)
   plt.title('TbT PSD ')
   plt.legend
   plt.ylim([1e-10,1e-2])
   plt.xlim([0,Ftbt/2])
   #plt.xlabel('Freq [Hz]')
   plt.ylabel('PSD X [um'+r'$^2$'+'/Hz]')
   plt.grid()

   plt.subplot(212, sharex=axs2[0])
   plt.semilogx(f,cs_psdX,'.-',f,cs_psdY,'.-')
   plt.title('TbT Int. PSD ')
   plt.ylabel('Int PSD [um]')
   plt.xlabel('Freq [Hz]')
   #plt.xlim([1,fs/2])
   plt.gca().legend(('X','Y'),loc=2)
   plt.grid()
'''





def main():

   plt.ion()
   if len(sys.argv) != 2:
       print ("Missing input file...")
       sys.exit() 
   else:
       fname = sys.argv[1]


   tbt_data=read_tbtdata(fname)
   plot_tbt(tbt_data)
   
   plot_psd(tbt_data,Ftbt)
   

   
   plt.show()
   input('Press any key to quit...')




if __name__ == "__main__":
    main()

