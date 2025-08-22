import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
#import scipy.signal as signal
import seaborn as sns


Frf = 499.68e6

h = 1320
hADC = 310 
hIf = 80 
Fs = Frf/h*hADC
Ftbt = Frf/h
Ts =  1/Fs
adcFS = 8192.0
#N = 100000 
numTurns = 200 
kx=10e-3
ky=10e-3

Fpt = Frf + 5.234*Ftbt



def read_adcdata(fname):
   a = []
   b = []
   c = []
   d = []


   data = np.loadtxt(fname)

   print("Read %d data points" % len(data[:,0]))
   #set to an integer number of turns
   numturns = int(len(data[:,0])/hADC)
   print("Num Turns : %d" % (numturns))
   newlen=numturns*hADC
   print("New Length : %d" % (newlen))

   cha = np.array(data[0:newlen,0]) 
   chb = np.array(data[0:newlen,1]) 
   chc = np.array(data[0:newlen,2]) 
   chd = np.array(data[0:newlen,3]) 
   print("ChA Numpts : %d" % len(cha))

   print("ChA mean: %f" % np.mean(cha))
   print("ChB mean: %f" % np.mean(chb))
   print("ChC mean: %f" % np.mean(chc))
   print("ChD mean: %f" % np.mean(chd))
   print("Subtracting Mean...")
   #cha = cha - np.mean(cha)
   #chb = chb - np.mean(chb)
   #chc = chc - np.mean(chc)
   #chd = chd - np.mean(chd)
   #print("ChA mean: %f" % np.mean(cha))
   #print("ChB mean: %f" % np.mean(chb))
   #print("ChC mean: %f" % np.mean(chc))
   #print("ChD mean: %f" % np.mean(chd))

   cha = cha / adcFS
   chb = chb / adcFS
   chc = chc / adcFS
   chd = chd / adcFS

   return cha,chb,chc,chd

def gen_adcdata():
   # generate fake data
   t = np.linspace(0,Ts*hADC*numTurns,hADC*numTurns)
   print("Fake Data Len: %f" % len(t))
   
   #sine = .8 * np.sin(2*np.pi*Frf/h*hIf*t)
   sine = .4 * np.sin(2*np.pi*Frf*t) + .05 * np.sin(2*np.pi*Fpt*t)
   cha = 1.001 * sine + np.random.randn(len(t))*0.0001
   chb = sine + np.random.randn(len(t))*0.00001
   chc = sine + np.random.randn(len(t))*0.00001
   chd = sine + np.random.randn(len(t))*0.00001
   return cha,chb,chc,chd

def calc_psd(y):
   N = len(y)  
   #print ("len(y)=%d" % N)
   #f = np.linspace(0,Fs/2,N/2+1)
   #print ("len(t)=%f" % len(adc_data))
   #print ("len(f)=%f" % len(f))
   #print ("Total Time=%f" % (Ts*N))

   w = np.hanning(N)
   x = w * y
   xfft = np.abs(np.fft.rfft(x)) / (N/2.0)

   p = 20*np.log10(xfft)
   return y,p

def plot_tbt(a,b,c,d,x,y):


   #plot TbT results
   fig,axes = plt.subplots(nrows=3,ncols=2)

   ax1=plt.subplot(321)
   plt.plot(a,'b-')
   plt.ylabel('TbT ChA')
   plt.title( ''r'$\mu$=%4.3f um 'r'$\sigma$=%4.3f um' % (np.mean(a), np.std(a))) 
   #plt.xlabel('Turn #')
   plt.grid()

   ax2=plt.subplot(322, sharex=ax1)
   plt.plot(b,'b-')
   plt.ylabel('TbT ChB')
   plt.title( ''r'$\mu$=%4.3f um 'r'$\sigma$=%4.3f um' % (np.mean(b), np.std(b))) 
   #plt.xlabel('Turn #')
   plt.grid()

   ax3=plt.subplot(323, sharex=ax1)
   plt.plot(c,'b-')
   plt.ylabel('TbT ChC')
   plt.title( ''r'$\mu$=%4.3f um 'r'$\sigma$=%4.3f um' % (np.mean(c), np.std(c))) 
   #plt.xlabel('Turn #')
   plt.grid()

   ax4=plt.subplot(324, sharex=ax1)
   plt.plot(d,'b-')
   plt.ylabel('TbT ChD')
   plt.title( ''r'$\mu$=%4.3f um 'r'$\sigma$=%4.3f um' % (np.mean(d), np.std(d))) 
   #plt.xlabel('Turn #')
   plt.grid()

   ax5=plt.subplot(325, sharex=ax1)
   plt.plot(x,'b-')
   plt.ylabel('TbT XPos (um)')
   plt.xlabel('Turn #')
   plt.title(''r'$\mu$=%4.3f um 'r'$\sigma$=%4.3f um' % (np.mean(x), np.std(x))) 
   plt.grid()

   ax6=plt.subplot(326, sharex=ax1)
   plt.plot(y,'b-')
   plt.ylabel('TbT YPos (um)')
   plt.xlabel('Turn #')
   plt.title( ''r'$\mu$=%4.3f um 'r'$\sigma$=%4.3f um' % (np.mean(y), np.std(y))) 
   plt.grid()
   fig.suptitle("TbT Position")
 
   print("Xpos mean: %f um \tstd: %f um" % (np.mean(x), np.std(x)))
   print("ypos mean: %f um \tstd: %f um" % (np.mean(y), np.std(y)))

def plot_adc(a,b,c,d):
   N = len(a)
   t = np.linspace(0,Ts*hADC*numTurns,hADC*numTurns)
   a = (a*32767)
   b = (b*32767)
   c = (c*32767)
   d = (d*32767)
   ylim = [-40000,40000]
   fig,axes = plt.subplots(nrows=2,ncols=2)
   ax1=plt.subplot(221)
   plt.plot(a,'b-o')
   plt.ylabel('adu')
   ax1.set_ylim(ylim)
   #plt.xlabel('freq (MHz)')
   plt.title('ChA')
   plt.grid()
   ax2=plt.subplot(222, sharex=ax1)
   plt.plot(b,'b-o')
   plt.ylabel('adu')
   ax2.set_ylim(ylim)
   #plt.xlabel('freq (MHz)')
   plt.title('ChB')
   plt.grid()
   ax3=plt.subplot(223, sharex=ax1)
   plt.plot(c,'b-o')
   plt.ylabel('adu')
   plt.xlabel('sample num')
   plt.title('ChC')
   ax3.set_ylim(ylim)
   plt.grid()
   ax4=plt.subplot(224, sharex=ax1)
   plt.plot(d,'b-o')
   plt.ylabel('adu')
   plt.xlabel('sample num')
   plt.title('ChD')
   plt.grid()
   ax4.set_ylim(ylim)
   titlestr = "ADC : Sample Rate 310*40*Frf/1320 = 4.69396GSPS" 
   fig.suptitle(titlestr)

def plot_adcoverlay(rawdata,fignum):

   print("Num Samples: %d" % len(rawdata))
   N = len(rawdata)

   #fig,axes = plt.subplots(nrows=1,ncols=1)
   #for i in range(6):
   #    plt.plot(rawdata[i*hADC:(i+1)*hADC-1])

   print("Hello")


   plt.figure(5) 
   plt.plot(rawdata[hADC*0:hADC],'b.-') 
   plt.plot(rawdata[hADC*5:hADC*6],'g.-')
   plt.plot(rawdata[hADC*10:hADC*11],'r.-')
   plt.plot(rawdata[hADC*15:hADC*16],'y.-')
   plt.plot(rawdata[hADC*20:hADC*21],'c.-')
   plt.plot(rawdata[hADC*25:hADC*26],'m.-')
 

   plt.grid()

   '''
   fig,axes = plt.subplots(nrows=2,ncols=2)
   ax1=plt.subplot(221)
   plt.plot(rawdata[0:12400],'b-o')
   plt.grid()
   ax2=plt.subplot(222)
   plt.plot(rawdata[12390:12410],'b-o')
   plt.grid()
   ax3=plt.subplot(223, sharex=ax1)
   plt.plot(rawdata[2*hADC:3*hADC],'b-o')
   plt.grid()
   ax4=plt.subplot(224, sharex=ax1)
   plt.plot(rawdata[3:hADC:4*hADC],'b-o')
   plt.grid()
   '''

def plot_psd(a,b,c,d):
   N = len(a)*2-1
   f = np.linspace(0,Fs/2,N//2+1)
   f = f/1e6 #scale to MHz  
   ylim = [-160,0]
   fig,axes = plt.subplots(nrows=2,ncols=2)

   ax1=plt.subplot(221)
   plt.plot(f,a,'b')
   plt.ylabel('dBFS')
   ax1.grid(True) 
   ax1.set_ylim(ylim)
   #plt.xlabel('freq (MHz)')
   plt.title('PSD ChA')

   ax2=plt.subplot(222, sharex=ax1)
   plt.plot(f,b,'b')
   plt.ylabel('dBFS')
   ax2.grid(True) 
   ax2.set_ylim(ylim)
   #plt.xlabel('freq (MHz)')
   plt.title('PSD ChB')

   ax3=plt.subplot(223, sharex=ax1)
   plt.plot(f,c,'b')
   plt.ylabel('dBFS')
   plt.xlabel('freq (MHz)')
   plt.title('PSD ChC')
   ax3.set_ylim(ylim)
   ax3.grid(True)

   ax4=plt.subplot(224, sharex=ax1)
   plt.plot(f,d,'b')
   plt.ylabel('dbFS')
   plt.xlabel('freq (MHz)')
   plt.title('PSD ChD')
   ax4.grid(True)
   ax4.set_ylim(ylim)
   #titlestr = "ADC PSD   " + "numpts=" + str(2*(len(f)-1))
   #fig.suptitle(titlestr)

def plot_adcchan(y,p,plottitle):
   N = len(y)  
   t = np.linspace(0,Ts*hADC*numTurns,hADC*numTurns)
 
   #t = np.arange(0,N*Ts-Ts,Ts)
   f = np.linspace(0,Fs/2,N/2+1)
 
   fig,axes = plt.subplots(nrows=2,ncols=2)
   ax1=plt.subplot(211)
   plt.plot(t*1000,y,'.-')
   plt.ylabel('ADC counts')
   plt.xlabel('Time (msec)')
   plt.grid()
   ax2=plt.subplot(212)
   plt.plot(f/1e6,p)
   plt.ylabel('Power dBFS')
   plt.xlabel('Freq (MHz)')
   plt.grid()
   fig.suptitle(plottitle)
   plt.draw()
   plt.pause(0.01)

def to_twos_complement(val, bits=16):
    # If the value is negative, we calculate the two's complement
    if val < 0:
        val = (1 << bits) + val  # Add the bitmask to convert to two's complement
    
    # Format the number to the specified bit width
    return val #format(val, f'0{bits}b')


def fir_filter(input_signal):

  b = [0.1321, 0.3679, 0.3679, 0.1321]

  filt_signal = np.zeros_like(input_signal)

  for n in range(len(input_signal)):
    # Implement the FIR filter difference equation for each time step
    for k in range(4):
        if n - k >= 0:  # Ensure we are within the bounds of the signal
            filt_signal[n] += b[k] * input_signal[n - k]

  plt.figure(10)
  plt.plot(input_signal, label='Input Signal')
  plt.plot(filt_signal, label='Filtered Output', linestyle='--')
  plt.legend()
  plt.title('4-Tap FIR Filter Implementation')
  plt.xlabel('Time')
  plt.ylabel('Amplitude')

  return filt_signal

def downconvert(adc):
   numpts = len(adc)
   numturns = int(len(adc)/hADC)
   #adc = np.reshape(x,(numturns,hADC))
   t = np.linspace(0, (numpts - 1) * Ts, numpts)

   sine = np.sin(2*np.pi*Frf/h*hIf*t)
   cos = np.cos(2*np.pi*Frf/h*hIf*t) 

   #for i in range (0,310):
   #   print("%d: Sine: %10f\t  %8d\t  %8x\t  Cos: %f" % (i,sine[i],sine[i]*32768,to_twos_complement(int(sine[i]*32768)),cos[i]))

   coeffs = [0.1321, 0.3679, 0.3679, 0.1321]


   tbt = np.zeros((numturns,1))

   iraw = adc * cos 
   qraw = adc * sine

   #ifilt = lfilter(coeffs,1,iraw)
   #qfilt = lfilter(coeffs,1,qraw)
   ifilt = fir_filter(iraw)
   qfilt = fir_filter(qraw)
   
   print("Len i: %d   len ifilt: %d" % (len(iraw), len(ifilt)))
   print("Numturns: %d" % (numturns))
   
   tbt=np.zeros(numturns)
   for i in range(0,numpts,hADC):
      for j in range(0,hADC):  
         tbt[i // hADC] += np.sqrt(np.square(ifilt[i+j])+np.square(qfilt[i+j])) 


   #remove 1st turn due to filter response
   #tbt = tbt[1:]


   #print(tbt)
   
   '''
   plt.figure(5) 
   plt.title('Sine, Cos')
   plt.plot(sine,'b.-') 
   plt.plot(cos,'g.-')
   plt.figure(6)
   plt.title("Mag")
   plt.plot(mag,'g.-')   
   plt.figure(7)
   plt.title("I,Ifilt")
   plt.plot(i,'b.-')
   plt.plot(ifilt,'r.-')
   plt.figure(8)  
   plt.title('Q, Qfilt')       
   plt.plot(q,'b.-')
   plt.plot(qfilt,'r.-')
   plt.figure(9)
   plt.title('TbT')
   plt.plot(tbt,'b.-')
   plt.draw() 
   '''
   
   return tbt

def downconvertold(x):
   numturns = int(len(x)/hADC)
   adc = np.reshape(x,(numturns,hADC))
   t = np.arange(0,hADC*Ts,Ts)
   sine = np.sin(2*np.pi*Frf/h*hIf*t)
   cos = np.cos(2*np.pi*Frf/h*hIf*t) 

   coeffs = [0.1321, 0.3679, 0.3679, 0.1321]

   tbt = np.zeros((numturns,1))
   for x in range(0,numturns):
      i = adc[x,:] * cos 
      q = adc[x,:] * sine

      ifilt = lfilter(coeffs,1,i)
      qfilt = lfilter(coeffs,1,q)


      mag = np.sqrt(np.square(ifilt)+np.square(qfilt)) 
      tbt[x] = np.sum(mag)
   print(tbt)
   
   
   plt.figure(5) 
   plt.title('Sine, Cos')
   plt.plot(sine,'b.-') 
   plt.plot(cos,'g.-')
   plt.figure(6)
   plt.title("Mag")
   plt.plot(mag,'g.-')   
   plt.figure(7)
   plt.title("I,Ifilt")
   plt.plot(i,'b.-')
   plt.plot(ifilt,'r.-')
   plt.figure(8)  
   plt.title('Q, Qfilt')       
   plt.plot(q,'b.-')
   plt.plot(qfilt,'r.-')
   plt.figure(9)
   plt.title('TbT')
   plt.plot(tbt,'b.-')
   plt.draw() 
   
   
   return tbt

def bandpass_filt(adca,adcb,adcc,adcd):
   n = 10 
   fircoef = signal.firwin(n, cutoff = [0.5,0.53], window = "hamming", pass_zero= False)
   adca = signal.lfilter(fircoef,1,adca)
   adcb = signal.lfilter(fircoef,1,adcb)
   adcc = signal.lfilter(fircoef,1,adcc)
   adcd = signal.lfilter(fircoef,1,adcd)
   return adca,adcb,adcc,adcd 

def main():

   plt.ion()
   #plt.style.use('seaborn-whitegrid')
   #sns.set_style("whitegrid")
   plt.rc('font', size=8)
   if len(sys.argv) != 2:
       print  ("No input file, generating simulated adc data...")
       adca,adcb,adcc,adcd=gen_adcdata() 
   else:
       fname = sys.argv[1]
       adca,adcb,adcc,adcd=read_adcdata(fname)
       
   #adca,adcb,adcc,adcd = bandpass_filt(adca,adcb,adcc,adcd)

   print ("Number of Samples Collected: %d" % len(adca))
   print ("Number of Turns: %f" % (len(adca)/float(hADC)))
   
   plot_adc(adca,adcb,adcc,adcd)
   #plot_adcoverlay(adcb,10)

   ya,pa = calc_psd(adca) 
   yb,pb = calc_psd(adcb)
   yc,pc = calc_psd(adcc) 
   yd,pd = calc_psd(adcd)
   plot_psd(pa,pb,pc,pd)


   #plot_adc(ya,pa,"ADC chA")
   #y,p = calc_fft(adcb)
   #plot_adc(y,p,"ADC chB")
    
   
   a = downconvert(adca)
   b = downconvert(adcb)
   c = downconvert(adcc)
   d = downconvert(adcd)

   x = kx * (((a+d)-(b+c)) / (a+b+c+d)) * 1e6  #scale to um
   y = ky * (((a+b)-(c+d)) / (a+b+c+d)) * 1e6  #scale to um
   
   plot_tbt(a,b,c,d,x,y)
   
   
   
   #plt.figure(10)
   #plt.plot(a)
   #plt.plot(b)
   #plt.plot(c)
   #plt.plot(d)
   #plt.grid() 



 
   plt.figure(4)
   plt.plot(adca[0:310],'b.-')
   plt.plot(adca[310:620],'r.-')   
   plt.plot(adca[620:930],'g.-') 
   plt.figure(5)
   plt.plot(adca[0:1000],'b.-')
   
   plt.show()
   input('Press any key to quit...')
   





if __name__ == "__main__":
    main()



