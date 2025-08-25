import cothread
import epics
import numpy as np
import time
import argparse

datapts = 100000




def get_waveform(PVs,numpts):
  

  data = []
  for i in range(len(PVs)):
     data.append(PVs[i].get())

  waveform = np.asarray(data, dtype=np.float32)
  waveform = waveform[:,0:numpts] 

  return waveform 




def main():
    parser = argparse.ArgumentParser(description="Test DMA Triggers...")
    parser.add_argument("bpm_prefix", type=str, help="BPM prefix string (e.g., lab-BI{BPM:2})")
    parser.add_argument("time_delay", type=float, help="Time between triggers")
    args = parser.parse_args()
    
    bpm_prefix = args.bpm_prefix
    delay = args.time_delay
 

    trig_pv = epics.PV(bpm_prefix+'Trig:Strig-SP')

    # Trigger the BPM
    while True:
       trig_pv.put(1)
       print("Triggering BPM...")
       time.sleep(delay)


if __name__ == "__main__":
    main()




