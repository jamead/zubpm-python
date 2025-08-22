import cothread
import epics
import numpy as np
import time
import argparse



def get_waveform(PVs,numpts):
  

  data = []
  for i in range(len(PVs)):
     data.append(PVs[i].get())

  waveform = np.asarray(data, dtype=np.float32)
  waveform = waveform[:,0:numpts] 

  return waveform 




def main():
    parser = argparse.ArgumentParser(description="Read ADC data from a BPM and save to file.")
    parser.add_argument("bpm_prefix", type=str, help="BPM prefix string (e.g., lab-BI{BPM:2})")
    parser.add_argument("outfile", type=str, help="Output filename (e.g., adc_data.txt)")
    args = parser.parse_args()

    bpm_prefix = args.bpm_prefix
    out_filename = args.outfile

    tbt_pv = [
        epics.PV(bpm_prefix+'ADC:A:Buff-Wfm'),
        epics.PV(bpm_prefix+'ADC:B:Buff-Wfm'),
        epics.PV(bpm_prefix+'ADC:C:Buff-Wfm'),
        epics.PV(bpm_prefix+'ADC:D:Buff-Wfm')]
  

    trig_pv = epics.PV(bpm_prefix+'Trig:Strig-SP')

    # Trigger the BPM
    trig_pv.put(1)
    time.sleep(1)

    # Read waveform
    adc_data = get_waveform(tbt_pv, 1000000)
    print(type(adc_data))
    rows, cols = adc_data.shape
    print(f"Number of rows: {rows}")
    print(f"Number of columns: {cols}")

    # Save to file
    np.savetxt(out_filename, adc_data.T, fmt="%d", delimiter=" ")
    print(f"Saved ADC data to {out_filename}")


if __name__ == "__main__":
    main()




