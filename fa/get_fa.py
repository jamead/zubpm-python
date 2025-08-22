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
    parser = argparse.ArgumentParser(description="Read TbT data from a BPM and save to file.")
    parser.add_argument("bpm_prefix", type=str, help="BPM prefix string (e.g., lab-BI{BPM:2})")
    parser.add_argument("outfile", type=str, help="Output filename (e.g., fa_data.txt)")
    args = parser.parse_args()

    bpm_prefix = args.bpm_prefix
    out_filename = args.outfile

    tbt_pv = [
        epics.PV(bpm_prefix+'FA:A:Buff-Wfm'),
        epics.PV(bpm_prefix+'FA:B:Buff-Wfm'),
        epics.PV(bpm_prefix+'FA:C:Buff-Wfm'),
        epics.PV(bpm_prefix+'FA:D:Buff-Wfm'),
        epics.PV(bpm_prefix+'FA:X:Buff-Wfm'),
        epics.PV(bpm_prefix+'FA:Y:Buff-Wfm')]
  

    trig_pv = epics.PV(bpm_prefix+'Trig:Strig-SP')

    # Trigger the BPM
    trig_pv.put(1)
    time.sleep(1)

    # Read waveform
    tbt_data = get_waveform(tbt_pv, 1000000)
    print(type(tbt_data))
    rows, cols = tbt_data.shape
    print(f"Number of rows: {rows}")
    print(f"Number of columns: {cols}")

    # Save to file
    np.savetxt(out_filename, tbt_data.T, fmt="%d", delimiter=" ")
    print(f"Saved FA data to {out_filename}")


if __name__ == "__main__":
    main()




