from detector import Detector_SSD
from pathlib import Path



def main(vids_p = r'Z:\test tasks\EORA\vids'):
    Det = Detector_SSD()
    Paths = [x for x in Path(vids_p).glob('*.avi')]

    for p in Paths:
        Det.save_detections(p, fps=1)

if __name__=='__main__':
    main()