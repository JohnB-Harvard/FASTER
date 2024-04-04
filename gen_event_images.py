import glob
import h5py
from PIL import Image
import numpy as np

folders = glob.glob("*5mDepthRender")
print(folders)
for folder in folders:
    print(f"Starting with {folder}")
    f = h5py.File(f'{folder}\events.h5')

    # events is a list of tuples of form (x, y, polarity, time), sort by time
    events = list(f.get("events"))
    print(f"Found {len(events)} events")
    f.close()
    # events.sort(key=lambda e: e[3])
    # print(events[0])
    # print(events[-1])

    # events_a = [np.zeros((512,512), dtype=int) for _ in range(events[-1][3]+1)]
    # for event in events:
    #     assert(event[2] == 1 or event[2] == -1)
    #     events_a[event[3]][event[0]][event[1]] = event[2]
    # print("Created event images")

    # for i in range(len(events_a)):
    #     assert(events_a[i] is not None)
    #     events_im = Image.fromarray(events_a[i]*127+128).convert('P')
    #     leading_zeros = ''.join([str(0) for _ in range(4-len(str(i+1)))])
    #     events_im.save(f'{folder}\\events_{leading_zeros}{i+1}.png', 'PNG')