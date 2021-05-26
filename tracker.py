from sort import *
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy.optimize import linear_sum_assignment




class Tracker():
    def __init__(self, video_name):
        self.video_name = video_name
        self.save_path = r'result'
        self.mot_tracker = Sort()
        self.save_folder = Path('results')
        self.video_folder = Path('vids')

    def get_images(self, vid_path, fps=1):
        video = cv2.VideoCapture(str(vid_path))
        video.set(cv2.CAP_PROP_BUFFERSIZE, 5)
        max_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'fps: {fps}, max_frame: {max_frame}')

        frames_dict = {}
        for curr_frame in tqdm(range(max_frame)):  # max_frame
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_dict[curr_frame] = frame
        return frames_dict

    def save_Trackings(self):
        path = str(self.video_folder.joinpath(self.video_name + '.avi'))
        frames_dict = self.get_images(path)

        path = str(self.video_folder.joinpath(self.video_name + '.csv'))
        DF= pd.read_csv(str(path), converters={'bboxes': eval})

        h, w = frames_dict[0].shape[:2]
        # output_video = cv2.VideoWriter('output.mp4', -1, 24.0, (w,h))
        new_name = str(self.video_folder.joinpath(self.video_name + '_mod.avi'))
        output_video = cv2.VideoWriter(new_name, cv2.VideoWriter_fourcc(*"MJPG"), 24, (w, h))

        for idx, row in tqdm(DF.iterrows()):
            frame_num = row['frame_num']
            bboxes = row['bboxes']
            det_img = frames_dict[frame_num].copy()
            newname = os.path.join(self.save_path, str(frame_num) + '.png')

            # prepare deteted_bboxes
            det_list = np.array([bbox + [1.] for bbox in bboxes])
            if len(det_list) == 0:
                det_list = np.empty((0, 5))
            # update tracker state
            track_bbs_ids = self.mot_tracker.update(det_list)

            # draw rectangles
            for row in track_bbs_ids:
                x, y, x2, y2, _id = list(map(int, row))
                cv2.rectangle(det_img, (x, y), (x2, y2), (0, 255, 0), 4)
                _id = str(_id)
                cv2.putText(det_img, '#' + _id, (x + 5, y - 10), 0, 0.6, (0, 255, 0), thickness=2)

            #### show not tracked guys
            IoU = -iou_batch(det_list, track_bbs_ids)
            prev_indices, boxes_indices = linear_sum_assignment(IoU)

            # leave non-corresponding bboxes only
            mask = np.ones(det_list.shape[0])
            mask[prev_indices] = 0
            not_tracked_bboxes = det_list[mask.astype(bool)]
            # Draw them
            for row in not_tracked_bboxes:
                x, y, x2, y2, _id = list(map(int, row))
                cv2.rectangle(det_img, (x, y), (x2, y2), (0, 0, 255), 4)

            output_video.write(det_img)
            # cv2.imwrite(newname, det_img)
        output_video.release()




if __name__=='__main__':
    video_names = ['campus4-c0', 'campus4-c1', 'campus4-c2']

    for name in video_names:
        trackr = Tracker(name)
        trackr.save_Trackings()

