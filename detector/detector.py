import torch
from tqdm import tqdm
import cv2
from dataset import get_loader
import numpy as np
import pandas as pd

class Detector_SSD():
    def __init__(self):
        self.model = self.get_model()
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

    def get_model(self):
        ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp32')
        ssd_model.cuda().eval()
        return ssd_model

    @staticmethod
    def get_frames_dict(vid_path, fps=1):
        video = cv2.VideoCapture(str(vid_path))
        video.set(cv2.CAP_PROP_BUFFERSIZE, 5)
        max_frame = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'fps: {fps}, max_frame: {max_frame}')

        frames_dict = {}
        for curr_frame in tqdm(range(max_frame)): # max_frame
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_dict[curr_frame] = frame
        return frames_dict


    def save_detections(self, vid_path, classes_to_save=[1], fps=1):
        frames_dict = self.get_frames_dict(vid_path, fps=fps)
        Loader, Dataset = get_loader(frames_dict)


        bboxes_dict = {}
        for batch in tqdm(Loader):
            tensor = batch[0].cuda()

            with torch.no_grad():
                detections_batch = self.model(tensor)

            results_per_input = self.utils.decode_results(detections_batch)  # decoding results
            best_results_per_input = [self.utils.pick_best(results, 0.40) for results in
                                      results_per_input]  # filtering bboxes (NMS)

            for image_idx in range(len(best_results_per_input)):
                key = batch[1][image_idx].item()
                # original image
                image = Dataset.get_image(key)

                bboxes, classes, confidences = best_results_per_input[image_idx]
                # scaling bboxes to original image
                bboxes = bboxes.copy() * max(image.shape[:2])

                bboxes_dict[key] = []
                for idx in range(len(bboxes)):
                    if classes[idx] in classes_to_save:
                        left, bot, right, top = bboxes[idx].astype(np.int32)
                        bboxes_dict[key].append(list(map(int, [left, bot, right, top])))

        # Save
        csv_path = str(vid_path.with_name(vid_path.stem + ".csv"))
        d = {'frame_num': list(bboxes_dict.keys()),
             'bboxes': list(bboxes_dict.values())}
        df = pd.DataFrame(data=d).to_csv(csv_path, index=False)
        print(csv_path, 'saved')




