import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image

import x3d as resnet_x3d
import pandas as pd
from transforms.spatial_transforms import Compose, Normalize, ToTensor, CenterCropScaled


df_labels = pd.read_csv("data/kinetics_400_labels.csv")
labels_to_id = dict(df_labels[["id", "name"]].values)

KINETICS_MEAN = [110.63666788/255, 103.16065604/255, 96.29023126/255]
KINETICS_STD = [38.7568578/255, 37.88248729/255, 40.02898126/255]

BS = 8
BS_UPSCALE = 16  # CHANGE WITH GPU AVAILABILITY

GPUS = 4
BASE_BS_PER_GPU = BS * BS_UPSCALE // GPUS
CONST_BN_SIZE = 8

X3D_VERSION = 'M'


def inference_net(clip, model, spatial_transforms, device):
    video_transformed = [spatial_transforms(img) for img in clip]

    video_transformed = torch.stack(video_transformed, 0).permute(1, 0, 2, 3)  # T C H W --> C T H W
    video_transformed = video_transformed.unsqueeze(0)
    video_transformed = video_transformed.to(device)

    logits = model(video_transformed)

    logits = logits.to("cpu")
    logits_list = [x[0].item() for x in logits[0]]
    logits_sm = F.softmax((torch.Tensor(logits_list)), 0)
    prob, pred = torch.max(logits_sm, 0)
    print(labels_to_id[pred.item()])
    print(prob.item())
    del video_transformed
    del logits
    return labels_to_id[pred.item()], prob.item()


def run(video_fname, clip_size=30, save_video_name=None):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    crop_size = {'S': 160, 'M': 224, 'XL': 312}[X3D_VERSION]
    spatial_transforms = Compose([CenterCropScaled(crop_size),
                                  ToTensor(255),
                                  Normalize(KINETICS_MEAN, KINETICS_STD)])

    x3d = resnet_x3d.generate_model(x3d_version=X3D_VERSION, n_classes=400, n_input_channels=3,
                                    dropout=0.5, base_bn_splits=BASE_BS_PER_GPU//CONST_BN_SIZE)
    load_ckpt = torch.load(
        'models/x3d_multigrid_kinetics_rgb_sgd_204000.pt',
        map_location=torch.device('cpu')
    )
    x3d.load_state_dict(load_ckpt['model_state_dict'])
    x3d.eval()
    x3d.to(device)

    vidcap = cv2.VideoCapture(video_fname)
    success, image = vidcap.read()
    video = []
    results = []
    idx = 0
    if save_video_name:
        out_vid = cv2.VideoWriter(save_video_name, -1, 30.0, (int(image.shape[0] / 2), int(image.shape[1] / 2)))
    while success:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        video.append(image)
        if len(video) % clip_size == 0:
            pred_cls, prob = inference_net(video, x3d, spatial_transforms, device)
            if save_video_name:
                for img in video:
                    img = np.array(img)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = f"{pred_cls} {prob * 100}%"
                    cv2.putText(img, text, (10, 10), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
                    out_vid.write(img)

            video = []
            results.append([idx, pred_cls, prob])
            idx += 1
        success, image = vidcap.read()

    df_res = pd.DataFrame(results, columns=['idx', 'label', 'probability'])
    df_res.to_csv("results.csv")
    out_vid.release()


if __name__ == '__main__':
    video_fname = "data/IMG_4772.MOV"
    clip_size = 80  # process every 80 frames
    save_video_name = "vid_results.mp4"
    run(video_fname, clip_size=clip_size, save_video_name=save_video_name)
