import cv2
import pandas as pd


def visualise(video_name, input_preds, save_video_name="data/result.mp4"):
    vidcap = cv2.VideoCapture(video_name)
    success, image = vidcap.read()

    print(f"create video writer {video_name}, wiht shape {image.shape}")
    out_vid = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (image.shape[1], image.shape[0]))
    df = pd.read_csv(input_preds)
    for index, row in df.iterrows():
        for frame_idx in range(row["frame_end"] - row["start_end"]):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"{row['preds_cls']} {row['prob'] * 100}%"
            cv2.putText(image, text, (10, 100), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
            out_vid.write(image)

            success, image = vidcap.read()
            if not success:
                break


if __name__ == "__main__":
    video_name = "data/IMG_4772.MOV"
    input_preds = "results.csv"
    visualise(video_name, input_preds)

