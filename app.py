import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import gradio as gr
import os
from tqdm import tqdm

# -----------------------------
# Load Models (GPU if available)
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# YOLOv8 (object detection)
yolo_model = YOLO("models/yolov8s.pt").to(device)

# SAM (segmentation)
sam_checkpoint = "models/sam_vit_b.pth"
sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint).to(device)
sam_predictor = SamPredictor(sam)

# -----------------------------
# Helper: Parse objects from text
# -----------------------------
def get_target_objects(description):
    words = description.lower().split()
    possible_objects = [
        "person", "cup", "bottle", "box", "vase", "chair",
        "dog", "cat", "laptop", "car", "cell phone", "book", "potted plant"
    ]
    targets = [obj for obj in possible_objects if obj in words]
    return list(set(targets))

# -----------------------------
# Main Editing Function
# -----------------------------
def edit_video(video_path, description, bg_option="none", bg_file_path=None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs("outputs", exist_ok=True)
    output_path = "outputs/edited_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Extract objects from text
    target_objects = get_target_objects(description)

    # Prepare custom background (image or video)
    custom_bg_img = None
    custom_bg_video = None
    if bg_option == "custom" and bg_file_path is not None:
        # Try to open as image first
        custom_bg_img = cv2.imread(bg_file_path)
        if custom_bg_img is not None:
            custom_bg_img = cv2.resize(custom_bg_img, (width, height))
        else:
            # If not an image, try video
            custom_bg_video = cv2.VideoCapture(bg_file_path)
            if not custom_bg_video.isOpened():
                print("⚠️ Could not open custom background file.")
                custom_bg_video = None

    print(f"🎯 Target objects: {target_objects if target_objects else 'None'}")
    print(f"🎨 Background option: {bg_option}")

    for frame_idx in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break

        # -----------------------------
        # 1. Object detection (YOLO)
        # -----------------------------
        results = yolo_model(frame, verbose=False)
        detections = results[0].boxes

        if detections is not None and len(detections) > 0:
            for box, cls in zip(detections.xyxy.cpu().numpy(), detections.cls.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                label = results[0].names[int(cls)]

                # -----------------------------
                # 2. Remove only target objects
                # -----------------------------
                if target_objects and label in target_objects:
                    print(f"🗑 Frame {frame_idx}: Removing {label} at ({x1},{y1},{x2},{y2})")
                    sam_predictor.set_image(frame)
                    masks, _, _ = sam_predictor.predict(
                        box=np.array([x1, y1, x2, y2]),
                        multimask_output=False
                    )
                    mask = masks[0].astype(np.uint8) * 255
                    frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

        # -----------------------------
        # 3. Apply Background Effect
        # -----------------------------
        if bg_option == "dark":
            frame = cv2.convertScaleAbs(frame, alpha=0.5, beta=0)  # darker
        elif bg_option == "light":
            frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=30)  # brighter
        elif bg_option == "custom":
            # Prepare background frame (image or video)
            if custom_bg_img is not None:
                bg_frame = custom_bg_img
            elif custom_bg_video is not None:
                ret_bg, bg_frame = custom_bg_video.read()
                if not ret_bg:  # loop video if ended
                    custom_bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_bg, bg_frame = custom_bg_video.read()
                bg_frame = cv2.resize(bg_frame, (width, height))
            else:
                bg_frame = np.zeros_like(frame)

            # Segment main objects (like people, etc.)
            sam_predictor.set_image(frame)
            masks, _, _ = sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                multimask_output=False
            )
            mask = masks[0].astype(np.uint8)
            mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

            # Keep foreground, replace background
            frame = frame * mask_3ch + bg_frame * (1 - mask_3ch)

        out.write(frame)

    cap.release()
    out.release()
    if custom_bg_video is not None:
        custom_bg_video.release()
    print(f"✅ Video saved at: {output_path}")
    return output_path

# -----------------------------
# Gradio Interface
# -----------------------------
def gradio_interface(video, text, bg_option, bg_file):
    bg_file_path = None
    if bg_option == "custom" and bg_file is not None:
        bg_file_path = bg_file
    return edit_video(video, text, bg_option, bg_file_path)

with gr.Blocks() as demo:
    gr.Markdown("## 🎬 AI-based Video Editing with RTX Acceleration")
    with gr.Row():
        video_input = gr.Video(label="Upload Video")
        text_input = gr.Textbox(
            label="Edit Description",
            placeholder="e.g. Remove cup and vase"
        )
    with gr.Row():
        bg_option = gr.Dropdown(
            ["none", "dark", "light", "custom"],
            label="Background Option",
            value="none"
        )
        bg_file = gr.File(
            label="Custom Background (Image or Video)",
            file_types=[".jpg", ".png", ".mp4", ".avi", ".mov"],
            type="filepath"
        )
    output_video = gr.Video(label="Final Edited Video")
    run_button = gr.Button("Run Edit")

    run_button.click(
        fn=gradio_interface,
        inputs=[video_input, text_input, bg_option, bg_file],
        outputs=output_video
    )

if __name__ == "__main__":
    demo.launch()
