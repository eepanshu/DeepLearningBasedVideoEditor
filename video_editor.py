import cv2
import numpy as np
from PIL import Image
import re
from typing import List, Tuple, Optional
import os
import tempfile
from typing import Dict

class VideoEditor:
    def __init__(self):
        """Initialize the video editor with AI models."""
        self.device = 'cpu'
        self.has_yolo = False
        self.yolo_model = None
        self._torch = None
        # Optional: Grounded-SAM (GroundingDINO + SAM) for prompt-guided segmentation
        self.has_gsam = False
        self._gsam = {
            'gdino': None,
            'sam_predictor': None
        }
        # Optional SAM2
        self.has_sam2 = False
        self.sam2_predictor = None
        # Map colloquial names to COCO class aliases for better matching
        self.class_aliases = {
            'cup': ['cup', 'wine glass', 'bottle', 'mug'],
            'mug': ['cup', 'wine glass', 'bottle', 'mug'],
            'glass': ['wine glass', 'cup', 'bottle'],
            'phone': ['cell phone'],
            'mobile': ['cell phone'],
            'sofa': ['couch'],
            'tv': ['tv']
        }

        # Try to enable YOLO if torch/ultralytics are available
        try:
            import torch  # noqa: F401
            from ultralytics import YOLO  # type: ignore
            self._torch = torch
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {self.device}")
            # Prefer segmentation model for mask-level removal
            preferred = os.getenv('MODEL_NAME', '').strip() or 'yolov8n-seg.pt'
            tried = []
            candidates = [preferred, 'yolov8s-seg.pt', 'yolov8m-seg.pt', 'yolov8n.pt']
            for name in candidates:
                if not name:
                    continue
                try:
                    self.yolo_model = YOLO(name)
                    print(f"Loaded model: {name}")
                    break
                except Exception as _:
                    tried.append(name)
                    continue
            if self.yolo_model is None:
                raise RuntimeError(f"Failed to load any YOLO model from: {tried}")
            self.has_yolo = True
            print("YOLO loaded successfully. Object removal is enabled.")
        except Exception as e:
            print("YOLO/PyTorch not available. Style edits will work; object removal disabled.")
        
        # Try to enable Grounded-SAM as a more accurate, prompt-based segmenter
        try:
            import torch
            from huggingface_hub import hf_hub_download  # type: ignore
            from groundingdino.util.inference import Model as GDINOModel  # type: ignore
            from segment_anything import sam_model_registry, SamPredictor  # type: ignore

            gdino_cfg = hf_hub_download(repo_id="ShilongLiu/GroundingDINO", filename="GroundingDINO_SwinT_OGC.py")
            gdino_ckpt = hf_hub_download(repo_id="ShilongLiu/GroundingDINO", filename="groundingdino_swint_ogc.pth")
            gdino = GDINOModel(model_config_path=gdino_cfg, model_checkpoint_path=gdino_ckpt)
            gdino.to(self.device)

            sam_ckpt = hf_hub_download(repo_id="facebookresearch/segment-anything", filename="sam_vit_b_01ec64.pth")
            sam = sam_model_registry["vit_b"](checkpoint=sam_ckpt)
            sam.to(self.device)
            sam_predictor = SamPredictor(sam)

            self._gsam['gdino'] = gdino
            self._gsam['sam_predictor'] = sam_predictor
            self.has_gsam = True
            print("Grounded-SAM loaded. Prompt-based object removal enabled.")
        except Exception as e:
            self.has_gsam = False
            print(f"Grounded-SAM not available: {e}")

        # Try to enable SAM2 (higher quality masks). If it fails, we keep SAMv1.
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

            sam2_ckpt = hf_hub_download(repo_id="facebook/sam2-hiera-tiny", filename="sam2_hiera_tiny.pt")
            sam2_model = build_sam2(checkpoint=sam2_ckpt)
            predictor2 = SAM2ImagePredictor(sam2_model)
            # Move to device if possible
            try:
                predictor2.model.to(self.device)
            except Exception:
                pass
            self.sam2_predictor = predictor2
            self.has_sam2 = True
            print("SAM2 loaded for segmentation.")
        except Exception as _e:
            self.has_sam2 = False

        # COCO class names for object detection
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def parse_edit_instruction(self, instruction: str) -> dict:
        """Parse natural language instruction to extract edit parameters."""
        instruction = instruction.lower().strip()
        
        # Object removal patterns
        remove_patterns = [
            r'remove\s+the\s+([^from]+?)\s+from\s+the\s+([^\.]+)',
            r'delete\s+the\s+([^from]+?)\s+from\s+the\s+([^\.]+)',
            r'get\s+rid\s+of\s+the\s+([^from]+?)\s+from\s+the\s+([^\.]+)'
        ]
        
        for pattern in remove_patterns:
            match = re.search(pattern, instruction)
            if match:
                return {
                    'type': 'object_removal',
                    'object': match.group(1).strip(),
                    'location': match.group(2).strip()
                }
        
        # Style change patterns
        style_patterns = [
            r'make\s+it\s+look\s+like\s+(night[\s\-]?time|evening)',
            r'apply\s+(night[\s\-]?time|evening)\s+effect',
            r'change\s+to\s+(night[\s\-]?time|evening)',
            r'apply\s+vintage\s+filter',
            r'make\s+it\s+vintage',
            r'apply\s+black\s+and\s+white',
            r'make\s+it\s+black\s+and\s+white'
        ]
        
        for pattern in style_patterns:
            match = re.search(pattern, instruction)
            if match:
                style = match.group(1) if len(match.groups()) > 0 else match.group(0)
                return {
                    'type': 'style_change',
                    'style': style.strip()
                }
        
        # Default to object removal if no pattern matches
        return {
            'type': 'object_removal',
            'object': instruction,
            'location': 'scene'
        }
    
    def detect_objects(self, frame: np.ndarray) -> List[dict]:
        """Detect/segment objects in a frame using YOLO.
        Returns list of dicts with bbox, mask (np.uint8 0/255) if available, class_id, class_name, confidence.
        """
        if not self.has_yolo or self.yolo_model is None:
            return []

        results = self.yolo_model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            masks = getattr(result, 'masks', None)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence > 0.25:  # Lowered confidence threshold to catch smaller objects
                        det = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.coco_classes[class_id] if class_id < len(self.coco_classes) else 'unknown'
                        }
                        # If segmentation masks available, include per-instance mask
                        if masks is not None and masks.data is not None and i < len(masks.data):
                            mask_tensor = masks.data[i].cpu()
                            mask_np = (mask_tensor.numpy() * 255).astype(np.uint8)
                            # Resize mask to frame size if needed
                            if mask_np.shape[:2] != frame.shape[:2]:
                                mask_np = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                            det['mask'] = mask_np
                        detections.append(det)
        
        return detections

    def segment_with_gsam(self, frame: np.ndarray, prompt: str) -> List[dict]:
        """Use GroundingDINO + SAM to get masks for the text prompt."""
        if not self.has_gsam:
            return []
        try:
            import supervision as sv  # type: ignore
            gdino = self._gsam['gdino']
            predictor = self._gsam['sam_predictor']

            image_bgr = frame
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            # Try rich prompt with aliases to improve recall
            aliases = self.class_aliases.get(prompt.strip().lower(), [])
            caption = ", ".join([prompt] + aliases) if aliases else prompt
            detections = gdino.predict_with_caption(
                image=image_rgb,
                caption=caption,
                box_threshold=0.20,
                text_threshold=0.20
            )
            if detections is None or len(detections) == 0:
                return []

            # detections: xyxy in absolute
            boxes = detections.xyxy if hasattr(detections, 'xyxy') else detections
            out = []
            predictor.set_image(image_rgb)
            best = []
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                # SAM expects box in XYXY as numpy array
                import numpy as np
                box_np = np.array([x1, y1, x2, y2])
                masks, scores, logits = predictor.predict(box=box_np[None, :], multimask_output=False)
                if (masks is None or len(masks) == 0) and self.has_sam2 and self.sam2_predictor is not None:
                    # Try SAM2
                    try:
                        self.sam2_predictor.set_image(image_rgb)
                        m2, s2, _ = self.sam2_predictor.predict(box=box_np[None, :], multimask_output=False)
                        masks = m2
                        scores = s2
                    except Exception:
                        masks = None
                if masks is None or len(masks) == 0:
                    continue
                mask = (masks[0].astype(np.uint8)) * 255
                # Ensure mask size matches frame
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                best.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(scores[0]) if scores is not None else 0.9,
                    'class_id': -1,
                    'class_name': 'prompt',
                    'mask': mask
                })
            # Return highest-confidence first
            best.sort(key=lambda d: d['confidence'], reverse=True)
            return best
        except Exception:
            return []
    
    def find_object_by_name(self, detections: List[dict], object_name: str) -> Optional[dict]:
        """Find the best matching object by name."""
        object_name = object_name.lower().strip()
        best_match = None
        best_score = 0
        
        for detection in detections:
            class_name = detection['class_name'].lower()
            
            # Expand aliases
            candidate_names = [class_name]
            if object_name in self.class_aliases:
                candidate_names.extend(self.class_aliases[object_name])

            # Exact/alias match
            if any((nm in class_name or class_name in nm) for nm in candidate_names):
                if detection['confidence'] > best_score:
                    best_match = detection
                    best_score = detection['confidence']
            
            # Partial match for common objects
            elif any(word in class_name for word in object_name.split()):
                if detection['confidence'] > best_score:
                    best_match = detection
                    best_score = detection['confidence']
        
        return best_match
    
    def remove_object(self, frame: np.ndarray, bbox: List[int], mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Remove object from frame using inpainting with refined mask and feathered blending."""
        h, w = frame.shape[:2]
        if mask is None:
            x1, y1, x2, y2 = bbox
            pad_x = max(2, int(0.12 * (x2 - x1)))
            pad_y = max(2, int(0.12 * (y2 - y1)))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w - 1, x2 + pad_x)
            y2 = min(h - 1, y2 + pad_y)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255

        # 1) Mask refine: close to fill holes, erode to keep inside, then distance-based feather
        obj_area = max(1, int(mask.sum() / 255))
        base_r = max(3, min(21, int(np.sqrt(obj_area) * 0.03)))
        k = 2 * base_r + 1
        kernel = np.ones((k, k), np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        core = cv2.erode(closed, kernel, iterations=1)

        # 2) First-pass Telea at native scale
        telea = cv2.inpaint(frame, core, base_r, cv2.INPAINT_TELEA)

        # 3) Optional biharmonic smoothing on V channel to remove residual blotches
        try:
            from skimage.restoration import inpaint as sk_inpaint
            hsv = cv2.cvtColor(telea, cv2.COLOR_BGR2HSV)
            v = hsv[:, :, 2]
            v_f = sk_inpaint.inpaint_biharmonic(v.astype(float)/255.0, (core>0), multichannel=False)
            hsv[:, :, 2] = np.clip(v_f*255.0, 0, 255).astype(np.uint8)
            telea = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        except Exception:
            pass

        # 4) Feathered blending to avoid seams around boundaries
        dist = cv2.distanceTransform((core>0).astype(np.uint8), cv2.DIST_L2, 3)
        if dist.max() > 0:
            alpha = (dist / dist.max())
        else:
            alpha = (core>0).astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha3 = np.stack([alpha, alpha, alpha], axis=-1)
        blended = (alpha3 * telea + (1 - alpha3) * frame).astype(np.uint8)

        # 5) Edge-preserving smoothing inside the hole to equalize textures
        try:
            smoothed = cv2.bilateralFilter(blended, d=9, sigmaColor=25, sigmaSpace=25)
            blended[core>0] = smoothed[core>0]
        except Exception:
            pass

        return blended
    
    def apply_night_effect(self, frame: np.ndarray) -> np.ndarray:
        """Apply night-time effect to frame."""
        # Convert to HSV for better color manipulation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Reduce brightness and saturation
        hsv[:, :, 1] = hsv[:, :, 1] * 0.3  # Reduce saturation
        hsv[:, :, 2] = hsv[:, :, 2] * 0.4  # Reduce brightness
        
        # Add blue tint for night effect
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + 20, 0, 179)  # Shift hue towards blue
        
        # Convert back to BGR
        night_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Add some noise and darken further
        noise = np.random.normal(0, 10, night_frame.shape).astype(np.uint8)
        night_frame = cv2.add(night_frame, noise)
        night_frame = np.clip(night_frame, 0, 255).astype(np.uint8)
        
        return night_frame
    
    def apply_vintage_effect(self, frame: np.ndarray) -> np.ndarray:
        """Apply vintage/sepia effect to frame."""
        # Convert to sepia
        sepia = np.array([[0.393, 0.769, 0.189],
                         [0.349, 0.686, 0.168],
                         [0.272, 0.534, 0.131]])
        
        vintage = cv2.transform(frame, sepia)
        vintage = np.clip(vintage, 0, 255).astype(np.uint8)
        
        # Add vignette effect
        rows, cols = vintage.shape[:2]
        X_resultant_kernel = cv2.getGaussianKernel(cols, cols/3)
        Y_resultant_kernel = cv2.getGaussianKernel(rows, rows/3)
        kernel = Y_resultant_kernel * X_resultant_kernel.T
        mask = kernel / kernel.max()
        vintage = vintage * mask[:, :, np.newaxis]
        
        return vintage.astype(np.uint8)
    
    def apply_black_white_effect(self, frame: np.ndarray) -> np.ndarray:
        """Apply black and white effect to frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    def process_video(self, video_path: str, instruction: str, progress_cb: Optional[callable] = None, initial_mask_path: Optional[str] = None) -> str:
        """Process video based on instruction. Returns output video path.
        progress_cb: optional callable taking (current, total, message)
        """
        # Parse instruction
        edit_params = self.parse_edit_instruction(instruction)
        print(f"Parsed instruction: {edit_params}")
        
        # Load video and read first frame reliably
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file.")

        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            cap.release()
            raise RuntimeError("No frames found in the input video.")

        fps_val = cap.get(cv2.CAP_PROP_FPS)
        fps = int(fps_val) if fps_val and fps_val > 1 else 24
        height, width = first_frame.shape[:2]

        # Create output writer after we know dimensions and fps
        output_path = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise RuntimeError("Failed to initialize video writer.")

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        
        edited_frames = 0
        prev_bbox: Optional[List[int]] = None
        prev_mask: Optional[np.ndarray] = None
        smooth_alpha = 0.7  # weight for previous bbox to reduce jitter
        tracker = None  # OpenCV tracker for temporal stability when detection misses
        frame = first_frame

        # If user provided a mask on the first frame, prime tracker/mask
        if initial_mask_path and os.path.exists(initial_mask_path):
            try:
                user_mask_img = cv2.imread(initial_mask_path, cv2.IMREAD_UNCHANGED)
                if user_mask_img is not None:
                    if user_mask_img.ndim == 3:
                        user_mask = user_mask_img[:, :, -1] if user_mask_img.shape[2] == 4 else cv2.cvtColor(user_mask_img, cv2.COLOR_BGR2GRAY)
                    else:
                        user_mask = user_mask_img
                    user_mask = cv2.resize(user_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    _, user_mask = cv2.threshold(user_mask, 10, 255, cv2.THRESH_BINARY)
                    prev_mask = user_mask
                    ys, xs = np.where(user_mask > 0)
                    if len(xs) > 0:
                        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                        prev_bbox = [x1, y1, x2, y2]
                        try:
                            tracker = cv2.legacy.TrackerCSRT_create()
                            tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                        except Exception:
                            tracker = None
            except Exception:
                pass
        while True:
            
            frame_count += 1
            if progress_cb:
                try:
                    progress_cb(frame_count, total_frames or 1, f"Processing frame {frame_count}/{total_frames or '?'}")
                except Exception:
                    pass
            else:
                print(f"Processing frame {frame_count}/{total_frames}")
            
            if edit_params['type'] == 'object_removal':
                # Detect objects in frame
                detections = []
                # If user supplied initial mask we try tracker first to avoid re-detecting each frame
                used_tracker = False
                if tracker is not None and prev_bbox is not None:
                    ok, box = tracker.update(frame)
                    if ok:
                        x, y, w_box, h_box = box
                        tb = [int(x), int(y), int(x + w_box), int(y + h_box)]
                        detections = [{
                            'bbox': tb,
                            'confidence': 0.99,
                            'class_id': -1,
                            'class_name': 'tracked',
                            'mask': prev_mask
                        }]
                        used_tracker = True
                if not used_tracker:
                    # Prefer prompt-based segmentation if available
                    if self.has_gsam:
                        detections = self.segment_with_gsam(frame, edit_params['object'])
                    if not detections:
                        detections = self.detect_objects(frame)
                
                # Find target object
                target_object = self.find_object_by_name(detections, edit_params['object'])
                
                if target_object:
                    # Prefer mask-based removal when available
                    cur_bbox = target_object['bbox']
                    # Smooth bbox with previous to reduce jitter
                    if prev_bbox is not None:
                        sx1 = int(smooth_alpha * prev_bbox[0] + (1 - smooth_alpha) * cur_bbox[0])
                        sy1 = int(smooth_alpha * prev_bbox[1] + (1 - smooth_alpha) * cur_bbox[1])
                        sx2 = int(smooth_alpha * prev_bbox[2] + (1 - smooth_alpha) * cur_bbox[2])
                        sy2 = int(smooth_alpha * prev_bbox[3] + (1 - smooth_alpha) * cur_bbox[3])
                        cur_bbox = [sx1, sy1, sx2, sy2]

                    cur_mask = target_object.get('mask')
                    if cur_mask is not None and prev_mask is not None:
                        # Merge with previous mask to fill small gaps
                        cur_mask = cv2.bitwise_or(cur_mask, prev_mask)

                    frame = self.remove_object(frame, cur_bbox, cur_mask)
                    prev_bbox = cur_bbox
                    prev_mask = cur_mask
                    # Initialize or update tracker with current bbox
                    try:
                        if tracker is None:
                            tracker = cv2.legacy.TrackerCSRT_create()
                            tracker.init(frame, (cur_bbox[0], cur_bbox[1], cur_bbox[2]-cur_bbox[0], cur_bbox[3]-cur_bbox[1]))
                        else:
                            tracker = cv2.legacy.TrackerCSRT_create()
                            tracker.init(frame, (cur_bbox[0], cur_bbox[1], cur_bbox[2]-cur_bbox[0], cur_bbox[3]-cur_bbox[1]))
                    except Exception:
                        tracker = None
                    edited_frames += 1
                else:
                    if not self.has_yolo:
                        # Fallback: no YOLO, skip object removal and just pass frame
                        if frame_count == 1:
                            print("Object removal requested but YOLO is unavailable. Skipping removal.")
                    else:
                        used_fallback = False
                        # If we have a tracker, predict bbox
                        if tracker is not None:
                            ok, box = tracker.update(frame)
                            if ok:
                                x, y, w_box, h_box = box
                                tb = [int(x), int(y), int(x + w_box), int(y + h_box)]
                                frame = self.remove_object(frame, tb, prev_mask)
                                prev_bbox = tb
                                edited_frames += 1
                                used_fallback = True
                        # Else reuse last known bbox/mask
                        if not used_fallback and (prev_bbox is not None or prev_mask is not None):
                            frame = self.remove_object(frame, prev_bbox or [0, 0, width, height], prev_mask)
                            edited_frames += 1
                            used_fallback = True
                        if not used_fallback:
                            print(f"Object '{edit_params['object']}' not found in frame {frame_count}")
            
            elif edit_params['type'] == 'style_change':
                style = edit_params['style'].lower()
                
                if 'night' in style or 'evening' in style:
                    frame = self.apply_night_effect(frame)
                elif 'vintage' in style:
                    frame = self.apply_vintage_effect(frame)
                elif 'black' in style and 'white' in style:
                    frame = self.apply_black_white_effect(frame)
                edited_frames += 1
            
            # Guard against size mismatch
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            out.write(frame)

            ret, frame = cap.read()
            if not ret:
                break

        cap.release()
        out.release()
        
        # Store stats for UI feedback
        self.last_result = {
            'total_frames': total_frames,
            'edited_frames': edited_frames,
            'edit_type': edit_params['type']
        }
        # Re-encode to H.264 for browser playback and return absolute path
        try:
            from moviepy.editor import VideoFileClip  # local import
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            web_safe_path = tmp.name
            tmp.close()
            with VideoFileClip(os.path.abspath(output_path)) as clip:
                clip.write_videofile(web_safe_path, codec='libx264', audio=False, fps=fps, preset='ultrafast', verbose=False, logger=None)
            try:
                os.remove(output_path)
            except Exception:
                pass
            return os.path.abspath(web_safe_path)
        except Exception:
            # If re-encode fails, return the original path
            return os.path.abspath(output_path)

def create_interface():
    """Create Gradio interface for the video editor with optional first-frame mask drawing."""
    import gradio as gr
    editor = VideoEditor()

    def get_first_frame(video_file):
        path = None
        if isinstance(video_file, str):
            path = video_file
        elif isinstance(video_file, dict) and 'path' in video_file:
            path = video_file['path']
        elif hasattr(video_file, 'name'):
            path = video_file.name
        if not path:
            return None
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        cv2.imwrite(tmp.name, frame)
        return tmp.name

    def process(video_file, instruction, mask_img, progress=gr.Progress(track_tqdm=False)):
        if video_file is None:
            return None, "Please upload a video file."
        if not instruction.strip():
            return None, "Please enter an edit instruction."
        # Resolve paths
        video_path = None
        if isinstance(video_file, str):
            video_path = video_file
        elif isinstance(video_file, dict) and 'path' in video_file:
            video_path = video_file['path']
        elif hasattr(video_file, 'name'):
            video_path = video_file.name
        if not video_path:
            return None, "Could not read uploaded video path."
        try:
            def cb(cur, tot, msg):
                p = min(1.0, max(0.0, float(cur) / float(max(1, tot))))
                progress(p, desc=msg)
            mask_path = None
            if isinstance(mask_img, str):
                mask_path = mask_img
            elif hasattr(mask_img, 'name'):
                mask_path = mask_img.name
            output_path = editor.process_video(video_path, instruction, progress_cb=cb, initial_mask_path=mask_path)
            stats = getattr(editor, 'last_result', None)
            if stats and stats['edited_frames'] == 0:
                status = "Processed, but no frames were changed. Draw a mask on the first frame or try a clearer prompt."
            elif stats:
                status = f"Video processed successfully! Edited {stats['edited_frames']} / {stats['total_frames']} frames."
            else:
                status = "Video processed successfully!"
            return gr.update(value=output_path, autoplay=True), status
        except Exception as e:
            return None, f"Error processing video: {str(e)}"

    with gr.Blocks(title="AI Video Editor") as app:
        gr.Markdown("Upload a video, optionally paint a mask on the first frame over the object, and submit.")
        with gr.Row():
            video_in = gr.File(label="Upload Video", file_types=['video'])
            mask_img = gr.ImageEditor(label="Optional mask on first frame")
        instruction = gr.Textbox(label="Edit Instruction", placeholder="e.g., remove the cup")
        submit = gr.Button("Submit")
        with gr.Row():
            out_video = gr.Video(label="Edited Video", autoplay=True, include_audio=False)
            status = gr.Textbox(label="Status")

        video_in.change(fn=get_first_frame, inputs=video_in, outputs=mask_img)
        submit.click(fn=process, inputs=[video_in, instruction, mask_img], outputs=[out_video, status])

    return app

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
