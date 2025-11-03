-import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import os
import torch
import tempfile
import subprocess
from pathlib import Path
import queue
import threading
from datetime import datetime
import shutil

if not torch.cuda.is_available():
    st.error("TensorRT engine requires CUDA GPU. No GPU found!")
    st.stop()

DEVICE = 'cuda'
torch.cuda.set_device(0)
st.sidebar.success(f"Using GPU: {torch.cuda.get_device_name(0)}")

st.title("YOLO Object Detection")

MODEL_PATH = "D:/1Work/Projects/Object Detection/ftrain/kaggle/working/runs/detect/train/weights/best.engine"
INPUT_SIZE = (1280, 1280)

def process_image(image):
    """Resize and pad image to fixed input size"""
    target_size = INPUT_SIZE
    img = image.copy()
    ratio = min(target_size[0] / img.size[0], target_size[1] / img.size[1]) #min to avoid distoring the image
    new_size = tuple(int(dim * ratio) for dim in img.size)
    img = img.resize(new_size, Image.Resampling.LANCZOS) #Image.Resampling.LANCZOS for high quality
    
    new_img = Image.new("RGB", target_size, (114, 114, 114)) # blank of image created with the target dimensions and backgroung color (114, 114, 114) 
    new_img.paste(img, ((target_size[0] - new_size[0]) // 2,
                       (target_size[1] - new_size[1]) // 2)) # the image in the center 
    return new_img

@st.cache_resource #When the function is called with the same model_path, it returns the cached model instead of reloading it.
def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            st.error(f"Engine file not found at: {model_path}")
            return None
        model = YOLO(model_path, task='detect')
        return model
    except Exception as e:
        st.error(f"Error loading TensorRT engine: {str(e)}")
        return None

class VideoProcessor:
    def __init__(self, source_path, batch_size=4):
        self.source_path = source_path
        self.batch_size = batch_size
        self.frame_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue()
        self.processing_done = threading.Event()
        self.cap = cv2.VideoCapture(source_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def extract_frames(self):
        frame_count = 0
        while True:
            frames_batch = []
            for _ in range(self.batch_size):
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_count += 1
                frames_batch.append((frame_count, frame))
            
            if not frames_batch:
                break
                
            self.frame_queue.put(frames_batch)
        
        self.cap.release()
        self.processing_done.set()

    def process_frames(self, model, conf_threshold):
        while not (self.processing_done.is_set() and self.frame_queue.empty()):
            try:
                frames_batch = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            processed_batch = []
            for frame_idx, frame in frames_batch:
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                processed_frame = process_image(frame_pil)
                frame = cv2.cvtColor(np.array(processed_frame), cv2.COLOR_RGB2BGR)
                
                results = model.predict(
                    source=frame,
                    conf=conf_threshold,
                    device=DEVICE,
                    verbose=False,
                    imgsz=INPUT_SIZE
                )
                annotated_frame = results[0].plot()
                processed_batch.append((frame_idx, annotated_frame))
            
            self.result_queue.put(processed_batch)

def save_processed_video(processor, output_path, progress_bar):
    # Create temporary directory for frames
    temp_dir = Path(tempfile.mkdtemp())
    try:
        frame_files = []
        processed_frames = 0

        # Save frames as images
        while processed_frames < processor.total_frames:
            try:
                batch = processor.result_queue.get(timeout=30)  # 30 second timeout
                for frame_idx, frame in batch:
                    frame_path = temp_dir / f"frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    frame_files.append(frame_path)
                    processed_frames += 1
                    progress_bar.progress(processed_frames / processor.total_frames)
            except queue.Empty:
                break

        # Sort frames by index
        frame_files.sort()

        # Use FFmpeg to create video from frames
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(processor.fps),
            '-pattern_type', 'sequence',
            '-i', str(temp_dir / 'frame_%06d.jpg'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',
            output_path
        ]
        
        subprocess.run(ffmpeg_cmd, check=True)

    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir)

model = load_model(MODEL_PATH)
if model is None:
    st.stop()

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
fps_placeholder = st.sidebar.empty()

upload_type = st.radio("Upload type", ["Image", "Video"])

if upload_type == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            processed_image = process_image(image)
            
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            results = model.predict(
                source=processed_image,
                conf=conf_threshold,
                device=DEVICE,
                verbose=False,
                imgsz=INPUT_SIZE
            )
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)
            
            annotated_image = results[0].plot()
            st.image(annotated_image, caption="Detected Image")
            st.sidebar.info(f"Inference time: {inference_time:.2f}ms")
            
            # Save and provide download for annotated image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                Image.fromarray(annotated_image).save(tmp_file.name)
                with open(tmp_file.name, 'rb') as f:
                    st.download_button(
                        label="Download annotated image",
                        data=f.read(),
                        file_name="annotated_image.png",
                        mime="image/png"
                    )
                os.unlink(tmp_file.name)
            
            for r in results:
                for box in r.boxes:
                    st.write(f"Class: {model.names[int(box.cls)]} | Confidence: {box.conf.item():.2f}")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

else:
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        try:
            # Save uploaded video to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as input_tmp:
                input_tmp.write(uploaded_file.read())
                input_video_path = input_tmp.name

            # Create output video path
            output_video_path = str(Path(tempfile.mkdtemp()) / 'output.mp4')

            # Initialize video processor
            processor = VideoProcessor(input_video_path)
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Start frame extraction thread
            extract_thread = threading.Thread(target=processor.extract_frames)
            extract_thread.start()

            # Start processing thread
            process_thread = threading.Thread(
                target=processor.process_frames,
                args=(model, conf_threshold)
            )
            process_thread.start()

            # Save processed video with progress bar
            status_text.text("Processing video...")
            save_processed_video(processor, output_video_path, progress_bar)

            # Wait for threads to complete
            extract_thread.join()
            process_thread.join()

            status_text.text("Processing complete! You can now download the video.")
            
            # Provide download button
            with open(output_video_path, 'rb') as f:
                st.download_button(
                    label="Download annotated video",
                    data=f.read(),
                    file_name=f"annotated_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    mime="video/mp4"
                )

            # Cleanup
            os.unlink(input_video_path)
            os.unlink(output_video_path)
            os.rmdir(str(Path(output_video_path).parent))

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            # Cleanup in case of error
            for path in [input_video_path, output_video_path]:
                if os.path.exists(path):
                    os.unlink(path)
            if os.path.exists(str(Path(output_video_path).parent)):
                os.rmdir(str(Path(output_video_path).parent))

gpu_memory = torch.cuda.memory_allocated(0) / 1024**2
st.sidebar.info(f"GPU Memory Used: {gpu_memory:.2f} MB")

