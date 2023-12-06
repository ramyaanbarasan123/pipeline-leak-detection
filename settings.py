from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, YOUTUBE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'default.png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'default_detect.png'
CONFUSION_MATRIX_IMAGE = IMAGES_DIR / 'confusion_matrix.png'
F1_CURVE_IMAGE = IMAGES_DIR / 'F1_curve.png'
P_CURVE_IMAGE = IMAGES_DIR / 'P_curve.png'
PR_CURVE_IMAGE = IMAGES_DIR / 'PR_curve.png'
R_CURVE_IMAGE = IMAGES_DIR / 'R_curve.png'
RESULTS_IMAGE = IMAGES_DIR / 'results.png'
RESULTS_CSV = IMAGES_DIR / 'results.csv'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'gas-leak.mp4'
VIDEOS_DICT = {
    'gas-leak': VIDEO_1_PATH,
}

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'leakage.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'

# Webcam
WEBCAM_PATH = 0
