# utils_image.py: 이미지 전처리 및 파일 정보 추출 (Label-Only, V3 전처리 동일)
import torch
import pydicom
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import os
from utils.utils_segmentation import segmentation
import cv2
import gradio as gr
import random
from datetime import datetime
from modules.STN import STN
import torch.nn.functional as F
from config import DEFAULT_DATETIME

to_tensor = transforms.ToTensor()
image_resize_transform = transforms.Resize((512,512), interpolation=transforms.InterpolationMode.BILINEAR)
mask_resize_transform = transforms.Resize((512,512), interpolation=transforms.InterpolationMode.NEAREST)

_stn_model = None


def _get_stn_model(device):
    global _stn_model
    if _stn_model is None:
        MODEL_PATH = "assets/weights/stn_weights.pth"
        if not os.path.exists(MODEL_PATH):
            raise gr.Error(f"STN 가중치 파일이 없습니다: {MODEL_PATH}")
        model = STN(in_shape=(1, 512, 512))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        _stn_model = model
    return _stn_model


def spatial_normalize(original_batch, preseg_batch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    stn_model = _get_stn_model(device)

    if preseg_batch.shape[-2:] != (512, 512):
        preseg_batch = F.interpolate(preseg_batch.float(), size=(512, 512), mode='nearest')

    preseg_batch = preseg_batch.to(device)
    original_batch = original_batch.to(device)

    with torch.no_grad():
        theta = stn_model(preseg_batch)

    grid = F.affine_grid(theta, original_batch.size(), align_corners=False)
    aligned_original = F.grid_sample(original_batch, grid, align_corners=False)

    return aligned_original


def _to_gray_uint8(img):
    """seg/STN 입력 규격(1채널 uint8)으로 정리.

    NPZ 로드 경로는 .tolist() → np.array 왕복을 거치며 int32가 되고, DICOM/PNG
    경로는 이미 uint8 2D다. 3채널이 들어오는 경우도 대비해 흑백으로 낮춘다.
    """
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f" and float(arr.max()) <= 1.5:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3:
        arr = arr[:, :, 0] if arr.shape[2] == 1 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def normalize_images(images):
    """1024 흑백 이미지 리스트 → GUNet 분할 + STN 정렬한 512 2D 배열 리스트.

    GUNet은 1채널 1024 고정 입력, STN은 1채널 512 입력이다.
    """
    seg_inputs, nrm_inputs = [], []
    for img in images:
        gray = _to_gray_uint8(img)
        seg_inputs.append(to_tensor(preprocess(gray)))
        nrm_inputs.append(to_tensor(preprocess_(gray)))

    img_seg_list = torch.stack(seg_inputs).float()   # [N, 1, 1024, 1024]
    img_nrm_list = torch.stack(nrm_inputs).float()   # [N, 1, 512, 512]

    mask_tensor = segmentation(img_seg_list).unsqueeze(1)          # [N, 1, 1024, 1024]
    img_list_tensor = spatial_normalize(img_nrm_list, mask_tensor)  # [N, 1, 512, 512]
    img_list_np = img_list_tensor.cpu().numpy()

    return [img_list_np[i, 0] for i in range(len(img_list_np))]


def preprocess(input_img):
    img = np.array(input_img, dtype=np.uint8)
    h, w = img.shape[:2]
    if h != 1024 or w != 1024:
        img = cv2.resize(img, (1024, 1024))
    return img


def preprocess_(input_img):
    img = np.array(input_img, dtype=np.uint8)
    h, w = img.shape[:2]
    if h != 512 or w != 512:
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    return img


def convert_ext(file_fullname):
    ext = os.path.splitext(file_fullname)[1].lower()
    if ext == ".jpg" or ext =='.jpeg' :    ext_str = "JPEG"
    elif ext == ".png":                    ext_str = "PNG"
    elif ext == ".dcm" or ext == ".dicom": ext_str = "DICOM"
    else: ext_str = "Unknown"
    return f'File Format: {ext_str}'

def convert_cnt(cnt_image) : return f'File Count: {cnt_image}'
def convert_pid(patient_id): return f'Patient ID: {patient_id}'


def dicom_info(ds):
    patient_id = ds.get("PatientID", "") or ""
    study_date = ds.get("StudyDate", "") or ds.get("AcquisitionDate", "")
    study_time = ds.get("StudyTime", "") or ds.get("AcquisitionTime", "")

    if len(study_date) == 8 and study_date.isdigit():
        date_str = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"
    else:
        date_str = "Unknown"

    if "." in study_time:
        study_time = study_time.split(".")[0]
    if len(study_time) == 6 and study_time.isdigit():
        time_str = f"{study_time[:2]}:{study_time[2:4]}:{study_time[4:]}"
    else:
        time_str = "Unknown"

    lines_str = f"{date_str} {time_str}"
    return patient_id, lines_str


def files_info(files):
    info_list = []
    hex_digits = '0123456789abcdef'

    skipped_files = []
    if files:
        files = list(files)
        for file in files:
            file_fullname = file.name.lower()
            file_name = os.path.basename(file_fullname)
            try:
                if file_fullname.endswith((".jpg", ".jpeg", ".png")):
                    pil_image = Image.open(file).convert("L")
                    pid = ''.join(random.choice(hex_digits) for _ in range(16))
                    dt_str = DEFAULT_DATETIME
                elif file_fullname.endswith((".dcm", ".dicom")):
                    dicom_data = pydicom.dcmread(file, force=True)
                    pid, dt_str = dicom_info(dicom_data)
                    image_array = dicom_data.pixel_array.astype(np.float32)
                    arr_min, arr_max = np.min(image_array), np.max(image_array)
                    if arr_max > arr_min:
                        image_array = (image_array - arr_min) / (arr_max - arr_min) * 255
                    else:
                        image_array = np.zeros_like(image_array)
                    image_array = image_array.astype(np.uint8)
                    pil_image = Image.fromarray(image_array).convert("L")
                else:
                    skipped_files.append(file_name)
                    continue

                pil_1024_image = preprocess(pil_image)

                info_list.append((dt_str, (pil_1024_image, file_name), pid))
            except Exception as e:
                print(f"[files_info] 파일 처리 실패: {file_name} — {e}")
                skipped_files.append(file_name)
                continue

    if skipped_files:
        gr.Warning(f"처리하지 못한 파일 {len(skipped_files)}개: {', '.join(skipped_files)}")

    if not info_list:
        raise gr.Error("처리 가능한 이미지 파일이 없습니다. 파일을 확인해주세요.")

    def parse_dt(dt_str):
        try:
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.min

    info_list.sort(key=lambda x: parse_dt(x[0]))

    sorted_dt, gallery_items, patient_ids = zip(*info_list)

    gallery_items = list(gallery_items)
    datetimes     = list(sorted_dt)
    patient_ids   = list(patient_ids)

    if len(set(patient_ids)) <= 1:
        if all(item == "" for item in patient_ids):
            gr.Warning("환자 ID가 없습니다. 임의로 생성합니다.")
            patient_ids.clear()
            patient_ids.append(''.join(random.choice(hex_digits) for _ in range(16)))

    cnt_image = len(gallery_items)

    print('segmentation and normalization start')
    origin_items = normalize_images([img for img, _ in gallery_items])

    inference_list = [[-1, -1, -1, -1] for _ in range(cnt_image)]

    ext_info = convert_ext(file_fullname)
    cnt_info = convert_cnt(cnt_image)
    pid_info = convert_pid(patient_ids[0])

    file_names = [name for _, name in gallery_items]
    gallery_display = [(img, str(i + 1)) for i, (img, _) in enumerate(gallery_items)]
    return gallery_display, origin_items, inference_list, datetimes, ext_info, cnt_info, pid_info, file_names
