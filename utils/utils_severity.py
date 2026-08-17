# utils_severity.py : 중증도 시각화 모듈 (Label-Only)
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

to_tensor = transforms.ToTensor()


def get_severity_color(severity):
    if severity < 0:
        return (128, 128, 128)
    severity = max(0, min(4, severity))
    mapping = {
        0: (255, 255, 255),
        1: (0, 255, 0),
        2: (255, 255, 0),
        3: (255, 100, 0),
        4: (255, 0, 0),
    }
    return mapping[severity]


def inference_with_overlay(img_list, inference_list, selected_idx1, selected_idx2):
    def get_image(idx):
        if idx is not None and idx < len(img_list):
            return img_list[idx]
        return None

    def get_inference(idx):
        if idx is not None and idx < len(inference_list):
            return inference_list[idx]
        return None

    img1 = get_image(selected_idx1)
    img2 = get_image(selected_idx2)
    inf1 = get_inference(selected_idx1)
    inf2 = get_inference(selected_idx2)

    def image_to_tensor(image):
        if image is not None:
            arr = np.array(image)
            arr = np.ascontiguousarray(arr).astype(np.float32)
            # 0~255 스케일 입력(NPZ 로드 등)을 0~1로 정규화 — 아래 *255 시
            # uint8 오버플로 랩(256-x)으로 흑백반전되는 것을 방지
            if arr.max() > 1.5:
                arr = arr / 255.0
            arr = np.clip(arr, 0.0, 1.0)
            return to_tensor(arr).to(device)
        return None

    tensor1 = image_to_tensor(img1)
    tensor2 = image_to_tensor(img2)

    tensors = []
    inferences = []
    if tensor1 is not None:
        tensors.append(tensor1)
        inferences.append(inf1)
    if tensor2 is not None:
        tensors.append(tensor2)
        inferences.append(inf2)
    if not tensors:
        return None

    data_tensor = torch.stack(tensors, dim=0)

    batch = data_tensor.shape[0]
    overlay_images = []
    data_tensor_np = data_tensor.cpu().numpy()

    quad_map = {
        (0, 0): ("RT", 0),
        (0, 1): ("LT", 1),
        (1, 0): ("RB", 2),
        (1, 1): ("LB", 3),
    }

    for i in range(batch):
        original_img = np.transpose(data_tensor_np[i], (1, 2, 0))
        original_img = (original_img * 255).astype(np.uint8)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

        overlay = original_img.copy()

        cell_h = 512 // 2
        cell_w = 512 // 2
        cv2.line(overlay, (0, cell_h), (512, cell_h), (0, 0, 0), 1)
        cv2.line(overlay, (cell_w, 0), (cell_w, 512), (0, 0, 0), 1)

        margin = 10
        text_offset = 30
        positions = {
            (0, 0): (margin, margin + text_offset),
            (0, 1): (cell_w * 2 - margin * 7, margin + text_offset),
            (1, 0): (margin, cell_h * 2 - margin),
            (1, 1): (cell_w * 2 - margin * 7, cell_h * 2 - margin),
        }

        inf_i = inferences[i]
        if inf_i is None or len(inf_i) < 4:
            overlay_images.append(overlay)
            continue

        for (row, col), (quad_label, idx) in quad_map.items():
            severity_val = int(inf_i[idx])
            if severity_val < 0:
                display_text = f"{quad_label}: X"
            else:
                display_text = f"{quad_label}: {severity_val}"
            text_color = get_severity_color(severity_val)
            text_x, text_y = positions[(row, col)]
            cv2.putText(
                overlay,
                display_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2,
                cv2.LINE_AA,
            )

        overlay_images.append(overlay)

    return overlay_images
