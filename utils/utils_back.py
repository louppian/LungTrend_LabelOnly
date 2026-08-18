# utils/utils_back.py — Label-Only 버전

import gradio as gr
from utils.utils_severity import inference_with_overlay, get_severity_color
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import cv2
from glob import glob
import torchvision.transforms as transforms
import torch
from datetime import datetime
from matplotlib import gridspec
from config import RESULTS_DIR, DEFAULT_USER_ID, SOURCE_NPZ_DIR

to_tensor = transforms.ToTensor()
image_resize_transform = transforms.Resize(
    (512, 512), interpolation=transforms.InterpolationMode.BILINEAR
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROI_MAPPING = {"RT": 0, "LT": 1, "RB": 2, "LB": 3}


# ════════════════════════════════════════════════
# 섹션 전환
# ════════════════════════════════════════════════

def switch_section(files):
    if files:
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)


# ════════════════════════════════════════════════
# 갤러리 선택 관련
# ════════════════════════════════════════════════

def strip_gallery_prefix(label_text: str) -> str:
    cleaned = re.sub(r'^[✔✓]\s*', '', label_text)
    cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
    cleaned = re.sub(r'^[✔✓]\s*', '', cleaned)
    cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
    return cleaned.strip()


GALLERY_BORDER_COLOR = (0, 180, 255)
GALLERY_BORDER_THICKNESS = 4


def _add_selection_border(img):
    if img is None:
        return img
    if isinstance(img, str):
        try:
            img = Image.open(img).convert("RGB")
        except Exception as e:
            print(f"[gallery] 이미지 열기 실패: {img} — {e}")
            return img
    if isinstance(img, Image.Image):
        img = img.convert("RGB")
    arr = np.array(img, copy=True)
    if arr.ndim < 2:
        return img
    if arr.dtype.kind == 'f':
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        # NPZ 경로의 갤러리 이미지는 int32로 들어온다. cv2 는 CV_32S 를 받지 않는다.
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    h, w = arr.shape[:2]
    t = GALLERY_BORDER_THICKNESS
    cv2.rectangle(arr, (0, 0), (w - 1, h - 1), GALLERY_BORDER_COLOR, t)
    return arr


def on_select_gallery(gallery, img_list, datetime_list, selected_indices, evt: gr.SelectData):
    overlay_chk = False
    if not gallery:
        return None, None, None, None, selected_indices, gallery, False

    idx = evt.index
    if idx < 0 or idx >= len(gallery):
        return None, None, None, None, selected_indices, gallery, False

    on_select_gallery.cached_names = [strip_gallery_prefix(text) for _, text in gallery]

    if idx in selected_indices:
        selected_indices.remove(idx)
    else:
        if len(selected_indices) < 4:
            selected_indices.append(idx)
        else:
            selected_indices.pop(0)
            selected_indices.append(idx)

    updated_gallery = []
    for i, (img, _) in enumerate(gallery):
        original_filename = on_select_gallery.cached_names[i]
        is_selected = i in selected_indices
        if is_selected:
            new_text = f"✔ {i+1}"
            display_img = _add_selection_border(img)
        else:
            new_text = f"{i+1}"
            display_img = img
        updated_gallery.append((display_img, new_text))
    gallery_update = gr.update(value=updated_gallery)

    main_images = []
    for i in range(4):
        if i < len(selected_indices):
            main_images.append(img_list[selected_indices[i]])
        else:
            main_images.append(None)

    if len(selected_indices) >= 2:
        overlay_chk = True

    return (
        main_images[0], main_images[1], main_images[2], main_images[3],
        selected_indices, gallery_update, overlay_chk
    )


# ════════════════════════════════════════════════
# 오버레이 / 클릭 헬퍼
# ════════════════════════════════════════════════

def inference_overlay(image1, image2, image3, image4, orgn_img_list,
                      overlay_flag, selected_indices, inference_list):
    new_images = [None, None, None, None]
    if len(selected_indices) == 0:
        return image1, image2, image3, image4, overlay_flag
    for pos, idx in enumerate(selected_indices):
        processed = inference_with_overlay(orgn_img_list, inference_list, idx, idx)
        new_images[pos] = processed[0] if processed and len(processed) > 0 else orgn_img_list[idx]
    for pos in range(len(selected_indices), 4):
        new_images[pos] = None
    return new_images[0], new_images[1], new_images[2], new_images[3], overlay_flag


def get_quadrant_label(x, y, width, height):
    center_x = width / 2
    center_y = height / 2
    row = 0 if y < center_y else 1
    col = 0 if x < center_x else 1
    mapping = {(0, 0): "RT", (0, 1): "LT",
               (1, 0): "RB", (1, 1): "LB"}
    return mapping[(row, col)]


def draw_click_border(image, quadrant_label):
    if image is None:
        return None
    image_copy = image.copy()
    if image_copy.dtype.kind == 'f':
        if np.max(image_copy) <= 1.0:
            image_copy = (image_copy * 255).astype(np.uint8)
        else:
            image_copy = image_copy.astype(np.uint8)
    h, w = image_copy.shape[:2]
    half_w, half_h = w // 2, h // 2
    quadrant_boxes = {
        "RT": (0, 0, half_w, half_h),
        "LT": (half_w, 0, w - 1, half_h),
        "RB": (0, half_h, half_w, h - 1),
        "LB": (half_w, half_h, w - 1, h - 1),
    }
    if quadrant_label in quadrant_boxes:
        x1, y1, x2, y2 = quadrant_boxes[quadrant_label]
        cv2.rectangle(image_copy, (x1, y1), (x2, y2), (255, 255, 255), 1)
    return image_copy


def safe_copy_image(img, fallback=None):
    if img is None:
        return fallback
    if isinstance(img, np.ndarray):
        return img.copy()
    if isinstance(img, Image.Image):
        return img.copy()
    return fallback


def get_adjusted_clean_image(first_img, idx, placeholder_img,
                             orgn_img_list, inference_list, overlay_flag):
    if overlay_flag:
        processed = inference_with_overlay(orgn_img_list, inference_list, idx, idx)
        if processed and len(processed) > 0:
            return processed[0]
        else:
            return safe_copy_image(first_img, fallback=placeholder_img)
    else:
        return safe_copy_image(first_img, fallback=placeholder_img)


# ════════════════════════════════════════════════
# 이미지 클릭 핸들러 (통합)
# ════════════════════════════════════════════════

def _on_image_click(slot_index,
                    image1, image2, image3, image4,
                    first_image1, first_image2, first_image3, first_image4,
                    overlay_flag, selected_indices, click_info,
                    orgn_img_list, inference_list, evt: gr.SelectData):
    x, y = evt.index
    quadrant_label = get_quadrant_label(x, y, 512, 512)

    images = [image1, image2, image3, image4]
    firsts = [first_image1, first_image2, first_image3, first_image4]

    fresh = []
    for pos in range(4):
        idx = selected_indices[pos] if pos < len(selected_indices) else 0
        fresh.append(get_adjusted_clean_image(
            firsts[pos], idx, images[pos],
            orgn_img_list, inference_list, overlay_flag,
        ))

    used_idx = selected_indices[slot_index] if slot_index < len(selected_indices) else 0
    bordered = draw_click_border(fresh[slot_index], quadrant_label)
    new_click_info = f"Image {used_idx + 1}, {quadrant_label}"

    result_images = list(fresh)
    result_images[slot_index] = bordered

    return (
        new_click_info,
        result_images[0], result_images[1], result_images[2], result_images[3],
        fresh[0], fresh[1], fresh[2], fresh[3],
    )


def on_image1_click(image1, image2, image3, image4,
                    first_image1, first_image2, first_image3, first_image4,
                    overlay_flag, selected_indices, click_info,
                    orgn_img_list, inference_list, evt: gr.SelectData):
    return _on_image_click(0, image1, image2, image3, image4,
                           first_image1, first_image2, first_image3, first_image4,
                           overlay_flag, selected_indices, click_info,
                           orgn_img_list, inference_list, evt)


def on_image2_click(image1, image2, image3, image4,
                    first_image1, first_image2, first_image3, first_image4,
                    overlay_flag, selected_indices, click_info,
                    orgn_img_list, inference_list, evt: gr.SelectData):
    return _on_image_click(1, image1, image2, image3, image4,
                           first_image1, first_image2, first_image3, first_image4,
                           overlay_flag, selected_indices, click_info,
                           orgn_img_list, inference_list, evt)


def on_image3_click(image1, image2, image3, image4,
                    first_image1, first_image2, first_image3, first_image4,
                    overlay_flag, selected_indices, click_info,
                    orgn_img_list, inference_list, evt: gr.SelectData):
    return _on_image_click(2, image1, image2, image3, image4,
                           first_image1, first_image2, first_image3, first_image4,
                           overlay_flag, selected_indices, click_info,
                           orgn_img_list, inference_list, evt)


def on_image4_click(image1, image2, image3, image4,
                    first_image1, first_image2, first_image3, first_image4,
                    overlay_flag, selected_indices, click_info,
                    orgn_img_list, inference_list, evt: gr.SelectData):
    return _on_image_click(3, image1, image2, image3, image4,
                           first_image1, first_image2, first_image3, first_image4,
                           overlay_flag, selected_indices, click_info,
                           orgn_img_list, inference_list, evt)


# ════════════════════════════════════════════════
# Inference / Score 관련
# ════════════════════════════════════════════════

def update_dropdown_default(click_info, label_list_edit):
    if click_info == "Unknown" or not label_list_edit:
        return "X"
    try:
        parts = click_info.split(",")
        image_str = parts[0].strip()
        region_str = parts[1].strip()
        image_idx = int(image_str.split(" ")[1]) - 1
        region_idx = ROI_MAPPING.get(region_str, 0)
        current_score = label_list_edit[image_idx][region_idx]
        if current_score == -1:
            score_str = "X"
        else:
            score_str = str(current_score)
        return score_str
    except Exception as e:
        print("update_dropdown_default error:", e)
        return "X"


def update_score(new_score, click_info, label_list_edit):
    if click_info == "Unknown" or not label_list_edit:
        return label_list_edit
    try:
        parts = click_info.split(",")
        image_str = parts[0].strip()
        region_str = parts[1].strip()
        image_idx = int(image_str.split(" ")[1]) - 1
        region_idx = ROI_MAPPING.get(region_str, 0)
        cleaned_score = new_score.replace("*", "").strip()
        if cleaned_score.upper() == "X":
            label_list_edit[image_idx][region_idx] = -1
            return label_list_edit
        score = int(cleaned_score)
        if not 0 <= score <= 4:
            gr.Warning(f"점수는 0~4 범위여야 합니다: {cleaned_score}")
            return label_list_edit
        label_list_edit[image_idx][region_idx] = score
    except ValueError:
        gr.Warning(f"숫자가 아닌 점수는 저장되지 않습니다: {new_score}")
    except Exception as e:
        print("Error : update_score: ", e)
    return label_list_edit


# ════════════════════════════════════════════════
# 그래프
# ════════════════════════════════════════════════

def update_graphs(label_list_edit, click_info, selected_indices):
    if not label_list_edit:
        return None

    plt.close('all')

    n_imgs = len(label_list_edit)
    xs = list(range(1, n_imgs + 1))

    upd_i = upd_roi = None
    if click_info != "Unknown":
        try:
            img_str, roi_str = [p.strip() for p in click_info.split(",")]
            upd_i = int(img_str.split(" ")[1]) - 1
            upd_roi = ROI_MAPPING.get(roi_str)
        except Exception:
            pass

    series = [(label_list_edit, "User Labels", "#E26114", "-", 3)]

    return _build_graph(series, n_imgs, xs, selected_indices, upd_i, upd_roi)


def _build_graph(series_list, n_imgs, xs, selected_indices, upd_i, upd_roi):
    roi_names = ['RT', 'LT', 'RB', 'LB']

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.2, 1, 1])
    axT = fig.add_subplot(gs[0, :])
    axRois = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
              fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])]
    fig.patch.set_facecolor("#0F0F11")

    handles, labels = [], []

    # ── Total ──
    axT.set_facecolor("#0F0F11")
    first_vals = None
    for data, label, color, style, zorder in series_list:
        if data and len(data) == n_imgs:
            tot = [sum(max(0, v) for v in r) for r in data]
            axT.plot(xs, tot, f's{style}', ms=4, lw=1.2, color=color, label=f"Total {label}", zorder=zorder)
            if first_vals is None:
                first_vals = tot

    if upd_i is not None:
        for data, label, color, style, zorder in series_list:
            if data and len(data) == n_imgs:
                tot = [sum(max(0, v) for v in r) for r in data]
                axT.scatter(upd_i + 1, tot[upd_i], s=60, color="#FF00FF", zorder=5)
                break

    for sel in selected_indices or []:
        if isinstance(sel, int) and 0 <= sel < n_imgs and first_vals:
            y = first_vals[sel]
            axT.annotate("", xy=(sel + 1, y), xytext=(sel + 1, y + 1),
                         arrowprops=dict(arrowstyle="->", color="red", lw=2))

    axT.set_xlabel("Shots", color='white')
    axT.set_ylabel("Severity", color='white')
    axT.set_xticks(xs)
    axT.set_yticks(range(0, 17, 2))
    axT.tick_params(axis='both', colors='white')
    for sp in axT.spines.values():
        sp.set_color("white")

    # ── 각 ROI ──
    for i, (ax, name) in enumerate(zip(axRois, roi_names)):
        ax.set_facecolor("#0F0F11")
        first_roi = None
        for data, label, color, style, zorder in series_list:
            if data and len(data) == n_imgs:
                vals = [max(0, row[i]) for row in data]
                h, = ax.plot(xs, vals, f'o{style}', ms=3, lw=1, color=color, label=label, zorder=zorder)
                handles.append(h)
                labels.append(label)
                if first_roi is None:
                    first_roi = vals
                if upd_roi == i and upd_i is not None:
                    ax.scatter(upd_i + 1, vals[upd_i], color='#FF00FF', s=40, zorder=5)

        for sel in selected_indices or []:
            if isinstance(sel, int) and 0 <= sel < n_imgs and first_roi:
                ax.annotate("", xy=(sel + 1, first_roi[sel]),
                            xytext=(sel + 1, first_roi[sel] + 0.5),
                            arrowprops=dict(arrowstyle="->", color="red", lw=2))

        ax.set_title(name, color='white')
        ax.set_xlabel("Shots", color='white')
        ax.set_ylabel("Severity", color='white')
        ax.set_xticks(xs)
        ax.set_yticks(range(5))
        ax.tick_params(axis='both', colors='white')
        if i % 2 == 1:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
        for sp in ax.spines.values():
            sp.set_color("white")

    # ── 범례 ──
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    if uniq:
        legend = fig.legend(list(uniq.values()), list(uniq.keys()),
                            loc='upper center', bbox_to_anchor=(0.5, 0.94),
                            ncol=len(uniq), frameon=False)
        for t in legend.get_texts():
            t.set_color("white")

    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(top=0.88)
    return fig


# ════════════════════════════════════════════════
# 이미지 유틸
# ════════════════════════════════════════════════

def ensure_pil_image(obj):
    if obj is None:
        return None
    if isinstance(obj, Image.Image):
        return obj
    if isinstance(obj, np.ndarray):
        return Image.fromarray(obj)
    if isinstance(obj, str):
        if os.path.exists(obj):
            return Image.open(obj)
        else:
            raise ValueError(f"해당 경로를 찾을 수 없습니다: {obj}")
    if isinstance(obj, dict) and "name" in obj:
        path = obj["name"]
        if os.path.exists(path):
            return Image.open(path)
        else:
            raise ValueError(f"해당 path를 찾을 수 없습니다: {path}")
    raise ValueError(f"지원하지 않는 이미지 타입: {type(obj)}")


def ensure_four(img_list, fallback):
    if img_list is None or len(img_list) < 4:
        if fallback is None or len(fallback) < 4:
            return [None, None, None, None]
        else:
            return fallback
    return img_list


# ════════════════════════════════════════════════
# 메인 이미지 새로고침 (Overlay 토글 시)
# ════════════════════════════════════════════════

def refresh_main_images(orgn_img_list, inference_list, selected_indices,
                        overlay_flag, popup_click_info):
    new_main_images = [None, None, None, None]
    if not selected_indices or len(selected_indices) == 0:
        return new_main_images

    parse_idx = None
    parse_quad = None
    if popup_click_info and "," in popup_click_info:
        try:
            parts = popup_click_info.split(",")
            image_part = parts[0].strip()
            quad_part = parts[1].strip()
            x_num = image_part.split(" ")[1]
            parse_idx = int(x_num) - 1
            parse_quad = quad_part
        except Exception:
            pass

    for pos, idx in enumerate(selected_indices):
        if pos >= 4:
            break

        if overlay_flag:
            processed = inference_with_overlay(
                orgn_img_list, inference_list, idx, idx
            )
            final_img = processed[0] if processed else orgn_img_list[idx]
        else:
            final_img = orgn_img_list[idx]

        if parse_idx == idx and parse_quad:
            final_img = draw_click_border(final_img, parse_quad)
        new_main_images[pos] = final_img

    return new_main_images


# ════════════════════════════════════════════════
# 토글 버튼
# ════════════════════════════════════════════════

def toggle_overlay(orgn_img_list, inference_list, selected_indices,
                   popup_click_info, overlay_flag):
    new_overlay_flag = not overlay_flag
    main_images = refresh_main_images(
        orgn_img_list, inference_list, selected_indices,
        new_overlay_flag, popup_click_info,
    )
    btn_label = "[ O ]" if new_overlay_flag else "O"
    return (
        main_images[0], main_images[1], main_images[2], main_images[3],
        new_overlay_flag, btn_label
    )


# ════════════════════════════════════════════════
# 리셋
# ════════════════════════════════════════════════

def reset_page():
    print("페이지를 초기화 합니다")
    if hasattr(on_select_gallery, "cached_names"):
        delattr(on_select_gallery, "cached_names")

    return (
        gr.update(value=None),          # image_files
        gr.update(visible=True),        # upload_section
        gr.update(visible=False),       # main_section
        gr.update(value=[]),            # gallery

        gr.update(value=None), gr.update(value=None),
        gr.update(value=None), gr.update(value=None),

        gr.update(value=[]),            # selected_indices
        gr.update(value=None),          # orgn_img_list

        gr.update(value=None), gr.update(value=None),
        gr.update(value=None), gr.update(value=None),

        gr.update(value=""),            # patient_id_info

        gr.update(value=None),          # label_list_edit

        gr.update(value="Unknown"),     # popup_click_info
        gr.update(value="X"),           # popup_score_edit

        gr.update(value="File Format: Unknown"),
        gr.update(value="File Count: Unknown"),
        gr.update(value="Patient ID : "),
        gr.update(value=DEFAULT_USER_ID),

        gr.update(value=True),          # overlay_flag
        gr.update(value=False),         # file_is_npz_state
        gr.update(value=[]),            # file_names_state
        gr.update(value=None),          # source_npz_name
    )


# ════════════════════════════════════════════════
# NPZ + 엑셀 내보내기
# ════════════════════════════════════════════════

def _safe(txt):
    return re.sub(r"[\\/:*?\"<>|]", "_", str(txt).strip())


def _clean_gallery_label(label: str) -> str:
    cleaned = re.sub(r"^[✔✓]\s*", "", label)
    cleaned = re.sub(r"^\d+\.\s*", "", cleaned)
    return cleaned.strip()


def count_all_zero_images(label_list_edit):
    """4분면이 모두 0인 이미지 수 — 판독하지 않고 넘긴 장수 확인용."""
    if not label_list_edit:
        return 0
    return sum(1 for row in label_list_edit if all(int(v) == 0 for v in row))


def remaining_source_count():
    return len(glob(os.path.join(SOURCE_NPZ_DIR, "*.npz")))


def _source_npz_path(source_npz_name):
    """판독 대기 폴더에 있는 원본 npz 경로. 없으면 None."""
    if not source_npz_name:
        return None
    path = os.path.join(SOURCE_NPZ_DIR, os.path.basename(str(source_npz_name)))
    return path if os.path.exists(path) else None


def _verify_result_npz(path, scores, expect_images):
    """원본을 지우기 전에 합쳐 저장한 결과가 온전한지 확인."""
    try:
        with np.load(path, allow_pickle=True) as d:
            score_key = "labels" if "labels" in d.files else "inference_edit"
            saved = np.array(d[score_key].tolist()).tolist()
            if saved != [[int(v) for v in row] for row in scores]:
                print("[VERIFY] 점수가 일치하지 않음")
                return False
            if expect_images is not None:
                if "original_images" not in d.files:
                    print("[VERIFY] original_images 누락")
                    return False
                if np.asarray(d["original_images"]).shape != np.asarray(expect_images).shape:
                    print("[VERIFY] 이미지 형상 불일치")
                    return False
        return True
    except Exception as e:
        print(f"[VERIFY] 결과 파일을 읽지 못함: {e}")
        return False


def export_npy(label_list_edit,
               ext_pid, user_id_input, file_names=None,
               source_npz_name=None, orgn_img_list=None):

    if label_list_edit:
        unlabeled = []
        for i, row in enumerate(label_list_edit):
            for j, v in enumerate(row):
                if v == -1:
                    roi_name = ["RT", "LT", "RB", "LB"][j]
                    unlabeled.append(f"Image {i+1}-{roi_name}")
        if unlabeled:
            gr.Warning(f"레이블되지 않은 영역이 {len(unlabeled)}개 있습니다: {', '.join(unlabeled[:10])}{'...' if len(unlabeled) > 10 else ''}")

    patient_id = _safe(ext_pid.split("Patient ID:")[-1]) if "Patient ID:" in ext_pid else _safe(ext_pid)
    user_id = _safe(user_id_input or "Unknown") or "Unknown"

    now = datetime.now()
    yymmdd = now.strftime("%y%m%d")
    hhmmss = now.strftime("%H%M%S")

    # results/ 는 평평하게 유지하고, 판독이 끝난 건마다 파일 하나만 남긴다.
    # 원본 npz 의 이미지에 판독 점수를 얹어 저장한 뒤 원본을 지운다.
    BASE_DIR = RESULTS_DIR
    os.makedirs(BASE_DIR, exist_ok=True)
    npz_path = os.path.join(BASE_DIR, f"{patient_id}_{now.strftime('%m%d%H%M')}.npz")
    root, ext = os.path.splitext(npz_path)
    n = 1
    while os.path.exists(npz_path):
        npz_path = f"{root}_{n}{ext}"
        n += 1

    # 원본이 있으면 metadata를 물려받고, 이미 삭제된 뒤의 재-export는
    # 현재 화면 state의 이미지/라벨로 완전한 npz를 다시 구성한다.
    src_path = _source_npz_path(source_npz_name)
    export_data = {}
    if src_path:
        with np.load(src_path, allow_pickle=True) as src:
            export_data = {
                k: src[k] for k in src.files
                if k not in ("inference", "inference_edit", "labels")
            }
    elif orgn_img_list is not None:
        export_data["original_images"] = np.array(orgn_img_list)
        export_data["gallery"] = np.array([str(x) for x in (file_names or [])])

    # Label-Only 저장 포맷에서는 판독 점수를 labels 로 저장한다.
    # 기존 inference/inference_edit key 는 읽기 호환용으로만 유지한다.
    export_data.update({
        "patient_id":     patient_id,
        "user_id":        user_id,
        "timestamp":      now.strftime("%Y-%m-%d %H:%M:%S"),
        "source_npz":     str(source_npz_name or ""),
        "image_ids":      np.array([str(x) for x in (file_names or [])]),
        "labels":         label_list_edit,
    })
    np.savez(npz_path, **export_data)
    print(f"[NPZ] saved → {npz_path}")

    excel_path = os.path.join(BASE_DIR, "roi_export.xlsx")

    excel_locked = False
    if os.path.exists(excel_path):
        try:
            with open(excel_path, "a"):
                pass
        except PermissionError:
            excel_locked = True
            gr.Warning("roi_export.xlsx가 열려 있습니다. 결과는 별도 파일로 저장됩니다.")

    MAX_EXPORT_IMAGES = 30
    if label_list_edit and len(label_list_edit) > MAX_EXPORT_IMAGES:
        gr.Warning(f"엑셀에는 최대 {MAX_EXPORT_IMAGES}장까지만 저장됩니다 (현재 {len(label_list_edit)}장).")
    columns = ["PatientID", "UserID", "Timestamp"]
    for i in range(1, MAX_EXPORT_IMAGES + 1):
        columns += [f"RT{i}", f"LT{i}", f"RB{i}", f"LB{i}"]

    def _row(scores, label):
        row = [patient_id, label, now.strftime("%Y-%m-%d %H:%M:%S")]
        scores = scores or []
        for i in range(MAX_EXPORT_IMAGES):
            if i < len(scores):
                row.extend(scores[i])
            else:
                row.extend([""] * 4)
        return row

    rows_new = [
        _row(label_list_edit, user_id),
    ]
    df_new = pd.DataFrame(rows_new, columns=columns)

    old_frames = []
    if os.path.exists(excel_path):
        old_frames.append(pd.read_excel(excel_path))
    merged_fallbacks = []
    for fb in sorted(glob(os.path.join(BASE_DIR, "roi_export_*.xlsx"))):
        try:
            old_frames.append(pd.read_excel(fb))
            merged_fallbacks.append(fb)
        except Exception as e:
            print(f"[Excel] 폴백 파일 읽기 실패(건너뜀): {fb} — {e}")

    if old_frames:
        df_old = pd.concat(old_frames, ignore_index=True).drop_duplicates()
        mask = ((df_old["PatientID"] == patient_id)
                & (df_old["UserID"] == user_id)
                & (df_old["Timestamp"] == now.strftime("%Y-%m-%d %H:%M:%S")))
        df_old = df_old.loc[~mask]
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    try:
        df_final.to_excel(excel_path, index=False)
        print(f"[Excel] updated → {excel_path}")
        written_path = excel_path
    except PermissionError:
        written_path = os.path.join(BASE_DIR, f"roi_export_{yymmdd}_{hhmmss}.xlsx")
        df_final.to_excel(written_path, index=False)
        print(f"[Excel] locked, fallback → {written_path}")
        gr.Warning(f"roi_export.xlsx가 잠겨 있어 {os.path.basename(written_path)}로 저장했습니다. "
                   "Excel을 닫은 뒤 다음 export 시 원본에 자동 병합됩니다.")

    for fb in merged_fallbacks:
        if os.path.abspath(fb) != os.path.abspath(written_path):
            try:
                os.remove(fb)
                print(f"[Excel] 병합 완료된 폴백 파일 삭제: {fb}")
            except Exception as e:
                print(f"[Excel] 폴백 파일 삭제 실패(무시): {fb} — {e}")

    # 결과가 온전한 것을 확인한 뒤에만 원본을 지운다. 확인에 실패하면 원본을
    # 그대로 두므로 판독을 다시 할 수 있다.
    if src_path and _verify_result_npz(npz_path, label_list_edit,
                                       export_data.get("original_images")):
        try:
            os.remove(src_path)
            print(f"[DONE] {os.path.basename(src_path)} → {os.path.basename(npz_path)} (원본 삭제)")
            gr.Info(f"Export Complete - {os.path.basename(npz_path)} 저장, "
                    f"남은 판독 {remaining_source_count()}개")
            return
        except Exception as e:
            gr.Warning(f"원본 npz 삭제 실패 (결과는 저장됨): {e}")
    elif src_path:
        gr.Warning("결과 파일 검증에 실패해 원본을 남겨 둡니다. 다시 export 하세요.")

    gr.Info("Export Complete")
