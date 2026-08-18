# app.py — Label-Only 버전 (AI 추론 제거)

import warnings
warnings.filterwarnings("ignore", message="input's size at dim=.*does not match num_features")
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")

import gradio_client.utils as _gc_utils
_orig_json_schema = _gc_utils._json_schema_to_python_type
def _patched_json_schema(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _orig_json_schema(schema, defs)
_gc_utils._json_schema_to_python_type = _patched_json_schema

import gradio as gr
import os
import re
from utils.utils_image import files_info, normalize_images
from utils.utils_back import (
    switch_section, on_select_gallery, inference_overlay,
    on_image1_click, on_image2_click, on_image3_click, on_image4_click,
    update_score, update_dropdown_default, toggle_overlay,
    update_graphs, export_npy, reset_page, refresh_main_images,
    count_all_zero_images,
)
from config import DEFAULT_USER_ID, ZERO_ROW_WARN_THRESHOLD
import numpy as np
from datetime import datetime
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

css = """
#main-content { flex: 5 !important; }
#main-image { width: 400px !important; height: 400px !important; margin: 0 !important; }
#reset-button { width: 200px !important; margin: 0 !important; }
#gallery img { width: 200px !important; height: 200px !important; object-fit: contain !important; cursor: pointer !important; }
#sidebar { flex: 1 !important; position: fixed !important; right: 0 !important; top: 0 !important; bottom: 0 !important; width: 400px !important; background-color: #f6f7f9 !important; box-shadow: -2px 0 5px rgba(0,0,0,0.1) !important; overflow-y: auto !important; z-index: 100 !important; }
#toggle-wrapper { display: flex !important; flex-direction: row-reverse !important; align-items: center !important; justify-content: flex-end !important; gap: 8px !important; }
#colorbar, #graph { border: none !important; box-shadow: none !important; }
#popup-container { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.5); width: 300px; text-align: center; z-index: 3000 !important; overflow: visible !important; }
#popup-container ul.options.svelte-y6qw75 { position: fixed; top: 50%; left: 50%; z-index: 9999 !important; overflow: visible !important; background-color: #FFFFFF !important; color: #000000 !important; opacity: 1 !important; }
#popup-container ul.options.svelte-y6qw75 li { background-color: #000000 !important; color: #FFFFFF !important; opacity: 1 !important; }
#popup-container ul.options.svelte-y6qw75 li:hover { background-color: #EEEEEE !important; }
#overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.5); z-index: 2999; }
.dark #sidebar { background-color: #2a2b32 !important; box-shadow: -2px 0 5px rgba(0,0,0,0.3) !important; }
"""


def parse_after_colon(txt: str) -> str:
    if ":" in txt:
        return txt.split(":", 1)[-1].strip()
    return txt.strip()


def convert_to_image(item):
    if isinstance(item, np.ndarray):
        return item
    elif isinstance(item, list):
        try:
            arr = np.array(item)
            if arr.ndim in (2, 3):
                return arr
            return None
        except Exception as e:
            print("Error converting list to image:", e)
            return None
    elif isinstance(item, str):
        if os.path.exists(item):
            try:
                return np.array(Image.open(item).convert("RGB"))
            except Exception as e:
                print("Error loading image from path:", item, e)
                return None
        return None
    elif isinstance(item, Image.Image):
        return np.array(item)
    return item


def process_upload(files):
    if not files:
        return (None, None, "",
                "Unknown", "Unknown", "Unknown",
                "Patient ID : ", "File Format: Unknown", "File Count: Unknown",
                DEFAULT_USER_ID, False, [], [], [], None)

    first_file = files[0]
    fname = first_file.name.lower()

    if fname.endswith(".npy") or fname.endswith(".npz"):
        try:
            data = np.load(first_file.name, allow_pickle=True)
        except Exception as e:
            return (None, None, "",
                    "Unknown", f"npy (error: {e})", "Unknown",
                    "Patient ID : ", "File Format: Unknown", "File Count: Unknown",
                    DEFAULT_USER_ID, False, [], [], [], None)

        gallery_labels = None
        if isinstance(data, np.lib.npyio.NpzFile):
            orgn_images = data["original_images"] if "original_images" in data.files else None
            gallery_images = orgn_images
            # make_npz.py 는 gallery 에 uid2 라벨만 담는다 (구버전은 이미지+라벨 2열)
            if "gallery" in data.files:
                raw_labels = data["gallery"]
                if raw_labels.ndim == 1 and raw_labels.dtype.kind in "US":
                    gallery_labels = [str(x) for x in raw_labels.tolist()]
            patient_id = data["patient_id"] if "patient_id" in data.files else "Unknown"
            labels_loaded = data["labels"] if "labels" in data.files else (
                data["inference_edit"] if "inference_edit" in data.files else (
                    data["inference"] if "inference" in data.files else []))
            user_id_loaded = data["user_id"] if "user_id" in data.files else "Unknown"
            timestamp_loaded = data["timestamp"] if "timestamp" in data.files else np.datetime_as_string(np.datetime64('now'))
        elif isinstance(data, dict):
            orgn_images = data.get("original_images", None)
            gallery_images = orgn_images if orgn_images is not None else data.get("gallery", None)
            patient_id = data.get("patient_id", "Unknown")
            labels_loaded = data.get("labels", data.get("inference_edit", data.get("inference", [])))
            user_id_loaded = data.get("user_id", "Unknown")
            timestamp_loaded = data.get("timestamp", np.datetime_as_string(np.datetime64('now')))
        elif isinstance(data, np.ndarray) and data.shape == () and isinstance(data.item(), dict):
            data = data.item()
            orgn_images = data.get("original_images", None)
            gallery_images = orgn_images if orgn_images is not None else data.get("gallery", None)
            patient_id = data.get("patient_id", "Unknown")
            labels_loaded = data.get("labels", data.get("inference_edit", data.get("inference", [])))
            user_id_loaded = data.get("user_id", "Unknown")
            timestamp_loaded = data.get("timestamp", np.datetime_as_string(np.datetime64('now')))
        else:
            gallery_images = None
            orgn_images = None
            patient_id = "Unknown"
            labels_loaded = []
            user_id_loaded = DEFAULT_USER_ID
            timestamp_loaded = np.datetime_as_string(np.datetime64('now'))

        if isinstance(labels_loaded, np.ndarray):
            labels_loaded = labels_loaded.tolist()
        if isinstance(gallery_images, np.ndarray):
            gallery_images = gallery_images.tolist()
        if isinstance(orgn_images, np.ndarray):
            orgn_images = orgn_images.tolist()

        def _strip_check_prefix(label, fallback):
            if not label or label.strip() == "":
                label = fallback
            name = os.path.basename(label)
            name = re.sub(r"^[✔✓]\s*", "", name)
            name = re.sub(r"^\d+\.\s*", "", name)
            return name.strip()

        if gallery_images is not None:
            new_gallery, new_orgn, npz_file_names = [], [], []
            for i, item in enumerate(gallery_images):
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    img_raw, label_raw = item
                else:
                    img_raw, label_raw = item, ""
                img_arr = convert_to_image(img_raw)
                if gallery_labels is not None and i < len(gallery_labels):
                    label_raw = gallery_labels[i]
                clean_lbl = _strip_check_prefix(label_raw, first_file.name)
                npz_file_names.append(clean_lbl)
                new_gallery.append((img_arr, str(i + 1)))
                new_orgn.append(img_arr)
            gallery_images = new_gallery
            # 갤러리는 1024 원본, 메인 뷰는 segmentation + STN 정렬한 512
            # (DICOM/PNG 업로드 경로와 동일)
            orgn_images = normalize_images(new_orgn)
        else:
            npz_file_names = []

        image_count = len(gallery_images) if gallery_images is not None else 0
        file_is_npz = fname.endswith(".npz")

        return (
            gallery_images,
            orgn_images,
            timestamp_loaded,
            f"{patient_id}",
            "npz" if file_is_npz else "npy",
            f"{image_count}",
            f"Patient ID : {patient_id}",
            f"File Format: {'npz' if file_is_npz else 'npy'}",
            f"File Count: {image_count}",
            user_id_loaded,
            file_is_npz,
            labels_loaded,
            [],
            npz_file_names,
            os.path.basename(first_file.name),
        )
    else:
        result = files_info(files)
        # files_info returns: gallery, origin_items, inference_list, datetimes, ext_info, cnt_info, pid_info, file_names
        return (
            result[0],                        # gallery
            result[1],                        # orgn_img_list
            result[3],                        # datetime_state
            parse_after_colon(result[6]),     # patient_id_info
            parse_after_colon(result[4]),     # file_type_info
            parse_after_colon(result[5]),     # image_count_info
            result[6],                        # ext_pid
            result[4],                        # ext_str
            result[5],                        # ext_cnt
            DEFAULT_USER_ID,                   # user_id_loaded
            False,                             # file_is_npz
            result[2],                        # label_list_edit
            [],                               # selected_indices
            result[7],                        # file_names_state
            None,                             # source_npz_name (npz 업로드일 때만 채움)
        )


# ════════════════════════════════════════════════
# Gradio UI
# ════════════════════════════════════════════════

with gr.Blocks(css=css) as demo:

    # 상태 변수
    overlay_flag = gr.State(value=True)
    selected_indices = gr.State(value=[])
    orgn_img_list = gr.State(value=None)
    label_list_edit = gr.State(value=None)
    datetime_state = gr.State(value=None)
    first_image1 = gr.State(value=None)
    first_image2 = gr.State(value=None)
    first_image3 = gr.State(value=None)
    first_image4 = gr.State(value=None)
    file_is_npz_state = gr.State(value=False)
    file_names_state = gr.State(value=[])
    source_npz_name = gr.State(value=None)

    # ── 업로드 섹션 ──
    with gr.Column(visible=True) as upload_section:
        with gr.Row():
            with gr.Column():
                gr.Markdown("## File Upload")
                image_files = gr.File(
                    label="Upload Dicom Files", file_count="multiple",
                    file_types=[".dcm", ".dicom", ".jpg", ".jpeg", ".png", ".npy", ".npz"]
                )
            with gr.Column():
                gr.Markdown("## Loaded File Information")
                user_id_input = gr.Textbox(label="User ID", value=DEFAULT_USER_ID, interactive=False)
                patient_id_info = gr.Textbox(label="Patient ID", placeholder="Patient ID: Unknown")
                file_type_info = gr.Textbox(label="File Type", placeholder="File Type: Unknown", interactive=False)
                image_count_info = gr.Textbox(label="Image Count", placeholder="Image Count: Unknown", interactive=False)
        with gr.Row():
            process_button = gr.Button("Process Files")

    # ── 메인 섹션 ──
    with gr.Row(visible=False) as main_section:
        with gr.Column(elem_id="main-content"):
            with gr.Row(max_height=100):
                ext_pid = gr.Markdown(value="Patient ID : ")
                ext_str = gr.Markdown(value="File Format: Unknown")
                ext_cnt = gr.Markdown(value="File Count: Unknown")
                overlay_button = gr.Button("O", elem_id="overlay-button")
                reset_button = gr.Button("Reset", elem_id="reset-button")
                export_button = gr.Button("Export")
            with gr.Row(max_height=400):
                gallery = gr.Gallery(columns=10, allow_preview=False, show_label=False, height=300)
            with gr.Row():
                image1 = gr.Image(show_label=False, show_download_button=False, show_fullscreen_button=False, sources=[], elem_id='main-image')
                image2 = gr.Image(show_label=False, show_download_button=False, show_fullscreen_button=False, sources=[], elem_id='main-image')
                image3 = gr.Image(show_label=False, show_download_button=False, show_fullscreen_button=False, sources=[], elem_id='main-image')
                image4 = gr.Image(show_label=False, show_download_button=False, show_fullscreen_button=False, sources=[], elem_id='main-image')
            graph_scores = gr.Plot(show_label=False, elem_id="graph")

    # ── 팝업 ──
    with gr.Column(visible=False, elem_id="popup-container") as popup:
        popup_click_info = gr.Textbox(label="Selected Region", value="Unknown")
        popup_score_edit = gr.Dropdown(
            ["X", "0", "1", "2", "3", "4"], allow_custom_value=True, label="Score Edit"
        )
        with gr.Row():
            popup_save = gr.Button("Save")
            popup_cancel = gr.Button("Cancel")

    # ── Export 확인 팝업 (0000 이 임계값 이상일 때) ──
    with gr.Column(visible=False, elem_id="popup-container") as export_confirm:
        export_confirm_msg = gr.Markdown()
        with gr.Row():
            export_confirm_ok = gr.Button("Export")
            export_confirm_cancel = gr.Button("Cancel")

    # ════════════════════════════════════════════
    # 이벤트 바인딩
    # ════════════════════════════════════════════

    # 파일 업로드
    image_files.change(
        fn=process_upload,
        inputs=[image_files],
        outputs=[
            gallery, orgn_img_list, datetime_state,
            patient_id_info, file_type_info, image_count_info,
            ext_pid, ext_str, ext_cnt,
            user_id_input, file_is_npz_state, label_list_edit, selected_indices,
            file_names_state, source_npz_name
        ],
    )

    # Process Files 버튼
    process_button.click(
        fn=switch_section,
        inputs=[image_files],
        outputs=[upload_section, main_section],
    )

    # 갤러리 선택 → 메인 이미지 업데이트 → 오버레이 → first_image 캐싱 → 그래프
    gallery.select(
        fn=on_select_gallery,
        inputs=[gallery, orgn_img_list, datetime_state, selected_indices],
        outputs=[image1, image2, image3, image4, selected_indices, gallery, overlay_flag],
    ).then(
        fn=inference_overlay,
        inputs=[
            image1, image2, image3, image4,
            orgn_img_list,
            overlay_flag, selected_indices, label_list_edit,
        ],
        outputs=[image1, image2, image3, image4, overlay_flag],
    ).then(
        fn=lambda i1, i2, i3, i4, sel_idx, orgn_imgs: (
            orgn_imgs[sel_idx[0]] if len(sel_idx) > 0 else None,
            orgn_imgs[sel_idx[1]] if len(sel_idx) > 1 else None,
            orgn_imgs[sel_idx[2]] if len(sel_idx) > 2 else None,
            orgn_imgs[sel_idx[3]] if len(sel_idx) > 3 else None,
        ),
        inputs=[image1, image2, image3, image4, selected_indices, orgn_img_list],
        outputs=[first_image1, first_image2, first_image3, first_image4],
    ).then(
        fn=update_graphs,
        inputs=[label_list_edit, popup_click_info, selected_indices],
        outputs=[graph_scores],
    )

    # 이미지 클릭 → 팝업 표시
    for img, handler in zip(
        [image1, image2, image3, image4],
        [on_image1_click, on_image2_click, on_image3_click, on_image4_click],
    ):
        img.select(
            fn=handler,
            inputs=[
                image1, image2, image3, image4,
                first_image1, first_image2, first_image3, first_image4,
                overlay_flag, selected_indices, popup_click_info,
                orgn_img_list, label_list_edit,
            ],
            outputs=[
                popup_click_info,
                image1, image2, image3, image4,
                first_image1, first_image2, first_image3, first_image4,
            ],
        ).then(
            fn=lambda info: gr.update(visible=True),
            inputs=[popup_click_info],
            outputs=[popup],
        )

    # 팝업 저장 → 점수 반영 → 오버레이 갱신 → 팝업 닫기
    popup_save.click(
        fn=update_score,
        inputs=[popup_score_edit, popup_click_info, label_list_edit],
        outputs=[label_list_edit],
    ).then(
        fn=refresh_main_images,
        inputs=[orgn_img_list, label_list_edit, selected_indices, overlay_flag, popup_click_info],
        outputs=[image1, image2, image3, image4],
    ).then(
        fn=update_graphs,
        inputs=[label_list_edit, popup_click_info, selected_indices],
        outputs=[graph_scores],
    ).then(
        fn=lambda: gr.update(visible=False),
        inputs=[],
        outputs=[popup],
    )

    popup_cancel.click(
        fn=lambda: gr.update(visible=False),
        inputs=[],
        outputs=[popup],
    )

    # Overlay 토글
    overlay_button.click(
        fn=toggle_overlay,
        inputs=[
            orgn_img_list, label_list_edit, selected_indices,
            popup_click_info, overlay_flag,
        ],
        outputs=[image1, image2, image3, image4, overlay_flag, overlay_button],
    )

    # 팝업 값 변경
    popup_click_info.change(
        fn=update_dropdown_default,
        inputs=[popup_click_info, label_list_edit],
        outputs=[popup_score_edit],
    )

    popup_score_edit.input(
        fn=update_score,
        inputs=[popup_score_edit, popup_click_info, label_list_edit],
        outputs=label_list_edit,
    ).then(
        fn=update_dropdown_default,
        inputs=[popup_click_info, label_list_edit],
        outputs=[popup_score_edit],
    ).then(
        fn=refresh_main_images,
        inputs=[orgn_img_list, label_list_edit, selected_indices, overlay_flag, popup_click_info],
        outputs=[image1, image2, image3, image4],
    ).then(
        fn=update_graphs,
        inputs=[label_list_edit, popup_click_info, selected_indices],
        outputs=[graph_scores],
    )

    # Export — 4분면이 모두 0인 이미지가 임계값 이상이면 확인 팝업을 먼저 띄운다
    export_inputs = [label_list_edit, patient_id_info, user_id_input,
                     file_names_state, source_npz_name, orgn_img_list]

    def maybe_export(*args):
        zero_count = count_all_zero_images(args[0])
        if zero_count >= ZERO_ROW_WARN_THRESHOLD:
            return (
                gr.update(visible=True),
                gr.update(value=f"4분면이 모두 0인 이미지가 **{zero_count}장** 있습니다. "
                                "판독하지 않고 넘긴 것이 아닌지 확인하세요."),
            )
        export_npy(*args)
        return gr.update(visible=False), gr.update()

    def confirmed_export(*args):
        export_npy(*args)
        return gr.update(visible=False)

    export_button.click(
        fn=maybe_export,
        inputs=export_inputs,
        outputs=[export_confirm, export_confirm_msg],
    )

    export_confirm_ok.click(
        fn=confirmed_export,
        inputs=export_inputs,
        outputs=[export_confirm],
    )

    export_confirm_cancel.click(
        fn=lambda: gr.update(visible=False),
        inputs=[],
        outputs=[export_confirm],
    )

    # Reset
    reset_button.click(
        fn=reset_page,
        inputs=[],
        outputs=[
            image_files, upload_section, main_section,
            gallery,
            image1, image2, image3, image4,
            selected_indices,
            orgn_img_list,
            first_image1, first_image2, first_image3, first_image4,
            patient_id_info,
            label_list_edit,
            popup_click_info, popup_score_edit,
            ext_str, ext_cnt, ext_pid,
            user_id_input,
            overlay_flag, file_is_npz_state,
            file_names_state, source_npz_name,
        ],
    )


import webbrowser, threading, socket

def _find_free_port(start=7860, tries=10):
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start

SERVER_PORT = _find_free_port()

def _open_browser():
    webbrowser.open(f"http://127.0.0.1:{SERVER_PORT}")
threading.Timer(2, _open_browser).start()

import gradio.networking as _gn
_gn.url_ok = lambda url: True

demo.launch(server_name="127.0.0.1", server_port=SERVER_PORT, share=False, debug=True, inbrowser=False, show_api=False)

