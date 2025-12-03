import torch
import cv2
import argparse
import numpy as np
from sam2.build_sam import build_sam2_camera_predictor

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ----------- argparse 추가 -----------
parser = argparse.ArgumentParser()
parser.add_argument("--model_version", type=str, default="efficienttam", help="모델 버전 (e.g., sam2, sam2.1)")
parser.add_argument("--skip_frames", type=int, default=0, help="Número de frames a saltar entre inferencias")
args = parser.parse_args()
# ------------------------------------

# Modelo y predictor
model_version = args.model_version
skip_frames = args.skip_frames

sam2_checkpoint = f"./checkpoints/{model_version}/{model_version}_ti_512x512.pt"
model_cfg = f"{model_version}/{model_version}_ti_512x512.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

# Variables globales
point = None
point_selected = False
if_init = False
random_color = False

# Para control de frames saltados
frame_counter = 0
last_mask_logits = None

# Callback del mouse
def collect_point(event, x, y, flags, param):
    global point, point_selected
    if not point_selected and event == cv2.EVENT_LBUTTONDOWN:
        point = [x, y]
        point_selected = True

# Abrir cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", collect_point)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    temp_frame = frame.copy()

    if not point_selected:
        cv2.putText(temp_frame, "Select an object by clicking a point", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Camera", temp_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Una vez seleccionado el punto:
    if not if_init:
        if_init = True
        predictor.load_first_frame(frame)

        ann_frame_idx = 0
        ann_obj_id = (1,)
        labels = np.array([1], dtype=np.int32)
        points = np.array([point], dtype=np.float32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
            frame_idx=ann_frame_idx, obj_id=ann_obj_id,
            points=points, labels=labels
        )
        last_mask_logits = out_mask_logits

    else:
        # Control de saltos de frame
        if frame_counter % (skip_frames + 1) == 0:
            # Actualización real
            out_obj_ids, out_mask_logits = predictor.track(frame)
            last_mask_logits = out_mask_logits
        else:
            # Reutilizamos la última máscara válida
            out_mask_logits = last_mask_logits

        frame_counter += 1

    # Visualización de la máscara
    all_mask = np.zeros_like(frame, dtype=np.uint8)

    if random_color:
        color = tuple(np.random.randint(0, 256, size=3))
        out_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        colored_mask = np.zeros_like(frame, dtype=np.uint8)
        for c in range(3):
            colored_mask[:, :, c] = out_mask[:, :, 0] * color[c]
    else:
        out_mask = (out_mask_logits[0] > 0.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8) * 255
        colored_mask = cv2.cvtColor(out_mask, cv2.COLOR_GRAY2RGB)

    all_mask = cv2.addWeighted(all_mask, 1, colored_mask, 0.5, 0)
    frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
