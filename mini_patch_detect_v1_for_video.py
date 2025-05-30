import argparse
import os
import platform
import sys
from pathlib import Path
from overlap_cropping import crop_image_with_overlap
from torchvision.ops import nms
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracker.byte_tracker import STrack
from types import SimpleNamespace
import numpy as np
import torchreid
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import torch
import json

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from collections import defaultdict

def match_features_to_teams_in_memory(crop_features, team_data):
    team_features = team_data['features']
    team_names = team_data['filenames']
    team_features = F.normalize(team_features, dim=1)

    results = []

    for crop in crop_features:
        crop = F.normalize(crop.unsqueeze(0), dim=1)  # [1, 512]
        sims = torch.mm(team_features, crop.T).squeeze(1)  # [M]

        # 分類到 team
        team_scores = {}
        for team, sim in zip(team_names, sims):
            team_key = os.path.basename(team).split('_')[0]
            team_scores.setdefault(team_key, []).append(sim.item())

        # 對每個 team 做平均
        team_avg_scores = {team: sum(scores)/len(scores) for team, scores in team_scores.items()}
        results.append(team_avg_scores)  # 儲存每個人對每隊的分數

    return results

def crop_clothing_region(image, bbox, 
                         top_ratio=0.25, bottom_ratio=0.55, 
                         left_ratio=0.3, right_ratio=0.7):
    """
    Crop only the clothing region (centered shirt area) from a person bounding box.

    Args:
        image: input image (NumPy array)
        bbox: (x1, y1, x2, y2)
        top_ratio: vertical start (0 = top, 1 = bottom)
        bottom_ratio: vertical end
        left_ratio: horizontal start (0 = left, 1 = right)
        right_ratio: horizontal end
        
    Returns:
        cropped_image: central shirt region
    """
    x1, y1, x2, y2 = map(int, bbox)
    w = x2 - x1
    h = y2 - y1

    # Vertical bounds
    new_y1 = y1 + int(h * top_ratio)
    new_y2 = y1 + int(h * bottom_ratio)

    # Horizontal bounds
    new_x1 = x1 + int(w * left_ratio)
    new_x2 = x1 + int(w * right_ratio)

    # Clip to image bounds
    new_x1 = max(new_x1, 0)
    new_x2 = min(new_x2, image.shape[1])
    new_y1 = max(new_y1, 0)
    new_y2 = min(new_y2, image.shape[0])

    cropped = image[new_y1:new_y2, new_x1:new_x2]
    return cropped

def remove_partially_enclosed_boxes_same_class(detections, area_ratio_thresh=0.6, containment_thresh=0.9):
    """
    Remove boxes that are mostly enclosed by a significantly larger box of the same class.

    Args:
        detections: Tensor[N, 6] — [x1, y1, x2, y2, conf, cls]
        area_ratio_thresh: suppress if small_area / large_area < this
        containment_thresh: suppress if (intersection / small_area) > this

    Returns:
        Tensor[N', 6] — filtered detections
    """
    if len(detections) < 2:
        return detections

    keep = []
    suppressed = set()

    for i in range(len(detections)):
        if i in suppressed:
            continue

        box_i = detections[i, :4]
        cls_i = int(detections[i, 5])
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])

        for j in range(len(detections)):
            if i == j or j in suppressed:
                continue

            box_j = detections[j, :4]
            cls_j = int(detections[j, 5])
            area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])

            if cls_i != cls_j:
                continue

            # Determine which is small and which is large
            if area_i < area_j:
                small_idx, large_idx = i, j
                small_box, large_box = box_i, box_j
                small_area, large_area = area_i, area_j
            else:
                small_idx, large_idx = j, i
                small_box, large_box = box_j, box_i
                small_area, large_area = area_j, area_i

            # Calculate intersection
            xA = max(small_box[0], large_box[0])
            yA = max(small_box[1], large_box[1])
            xB = min(small_box[2], large_box[2])
            yB = min(small_box[3], large_box[3])
            inter_w = max(0, xB - xA)
            inter_h = max(0, yB - yA)
            inter_area = inter_w * inter_h

            containment = inter_area / (small_area + 1e-6)
            area_ratio = small_area / (large_area + 1e-6)

            if containment >= containment_thresh and area_ratio <= area_ratio_thresh:
                suppressed.add(small_idx)

        if i not in suppressed:
            keep.append(i)

    return detections[keep]

def simple_global_nms(dets, iou_thres=0.45, max_det=300):
    if dets.size(0) == 0:
        return dets
    max_wh = 4096  # for offsetting by class
    offsets = dets[:, 5] * max_wh
    boxes = dets[:, :4] + offsets[:, None]
    scores = dets[:, 4]
    keep = nms(boxes, scores, iou_thres)
    return dets[keep[:max_det]]

def get_image_patches(image_4k, crop_size=640, overlap=0.2):
    patches = crop_image_with_overlap(image_4k, crop_size, overlap)
    images = [patch for patch, _ in patches]
    offsets = [offset for _, offset in patches]
    return images, offsets

def preprocess_images(images, device, fp16=False):
    batch = []
    for img in images:
        if img is None or img.shape != (640, 640, 3):
            print("⚠️ Invalid image detected:", img.shape if img is not None else None)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float()  # [H, W, C] → [C, H, W]
        img /= 255.0  # normalize to 0–1
        batch.append(img)

    if not batch:
        print("❌ No valid patches to process!")
        return None

    batch = torch.stack(batch).to(device)
    return batch.half() if fp16 else batch

def draw_detections(image, detections, class_names, color=(0, 255, 0)):
    for det in detections:
        if len(det) == 6:
            x1, y1, x2, y2, conf, cls = det
            track_id = None
        else:
            x1, y1, x2, y2, conf, cls, track_id = det

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)
        label = f"{class_names[cls]} {conf:.2f}"
        if track_id is not None:
            label += f" ID:{int(track_id)}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        clothes_feature_path=ROOT / 'reid/reid_test_image/team/team.pt',  # path to clothing features
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    
    # Create tracker args manually
    tracker_args = SimpleNamespace(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        mot20=False,
        fps=30  # assuming you already have video fps
    )

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    screenshot = source.lower().startswith('screen')

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    #load reid model
    re_id_batch_size = 32
    re_id_model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=True
    )
    re_id_model.eval().to(device)

    # Preprocessing transform
    reid_transform = T.Compose([
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30  # default to 30 FPS if not available
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_idx = 0
    output_path = str(Path(save_dir) / ("after_gobalNMS_overlap_remove_annotated_" + Path(source).name))
    output_json_path = str(Path(save_dir) / "team_tracking.json")
    print(f"🔄 Saving video to: {output_path}")
    print(f"🔄 Saving tracking JSON to: {output_json_path}")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Initialize tracker
    ball_tracker = BYTETracker(tracker_args, frame_rate=tracker_args.fps)
    person_tracker = BYTETracker(tracker_args, frame_rate=tracker_args.fps)

    # load the team reference features
    if clothes_feature_path and os.path.exists(clothes_feature_path):
        print(f"🔍 Loading clothing features from: {clothes_feature_path}")
        team_features = torch.load(clothes_feature_path)
    
    # Initialize global dictionary once outside main loop if not already done
    if 'team_json_records' not in globals():
        team_json_records = defaultdict(lambda: {'frame_id': [], 'team_conf': [], 'bbox': []})

    while cap.isOpened():
        ret, high_resolution_image = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % vid_stride != 0:
            continue

        print(f"🔍 Processing frame {frame_idx}")

        images, offsets = get_image_patches(high_resolution_image, crop_size=imgsz[0], overlap=0.2)

        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        # for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            batch = preprocess_images(images, device, fp16=model.fp16)

        # Inference
        with dt[1]:
            pred = model(batch, augment=augment)
            # print("Predictions before NMS:", len(pred)) # 2
            # print(len(pred[0])) # 2
            # print(pred[0][0].shape) # torch.Size([32, 84, 8400])

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) # this take a [[tensor]] and return a list of tensors
            print("Predictions after NMS:", len(pred)) # 32
            print(pred[0].shape) # torch.Size([0, 6])

        # Make a copy of the original 4K image for drawing
        annotated_image = high_resolution_image.copy()
        all_detections = []
        

        # Process predictions
        for i, det in enumerate(pred):  # per image
            x_offset, y_offset = offsets[i]
            if det is not None and len(det):
                # Remap to original image coordinates
                det[:, [0, 2]] += x_offset
                det[:, [1, 3]] += y_offset
                all_detections.append(det)
            
        if all_detections:
            combined = torch.cat(all_detections, dim=0)  # shape [Total_Detections, 6]
        else:
            combined = torch.empty((0, 6), dtype=torch.float32, device=model.device)
        # print the shape of combined
        print("Combined shape:", combined.shape)
        # Apply global NMS
        if combined.shape[0] > 0:
            final = simple_global_nms(combined, iou_thres=iou_thres, max_det=max_det)
            final = remove_partially_enclosed_boxes_same_class(final, area_ratio_thresh=0.6, containment_thresh=0.9)
        else:
            final = []

        # the shape of final is [N, 6] where N is the number of detections, the 6 columns are [x1, y1, x2, y2, conf, cls]
        print("Final detections after global NMS & remove enclosed boxes:", final.shape if isinstance(final, torch.Tensor) else len(final))
        # Filter by class
        person_dets = final[final[:, 5] == 0]  # class 0 for person
        ball_dets = final[final[:, 5] == 32]   # example class id for ball (change if needed)
        person_dets_np = person_dets[:, :5].cpu().numpy() if person_dets.numel() else np.empty((0, 5))
        ball_dets_np = ball_dets[:, :5].cpu().numpy() if ball_dets.numel() else np.empty((0, 5))

        online_persons = person_tracker.update(person_dets_np, [height, width], [height, width])
        online_balls = ball_tracker.update(ball_dets_np, [height, width], [height, width])

        # print("Online persons after tracking:", len(online_persons))
        # print("Online ball after tracking:", len(online_balls))
        
        # Store all crop tensors and track info for matching
        crop_tensors = []
        crop_track_ids = []
        frame_crop_features = []

        # Format detections with track ID
        final_detections = []
        for t in online_persons:
            tlbr = t.tlbr  # (x1, y1, x2, y2)
            track_id = t.track_id
            conf = t.score
            cls = 0  # hardcode as person
            final_detections.append((tlbr[0], tlbr[1], tlbr[2], tlbr[3], conf, cls, track_id))

            # Crop the clothing region
            x1, y1, x2, y2 = map(int, [tlbr[0], tlbr[1], tlbr[2], tlbr[3]])
            crop_img = crop_clothing_region(high_resolution_image, (x1, y1, x2, y2))
            if crop_img.size == 0:
                continue  # skip empty crops
            pil_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            img_tensor = reid_transform(pil_img)
            crop_tensors.append(img_tensor)
            crop_track_ids.append(track_id) # may not be required, cause the index of crop_tensors is first N element of final_detections

        for t in online_balls:
            tlbr = t.tlbr
            track_id = t.track_id
            conf = t.score
            cls = 32  # hardcode as ball
            final_detections.append((tlbr[0], tlbr[1], tlbr[2], tlbr[3], conf, cls, track_id))

        # Batch ReID feature extraction
        if crop_tensors:
            batch_tensor = torch.stack(crop_tensors).to(device)
            with torch.no_grad():
                batch_features = re_id_model(batch_tensor)
            frame_crop_features.extend(batch_features.cpu())  # [N, 512]
        print(f"Extracted {len(frame_crop_features)} ReID features from crops.")

        # do clothing matching
        if clothes_feature_path and os.path.exists(clothes_feature_path):
            print(f"🔍 Matching clothing features with team data from: {clothes_feature_path}")
            team_scores = match_features_to_teams_in_memory(frame_crop_features, team_features)

        # Update tracking JSON records
        # Track current feature index to sync with crop detections
        feature_index = 0

        for det in final_detections:
            x1, y1, x2, y2, conf, cls, track_id = det
            if track_id not in team_json_records:
                team_json_records[track_id] = {
                    'track_id': track_id,
                    'frame_id': [],
                    'team_conf': [],
                    'bbox': []
                }

            team_json_records[track_id]['frame_id'].append(frame_idx)
            team_json_records[track_id]['bbox'].append([int(x1), int(y1), int(x2), int(y2), float(conf)])

            if cls == 0:  # Person
                if feature_index < len(team_scores):
                    # Ensure all scores are float for JSON
                    team_conf = {k: float(v) for k, v in team_scores[feature_index].items()}
                    team_json_records[track_id]['team_conf'].append(team_conf)
                    feature_index += 1
                else:
                    # Append empty if match score is missing
                    team_json_records[track_id]['team_conf'].append({})
            elif cls == 32:
                # Add empty team_conf to maintain same structure
                team_json_records[track_id]['team_conf'].append({})

        # print("team_scores: ",team_scores, )
        draw_detections(annotated_image, final_detections, names)
        out.write(annotated_image)

        if view_img:
            cv2.imshow("YOLO Detection", annotated_image)
            if cv2.waitKey(1) == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Saved video to: {output_path}")

    output_data = list(team_json_records.values())
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"✅ Saved team tracking JSON to {output_json_path}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--clothes-feature-path', type=str, default=ROOT / 'reid/reid_test_image/team/team.pt', help='path to clothing features for ReID')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

# Example usage:
# python3 mini_patch_detect.py --source './data/images/4k_video.mov' --img 640 --device 0 --weights './weight/yolov9-s.pt' --nosave --classes 0 32