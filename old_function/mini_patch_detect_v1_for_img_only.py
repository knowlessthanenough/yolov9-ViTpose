import argparse
import os
import platform
import sys
from pathlib import Path
from overlap_cropping import crop_image_with_overlap
from torchvision.ops import nms

import torch

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
    for x1, y1, x2, y2, conf, cls in detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{class_names[cls]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

@smart_inference_mode()
def run(
        weights=ROOT / 'yolo.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
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

    high_resolution_image = cv2.imread(source)
    if high_resolution_image is None:
        raise FileNotFoundError(f"❌ Could not load image: {source}")
    images, offsets = get_image_patches(high_resolution_image, crop_size=imgsz[0], overlap=0.2)

    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # for path, im, im0s, vid_cap, s in dataset:
    with dt[0]:
        batch = preprocess_images(images, device, fp16=model.fp16)

    # Inference
    with dt[1]:
        pred = model(batch, augment=augment)
        print("Predictions before NMS:", len(pred)) # 2
        print(len(pred[0])) # 2
        print(pred[0][0].shape) # torch.Size([32, 84, 8400])

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det) # this take a [[tensor]] and return a list of tensors
        print("Predictions after NMS:", len(pred)) # 32
        print(pred[0].shape) # torch.Size([0, 6])

    # Make a copy of the original 4K image for drawing
    annotated_image = high_resolution_image.copy()
    all_detections = []
    output_path = str(Path(source).with_name("after_gobalNMS_overlap_remove_annotated_" + Path(source).name))

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

    # # crop and save shirt-only images
    # crop_count = 0
    # for det in final:
    #     x1, y1, x2, y2, conf, cls = det
    #     cls = int(cls.item())

    #     if cls != 0:
    #         print(f"⚠️ Skipping non-person class: {cls}")
    #         continue  # skip non-person classes if filtering

    #     bbox = [x1, y1, x2, y2]
    #     crop = crop_clothing_region(high_resolution_image, bbox)

    #     crop_path = f"./crop/player_{crop_count:04d}.jpg"
    #     cv2.imwrite(crop_path, crop)
    #     print(f"✅ Saved shirt crop to: {crop_path}")
    #     crop_count += 1
    #--------------------------------------------------------------------------

    final_detections = [(x1.item(), y1.item(), x2.item(), y2.item(), conf.item(), int(cls.item()))
                    for x1, y1, x2, y2, conf, cls in final]

    draw_detections(annotated_image, final_detections, names)
    cv2.imwrite(output_path, annotated_image)
    print(f"✅ Saved image to: {output_path}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
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
# python3 mini_patch_detect.py --source './data/images/4k_image.jpg' --img 640 --device 0 --weights './weight/yolov9-s.pt' --nosave --classes 0 32