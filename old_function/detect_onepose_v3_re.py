import argparse
import os
import platform
import sys
from pathlib import Path
import torch
import numpy as np
import onepose

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER, Profile, check_file, check_img_size, check_imshow, colorstr, cv2,
    increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh
)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


def compute_perspective_transform(pts, width, height):
    """Compute the perspective transform matrix from 4 points to (width x height)."""
    src = np.float32(pts)
    dst = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return M


def perspective_transform_points(points, M):
    """
    Transform a list (or array) of 2D points using a 3x3 perspective matrix M.
    points: np.array of shape (N, 2).
    returns: np.array of shape (N, 2) of transformed points.
    """
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    transformed_pts = cv2.perspectiveTransform(pts, M)
    return transformed_pts.reshape(-1, 2)


def prepare_perspective(goal_image_coordinate, goal_realworld_size):
    """
    Compute perspective matrix if 4 corner points provided. 
    Returns (perspective_matrix, have_perspective).
    """
    if goal_image_coordinate and len(goal_image_coordinate) == 4:
        perspective_matrix = compute_perspective_transform(
            goal_image_coordinate,
            goal_realworld_size[0],
            goal_realworld_size[1]
        )
        print("Perspective matrix:", perspective_matrix)
        return perspective_matrix, True
    else:
        return None, False


def setup_output_dir(project, name, exist_ok, save_txt):
    """
    Create the output directory and returns its path.
    """
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # e.g. runs/detect/exp
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    return save_dir


def load_yolo_model(weights, device, dnn, data, half):
    """
    Load the YOLO model (DetectMultiBackend) and return:
    (model, stride, names, pt).
    """
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    return model, stride, names, pt


def load_pose_model():
    """
    Load the OnePose model (or other pose model).
    """
    pose_model = onepose.create_model('ViTPose_huge_simple_coco').to("cuda")
    return pose_model


def create_dataloader(source, imgsz, stride, pt, vid_stride):
    """
    Create a dataset/dataloader for images, video, or webcam.
    Returns (dataset, batch_size, is_webcam_mode).
    """
    source_str = str(source)
    is_file = Path(source_str).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source_str.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source_str.isnumeric() or source_str.endswith('.txt') or (is_url and not is_file)
    screenshot = source_str.lower().startswith('screen')

    if is_url and is_file:
        # For example: download if it's a direct URL
        source_str = check_file(source_str)

    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source_str, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source_str, img_size=imgsz, stride=stride, auto=pt)
        bs = 1
    else:
        dataset = LoadImages(source_str, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = 1

    return dataset, bs, webcam


def warmup_yolo_model(model, pt, bs, imgsz):
    """
    Warm up model with a simple forward pass. 
    """
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))


def infer_on_dataset(
    dataset,
    model,
    pose_model,
    names,
    pt,
    imgsz,
    conf_thres,
    iou_thres,
    max_det,
    classes,
    agnostic_nms,
    augment,
    visualize,
    have_perspective,
    perspective_matrix,
    goal_realworld_size,
    save_img,
    save_txt,
    save_conf,
    save_crop,
    line_thickness,
    hide_labels,
    hide_conf,
    view_img,
    save_dir
):
    """
    Main inference loop over dataset:
     1) Preprocess
     2) YOLO inference
     3) NMS
     4) Process predictions
     5) Visualization & saving results
    Returns (seen, windows, dt) for summary.
    """
    vid_path, vid_writer = [None] * 1, [None] * 1  # updated dynamically if multiple streams
    seen, windows = 0, []
    dt = (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:
        # -- Preprocessing
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # -- YOLO Inference
        with dt[1]:
            visualize_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize_path)

        # -- NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # -- Process predictions
        for i, det in enumerate(pred):
            seen += 1

            # If multi-webcam, get correct frame info
            if hasattr(dataset, 'count') and isinstance(path, list):
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}'
            )

            # -- Warp entire image if perspective is available
            if have_perspective:
                im0_warped = cv2.warpPerspective(
                    im0,
                    perspective_matrix,
                    (goal_realworld_size[0], goal_realworld_size[1])
                )
            else:
                im0_warped = im0

            # -- Rescale boxes (640->original)
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # -- Annotator
            annotator_warped = Annotator(im0_warped, line_width=line_thickness, example=str(names))

            # -- Print results (number of each class)
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
            s += '%gx%g ' % im.shape[2:]

            all_detections_skeletons_for_frame = []  # <--- collect skeletons for this frame here

            # -- Process each detection
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    detection_result = process_single_detection(
                        im0s,
                        xyxy,
                        conf,
                        cls,
                        have_perspective,
                        perspective_matrix,
                        goal_realworld_size,
                        pose_model,
                        save_txt,
                        save_conf,
                        txt_path,
                        save_crop,
                        hide_labels,
                        hide_conf,
                        names,
                    )

                    # detection_result might have bounding box + skeleton info
                    if detection_result is not None:
                        # store skeleton data
                        all_detections_skeletons_for_frame.append(detection_result)
            
            # -- Draw all bbox and skeletons
            draw_all_detections(
                im0_warped,
                all_detections_skeletons_for_frame,
                pose_model,
                line_thickness
            )


            print(im0_warped.shape)

            # Show window
            if view_img:
                handle_view_img(p, windows, im0_warped)

            # Save image / video
            if save_img:
                handle_save_results(
                    dataset,
                    i,
                    im0_warped,
                    save_path,
                    vid_path,
                    vid_writer
                )

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    return seen, windows, dt


def process_single_detection(
    im0s,
    xyxy,
    conf,
    cls,
    have_perspective,
    perspective_matrix,
    goal_realworld_size,
    pose_model,
    save_txt,
    save_conf,
    txt_path,
    save_crop,
    hide_labels,
    hide_conf,
    names
):
    """
    Gather bounding box & skeleton data, but do NOT draw anything here.
    We'll return a dict that can be drawn later.
    """
    x1, y1, x2, y2 = map(int, xyxy)
    c = int(cls)

    # Warp bounding box corners if needed
    if have_perspective:
        corners_src = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        corners_dst = perspective_transform_points(corners_src, perspective_matrix)
        wx1, wy1 = corners_dst[:, 0].min(), corners_dst[:, 1].min()
        wx2, wy2 = corners_dst[:, 0].max(), corners_dst[:, 1].max()
    else:
        wx1, wy1, wx2, wy2 = x1, y1, x2, y2

    # Initialize detection result
    detection_result = {
        'cls': c,
        'label_str': None,  # We'll populate later
        'bbox': [wx1, wy1, wx2, wy2],
        'conf': float(conf),
        'keypoints': None,  # We'll store skeleton coords here if it's a person
        'kp_conf': None     # We'll store skeleton confidences here if it's a person
    }

    # Build label text
    if not hide_labels:
        if hide_conf:
            detection_result['label_str'] = f'{names[c]}'
        else:
            detection_result['label_str'] = f'{names[c]} {conf:.2f}'

    # Pose estimation if it's person class=0
    keypoints_global = []
    if c == 0:  # person
        keypoints_dict, clamped_keypoints = handle_pose_estimation(
            im0s,
            x1, y1, x2, y2,
            have_perspective,
            perspective_matrix,
            pose_model
        )
        # Flatten for .txt saving
        if keypoints_dict and clamped_keypoints is not None:
            for (kx, ky), cpt in zip(clamped_keypoints, keypoints_dict['confidence']):
                keypoints_global.extend([kx, ky, float(cpt)])

            # Store skeleton data for later drawing
            detection_result['keypoints'] = clamped_keypoints
            detection_result['kp_conf'] = keypoints_dict['confidence']

    # Save to .txt if needed
    if save_txt and goal_realworld_size is not None:
        save_yolo_txt(
            c,
            wx1, wy1, wx2, wy2,
            goal_realworld_size,
            conf,
            keypoints_global,
            save_conf,
            txt_path
        )

    # Optionally, if you want to save crop images
    if save_crop:
        # You can store info needed to do cropping in the main draw function
        detection_result['save_crop'] = True
        detection_result['txt_path'] = txt_path
        detection_result['class_name'] = names[c]
    else:
        detection_result['save_crop'] = False

    return detection_result


def draw_all_detections(im0_warped, all_detections, pose_model, line_thickness=1):
    """
    Draw bounding boxes and skeletons for all detections in the frame.
    This function is called once after we've processed all detections.
    """
    annotator = Annotator(im0_warped, line_width=line_thickness)

    for det in all_detections:
        # BBox data
        x1, y1, x2, y2 = det['bbox']
        label_str = det['label_str']
        cls_id = det['cls']
        conf_val = det['conf']

        # Draw bounding box
        annotator.box_label([x1, y1, x2, y2], label_str, color=colors(cls_id, True))

        # Draw skeleton if any
        if det['keypoints'] is not None and det['kp_conf'] is not None:
            skel_info = {
                'points': det['keypoints'],
                'confidence': det['kp_conf']
            }
            onepose.visualize_keypoints(
                annotator.im,
                skel_info,
                pose_model.keypoint_info,
                pose_model.skeleton_info
            )

        # Optionally save crop if needed
        if det['save_crop']:
            save_cropped(
                annotator.im,
                x1, y1, x2, y2,
                det['class_name'],
                det['txt_path']
            )

    # Replace the original image with the annotated one
    final_img = annotator.result()
    im0_warped[:, :, :] = final_img  # copy back if needed so caller sees changes


def handle_pose_estimation(
    im0s,
    x1, y1, x2, y2,
    have_perspective,
    perspective_matrix,
    pose_model
):
    """
    Crops the region for a person and runs pose estimation (OnePose).
    Returns:
      keypoints_dict: the original dictionary from the model (with 'points' and 'confidence')
      clamped_keypoints: final coords (warped if perspective) for visualization.
    """
    x1_safe, y1_safe = max(0, x1), max(0, y1)
    x2_safe = min(im0s.shape[1], x2)
    y2_safe = min(im0s.shape[0], y2)

    # If bounding box too small, skip
    if (x2_safe - x1_safe) < 10 or (y2_safe - y1_safe) < 10:
        return None, []

    cropped_img = im0s[y1_safe:y2_safe, x1_safe:x2_safe]
    keypoints_dict = pose_model(cropped_img)  # must return { 'points': Nx2, 'confidence': Nx1 }
    points = keypoints_dict['points']         # shape: Nx2
    confidences = keypoints_dict['confidence']  # shape: Nx1

    # Shift from local crop to original image coords
    for idx in range(len(points)):
        points[idx][0] += x1_safe
        points[idx][1] += y1_safe

    # If perspective, warp skeleton coords
    if have_perspective and perspective_matrix is not None:
        skel_src = np.array(points, dtype=np.float32)
        skel_dst = perspective_transform_points(skel_src, perspective_matrix)

        # Optionally clamp them to the warped image size
        # but we can just pass them along if you don't need strict clamping
        return keypoints_dict, skel_dst  
    else:
        # Return the original coords
        return keypoints_dict, points


def save_yolo_txt(
    cls_id,
    wx1, wy1, wx2, wy2,
    goal_realworld_size,
    conf,
    keypoints_global,
    save_conf,
    txt_path
):
    """
    Saves bounding box (and optional keypoints) in YOLO format
    to a txt file, in the warped domain.
    """
    bw = wx2 - wx1
    bh = wy2 - wy1
    cx = wx1 + bw / 2
    cy = wy1 + bh / 2

    # Normalize in WARPED domain
    norm_cx = cx / goal_realworld_size[0]
    norm_cy = cy / goal_realworld_size[1]
    norm_w  = bw / goal_realworld_size[0]
    norm_h  = bh / goal_realworld_size[1]

    if save_conf:
        line = (cls_id, norm_cx, norm_cy, norm_w, norm_h, float(conf), *keypoints_global)
    else:
        line = (cls_id, norm_cx, norm_cy, norm_w, norm_h, *keypoints_global)

    with open(f'{txt_path}.txt', 'a') as f:
        f.write(('%g ' * len(line)).rstrip() % line + '\n' + '\n')


def draw_all_skeletons(im0_warped, all_skeletons, pose_model):
    """
    Loop over all skeletons in the current frame and draw them on the image.
    """
    for skel_info in all_skeletons:
        # skel_info is a dict with 'points' and 'confidence'
        onepose.visualize_keypoints(
            im0_warped,
            skel_info,
            pose_model.keypoint_info,
            pose_model.skeleton_info
        )


def handle_pose_estimation(
    im0s,
    x1, y1, x2, y2,
    have_perspective,
    perspective_matrix,
    pose_model
):
    """
    Crops region for a person and runs pose estimation.
    Returns:
      keypoints_dict
      clamped_keypoints (array-like)
    """
    x1_safe, y1_safe = max(0, x1), max(0, y1)
    x2_safe = min(im0s.shape[1], x2)
    y2_safe = min(im0s.shape[0], y2)

    # skip if bounding box too small
    if (x2_safe - x1_safe) < 10 or (y2_safe - y1_safe) < 10:
        return None, None

    cropped_img = im0s[y1_safe:y2_safe, x1_safe:x2_safe]
    keypoints_dict = pose_model(cropped_img)  # returns { 'points': Nx2, 'confidence': Nx1 }
    points = keypoints_dict['points']
    confidences = keypoints_dict['confidence']

    # shift from local crop to original coords
    for idx in range(len(points)):
        points[idx][0] += x1_safe
        points[idx][1] += y1_safe

    # if perspective, warp skeleton coords
    if have_perspective and perspective_matrix is not None:
        skel_src = np.array(points, dtype=np.float32)
        skel_dst = perspective_transform_points(skel_src, perspective_matrix)
        return keypoints_dict, skel_dst
    else:
        return keypoints_dict, points


def save_cropped(
    im_warped,
    wx1, wy1, wx2, wy2,
    class_name,
    txt_path
):
    """
    Save cropped region from warped image under `crops/<class_name>`.
    """
    h_warp, w_warp = im_warped.shape[:2]
    iw1 = max(int(wx1), 0)
    iw2 = min(int(wx2), w_warp)
    ih1 = max(int(wy1), 0)
    ih2 = min(int(wy2), h_warp)

    crop_img = im_warped[ih1:ih2, iw1:iw2]
    outdir = Path(txt_path).parent.parent / 'crops' / class_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Use the txt_path stem as a reference
    parent_stem = Path(txt_path).stem
    crop_path = outdir / f'{parent_stem}.jpg'
    cv2.imwrite(str(crop_path), crop_img)


def handle_view_img(p, windows, im0_warped_final):
    """
    Show results in a window, if user sets `view_img=True`.
    """
    if platform.system() == 'Linux' and p not in windows:
        windows.append(p)
        cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(str(p), im0_warped_final.shape[1], im0_warped_final.shape[0])
    cv2.imshow(str(p), im0_warped_final)
    cv2.waitKey(1)


def handle_save_results(
    dataset,
    i,
    im0_warped_final,
    save_path,
    vid_path,
    vid_writer
):
    """
    Save either a single image or frames of a video/stream.
    """
    print(f"Dataset mode: {dataset.mode}")
    if dataset.mode == 'image':
        # Save a single image
        cv2.imwrite(save_path, im0_warped_final)
    else:
        # Save video/stream
        if i < len(vid_path):
            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()

                fps, w_vid, h_vid = 30, im0_warped_final.shape[1], im0_warped_final.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))
                print(f"Initializing video writer: {save_path}, FPS: {fps}, Size: ({w_vid}, {h_vid})")
                vid_writer[i] = cv2.VideoWriter(
                    save_path,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps, 
                    (w_vid, h_vid)
                )

            if not vid_writer[i].isOpened():
                print(f"Error: Failed to initialize video writer for {save_path}")

            print(f"Writing frame {i} to video: {save_path}")
            vid_writer[i].write(im0_warped_final)


def summarize_and_cleanup(
    dt,
    seen,
    imgsz,
    save_dir,
    save_txt,
    save_img,
    update,
    weights
):
    """
    Print speed summary, final results, and handle any final cleanups/updates.
    """
    t = tuple(x.t / seen * 1E3 for x in dt) if seen else (0, 0, 0)
    LOGGER.info(
        f'Speed: {t[0]:.1f}ms pre-process, {t[1]:.1f}ms inference, '
        f'{t[2]:.1f}ms NMS per image at shape {(1, 3, *imgsz)}'
    )

    # Save summary
    if save_txt or save_img:
        label_files = list(save_dir.glob('labels/*.txt'))
        s = f"\n{len(label_files)} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

    # Update model if needed
    if update and weights and len(weights) > 0:
        strip_optimizer(weights[0])

@smart_inference_mode()
def run(
    weights=ROOT / 'yolo.pt',         # model path or triton URL
    source=ROOT / 'data/images',      # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / 'data/coco.yaml',     # dataset.yaml path
    imgsz=(640, 640),                 # inference size (height, width)
    conf_thres=0.25,                  # confidence threshold
    iou_thres=0.45,                   # NMS IOU threshold
    max_det=1000,                     # maximum detections per image
    device='',                        # cuda device, i.e. 0, 0,1,2,3 or cpu
    view_img=False,                   # show results
    save_txt=False,                   # save results to *.txt
    save_conf=False,                  # save confidences in --save-txt labels
    save_crop=False,                  # save cropped prediction boxes
    nosave=False,                     # do not save images/videos
    classes=None,                     # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,               # class-agnostic NMS
    augment=False,                    # augmented inference
    visualize=False,                  # visualize features
    update=False,                     # update all models
    project=ROOT / 'runs/detect',     # save results to project/name
    name='exp',                       # save results to project/name
    exist_ok=False,                   # existing project/name ok, do not increment
    line_thickness=1,                 # bounding box thickness (pixels)
    hide_labels=False,                # hide labels
    hide_conf=False,                  # hide confidences
    half=False,                       # use FP16 half-precision inference
    dnn=False,                        # use OpenCV DNN for ONNX inference
    vid_stride=1,                     # video frame-rate stride
    goal_image_coordinate=None,       # list of 4 points [[x,y], [x,y], [x,y], [x,y]]
    goal_realworld_size=None,         # output width x height
):
    """
    Main detection + pose estimation pipeline.

    If perspective_matrix is provided (via goal_image_coordinate),
    we will warp the entire image and bounding-box coordinates,
    then display/save the warped image + warped coords.
    Otherwise, fallback to the original image as usual.
    """

    # --- 1) Prepare perspective transform if needed ---
    perspective_matrix, have_perspective = prepare_perspective(
        goal_image_coordinate, goal_realworld_size
    )

    # --- 2) Setup output directory ---
    save_dir = setup_output_dir(project, name, exist_ok, save_txt)
    save_img = not nosave and not str(source).endswith('.txt')

    # --- 3) Load YOLO model ---
    model, stride, names, pt = load_yolo_model(weights, device, dnn, data, half)
    imgsz = check_img_size(imgsz, s=stride)

    # --- 4) Load Pose model ---
    pose_model = load_pose_model()

    # --- 5) Create dataloader ---
    dataset, bs, webcam_mode = create_dataloader(
        source, imgsz, stride, pt, vid_stride
    )

    # --- 6) Warm up YOLO model ---
    warmup_yolo_model(model, pt, bs, imgsz)

    # --- 7) Inference over dataset (main loop) ---
    seen, windows, dt = infer_on_dataset(
        dataset, 
        model,
        pose_model,
        names,
        pt,
        imgsz,
        conf_thres,
        iou_thres,
        max_det,
        classes,
        agnostic_nms,
        augment,
        visualize,
        have_perspective,
        perspective_matrix,
        goal_realworld_size,
        save_img,
        save_txt,
        save_conf,
        save_crop,
        line_thickness,
        hide_labels,
        hide_conf,
        view_img,
        save_dir
    )

    # --- 8) Summaries & Cleanup ---
    summarize_and_cleanup(
        dt,
        seen,
        imgsz,
        save_dir,
        save_txt,
        save_img,
        update,
        weights
    )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolo.pt', help='model path')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--goal_image_coordinate', type=int, default=[[99,201],[1822,221],[1813,793],[86,771]], help='four points for perspective transform')
    parser.add_argument('--goal_realworld_size', type=int, default=[2100, 700], help='output width x height')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

# sample usage
# python3 detect_onepose_v3.py --weights "./weight/yolov9-c-converted.pt" --device 0 --view-img --source "./data/video/8-1.mp4" --name 'test'