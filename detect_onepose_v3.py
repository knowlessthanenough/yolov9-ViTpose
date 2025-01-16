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
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        goal_image_coordinate=None,  # list of 4 points [[x,y], [x,y], [x,y], [x,y]]
        goal_realworld_size=None,  # output width x height
):
    """
    If perspective_matrix is provided (via goal_image_coordinate),
    we will warp the entire image and all bounding-box coordinates to that perspective,
    then display/save the warped image + warped coordinates.
    Otherwise, fallback to the original image as usual.
    """

    # 1) Possibly compute perspective matrix
    if goal_image_coordinate and len(goal_image_coordinate) == 4:
        perspective_matrix = compute_perspective_transform(
            goal_image_coordinate, goal_realworld_size[0], goal_realworld_size[1]
        )
        print("Perspective matrix: ", perspective_matrix)
        have_perspective = True
    else:
        perspective_matrix = None
        have_perspective = False

    source = str(source)
    # Decide whether we plan to save images:
    save_img = not nosave and not source.endswith('.txt')  # True if we want to save images/videos

    # 2) Basic checks for input
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download if URL

    # 3) Output directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # e.g. runs/detect/exp
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # 4) Load YOLO model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    # 5) Load Pose model
    pose_model = onepose.create_model('ViTPose_huge_simple_coco').to("cuda")

    # 6) Create dataloader
    bs = 1
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    vid_path, vid_writer = [None] * bs, [None] * bs

    # 7) Model warmup
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # ---------------- MAIN LOOP ----------------
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            # Preprocess input for YOLO
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]

        # --- YOLO Inference ---
        with dt[1]:
            # If --visualize is set, create a subfolder for feature maps
            visualize_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize_path)

        # --- NMS ---
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # --- Process predictions ---
        for i, det in enumerate(pred):  # i in [0..bs-1] if batch_size=bs
            seen += 1

            if webcam:  # if we have multiple webcam streams
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path object
            save_path = str(save_dir / p.name)  # e.g. .../runs/detect/exp/video.mp4
            txt_path = str(save_dir / 'labels' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}'
            )  # e.g. .../runs/detect/exp/labels/video_37

            # 1) If perspective matrix is available, warp the entire image
            if have_perspective:
                im0_warped = cv2.warpPerspective(im0, perspective_matrix, 
                                                 (goal_realworld_size[0], goal_realworld_size[1]))
            else:
                im0_warped = im0  # no transform

            # 2) Rescale detections to original image size
            if len(det):
                # Rescale boxes from img_size(640x640) to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # 3) We'll create an Annotator for the WARPED image
            annotator_warped = Annotator(im0_warped, line_width=line_thickness, example=str(names))

            # 4) Log shapes
            # Print results
            for c in det[:, 5].unique():
                n = (det[:, 5] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
            s += '%gx%g ' % im.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # for normalization in original domain
            imc = im0.copy() if save_crop else im0  

            # 5) Iterate over each detection
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    x1, y1, x2, y2 = map(int, xyxy)

                    # ---- Warp the bounding box corners if needed ----
                    if have_perspective:
                        corners_src = np.array([
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2]
                        ], dtype=np.float32)
                        corners_dst = perspective_transform_points(corners_src, perspective_matrix)
                        # from corners -> min/max
                        wx1, wy1 = corners_dst[:, 0].min(), corners_dst[:, 1].min()
                        wx2, wy2 = corners_dst[:, 0].max(), corners_dst[:, 1].max()
                    else:
                        wx1, wy1, wx2, wy2 = x1, y1, x2, y2

                    # -------------- Pose Estimation for persons --------------
                    keypoints_global = []
                    if c == 0:  # person
                        # safe bounding box in original image
                        x1_safe = max(0, x1)
                        y1_safe = max(0, y1)
                        x2_safe = min(im0s.shape[1], x2)
                        y2_safe = min(im0s.shape[0], y2)

                        if (x2_safe - x1_safe) > 10 and (y2_safe - y1_safe) > 10:
                            cropped_img = im0s[y1_safe:y2_safe, x1_safe:x2_safe]

                            # Run your OnePose model (returns dict with points & confidence)
                            keypoints_dict = pose_model(cropped_img)
                            points = keypoints_dict['points']
                            confidences = keypoints_dict['confidence']

                            # shift from local crop to original coordinates
                            for idx in range(len(points)):
                                points[idx][0] += x1_safe
                                points[idx][1] += y1_safe

                            # ---- warp skeleton if perspective is present ----
                            if have_perspective:
                                skel_src = np.array(points, dtype=np.float32)
                                skel_dst = perspective_transform_points(skel_src, perspective_matrix)
                                # After applying perspective_transform_points to skeleton keypoints
                                h_warp, w_warp = im0_warped.shape[:2]
                                clamped_keypoints = []
                                for kx, ky in skel_dst:
                                    kx_clamped = max(0, min(kx, w_warp - 1))
                                    ky_clamped = max(0, min(ky, h_warp - 1))
                                    clamped_keypoints.append([kx_clamped, ky_clamped])
                            else:
                                skel_dst = np.array(points, dtype=np.float32)

                            # Visualize skeleton on WARPED image if we want
                            # (But note: onepose.visualize_keypoints() draws on the original image,
                            #  so you'd have to adapt that function to draw on `im0_warped` 
                            #  or create a custom function that draws at the warped coords.)
                            # Inside the loop where skeletons are processed
                            if save_img or view_img:
                                if perspective_matrix is not None:
                                    # If perspective transformation is applied, draw on the warped image
                                    onepose.visualize_keypoints(
                                        im0_warped,
                                        {'points': clamped_keypoints, 'confidence': confidences},
                                        pose_model.keypoint_info,
                                        pose_model.skeleton_info
                                    )
                                else:
                                    # Draw on the original image
                                    onepose.visualize_keypoints(
                                        im0,
                                        keypoints_dict,
                                        pose_model.keypoint_info,
                                        pose_model.skeleton_info
                                    )
                            # For now, let's just store the warped skeleton coords:
                            for (kx, ky), conf_pt in zip(skel_dst, confidences):
                                keypoints_global.extend([kx, ky, float(conf_pt)])

                    # -------------- Save to .txt in the WARPED domain --------------
                    # We'll store the bounding box as xywh in the WARPED domain
                    if save_txt:
                        # For YOLO-format, we need W,H in warped domain
                        bw = wx2 - wx1
                        bh = wy2 - wy1
                        cx = wx1 + bw / 2
                        cy = wy1 + bh / 2

                        # Convert to normalized coords in the WARPED image
                        # (Width=goal_realworld_size[0], Height=goal_realworld_size[1])
                        norm_cx = cx / goal_realworld_size[0]
                        norm_cy = cy / goal_realworld_size[1]
                        norm_w  = bw / goal_realworld_size[0]
                        norm_h  = bh / goal_realworld_size[1]

                        if save_conf:
                            line = (c, norm_cx, norm_cy, norm_w, norm_h, float(conf), *keypoints_global)
                        else:
                            line = (c, norm_cx, norm_cy, norm_w, norm_h, *keypoints_global)

                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # -------------- Visualization on im0_warped --------------
                    if save_img or view_img or save_crop:
                        label = None
                        if not hide_labels:
                            label = f'{names[c]}' if hide_conf else f'{names[c]} {conf:.2f}'

                        # Draw bounding box in warped image
                        annotator_warped.box_label([wx1, wy1, wx2, wy2], label, color=colors(c, True))

                        # Optionally save cropped region from the warped image
                        if save_crop:
                            # clamp coords
                            h_warp, w_warp = im0_warped.shape[:2]
                            iw1 = max(int(wx1), 0)
                            iw2 = min(int(wx2), w_warp)
                            ih1 = max(int(wy1), 0)
                            ih2 = min(int(wy2), h_warp)

                            crop_img = im0_warped[ih1:ih2, iw1:iw2]
                            outdir = save_dir / 'crops' / names[c]
                            outdir.mkdir(parents=True, exist_ok=True)
                            crop_path = outdir / f'{p.stem}.jpg'
                            cv2.imwrite(str(crop_path), crop_img)

            # Final warped image with bounding boxes
            im0_warped_final = annotator_warped.result()
            print(im0_warped_final.shape)

            # -- Show results in a window --
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0_warped_final.shape[1], im0_warped_final.shape[0])
                cv2.imshow(str(p), im0_warped_final)
                cv2.waitKey(1)

            # -- Save the final (warped) image or video --
            if save_img:
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
                                vid_writer[i].release()  # release previous video writer

                            # if vid_cap:  # from a video
                                # fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                # w_vid = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                # h_vid = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                # print(f"Video FPS: {fps}, Size: ({w_vid}, {h_vid})")
                            else:
                                # if it's a stream, use the shape of the WARPED image
                                fps, w_vid, h_vid = 30, im0_warped_final.shape[1], im0_warped_final.shape[0]

                            save_path = str(Path(save_path).with_suffix('.mp4'))
                            # print(f"Saving video to: {save_path}")
                            print(f"Initializing video writer: {save_path}, FPS: {fps}, Size: ({w_vid}, {h_vid})")
                            vid_writer[i] = cv2.VideoWriter(
                                save_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps, (w_vid, h_vid)
                            )
                        if not vid_writer[i].isOpened():
                            print(f"Error: Failed to initialize video writer for {save_path}")
                        
                        # Write frame to video
                        print(f"Writing frame {i} to video: {save_path}")
                        vid_writer[i].write(im0_warped_final)

        #  -- Print timing per batch --
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # ---- Summaries ----
    t = tuple(x.t / seen * 1E3 for x in dt)
    LOGGER.info(f'Speed: {t[0]:.1f}ms pre-process, {t[1]:.1f}ms inference, '
                f'{t[2]:.1f}ms NMS per image at shape {(1, 3, *imgsz)}')

    if save_txt or save_img:
        label_files = list(save_dir.glob('labels/*.txt'))
        s = (f"\n{len(label_files)} labels saved to {save_dir / 'labels'}"
             if save_txt else '')
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


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