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
):
    source = str(source)
    # Decide whether we plan to save images:
    save_img = not nosave and not source.endswith('.txt')  # True if we want to save images/videos

    # Checks
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # If save_txt is True, we create a separate "labels" folder
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Initialize your OnePose model
    pose_model = onepose.create_model('ViTPose_huge_simple_coco').to("cuda")

    # Dataloader
    bs = 1  # default batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)  # for multiple streams, bs = number of streams
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Prepare video writer containers if we’re saving videos
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            # Preprocess
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # If --visualize is set, create a subfolder for feature maps
            visualize_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize_path)

        # NMS
        with dt[2]:
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det
            )

        # Process predictions
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

            # String for logging
            s += '%gx%g ' % im.shape[2:]  # print string of image shape
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain (w, h, w, h)
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Go through each detection
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    # normalized xywh for saving
                    xywh_norm = (
                        xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn
                    ).view(-1).tolist()

                    # If person class => run your pose model
                    keypoints_global = []
                    if c == 0:  # (person)
                        x1, y1, x2, y2 = map(int, xyxy)
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(im0s.shape[1], x2)
                        y2 = min(im0s.shape[0], y2)

                        if (x2 - x1) > 10 and (y2 - y1) > 10:
                            cropped_img = im0s[y1:y2, x1:x2]

                            # Run your OnePose model (returns dict with points & confidence)
                            keypoints_dict = pose_model(cropped_img)
                            points = keypoints_dict['points']
                            confidences = keypoints_dict['confidence']

                            # Adjust points back to the global coordinate system
                            for idx in range(len(points)):
                                points[idx][0] += x1
                                points[idx][1] += y1

                            # If --save-img (or --view-img) => draw keypoints on image
                            if save_img or view_img:
                                onepose.visualize_keypoints(
                                    im0,
                                    keypoints_dict,
                                    pose_model.keypoint_info,
                                    pose_model.skeleton_info
                                )

                            # Flatten (kx, ky, conf) into line for saving
                            for (kx, ky), conf_pt in zip(points, confidences):
                                keypoints_global.extend([kx, ky, float(conf_pt)])

                    # Only save to .txt if --save-txt is set
                    if save_txt:
                        # Create label line
                        if save_conf:
                            line = (cls, *xywh_norm, float(conf), *keypoints_global)
                        else:
                            line = (cls, *xywh_norm, *keypoints_global)

                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # If we want to draw bounding boxes on the final image
                    if save_img or view_img or save_crop:
                        label = None
                        if not hide_labels:
                            if hide_conf:
                                label = f'{names[c]}'
                            else:
                                label = f'{names[c]} {conf:.2f}'

                        # Draw bounding box
                        annotator.box_label(xyxy, label, color=colors(c, True))

                        # Optionally save cropped region
                        if save_crop:
                            save_one_box(xyxy, imc,
                                         file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg',
                                         BGR=True)

            # End of detection loop

            # Stream results if desired
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(0)  # 1 millisecond

            # Save results (image or video)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    # Make sure i is in range
                    if i < len(vid_path):
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer

                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]

                            save_path = str(Path(save_path).with_suffix('.mp4'))
                            vid_writer[i] = cv2.VideoWriter(
                                save_path,
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps, (w, h)
                            )
                        vid_writer[i].write(im0)

        # Print timing info (inference only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}"+
                    f"{dt[1].dt * 1E3:.1f}ms")

    # Final prints
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(
        f'Speed: {t[0]:.1f}ms pre-process, {t[1]:.1f}ms inference, '
        f'{t[2]:.1f}ms NMS per image at shape {(1, 3, *imgsz)}'
    )
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
# python3 detect_onepose_v1.py --weights "./weight/yolov9-c-converted.pt" --device 0 --view-img --source "./data/video/8-1.mp4" --nosave