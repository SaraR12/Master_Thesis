import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
#import torch
#import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

# https://github.com/pytorch/pytorch/issues/3678
import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized

from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import torch.backends.cudnn as cudnn
# Mapping
from Mapping.mapper import Mapper
from homographies import getKeypoints

PLANAR_MAP = cv2.imread('Mapping/plane.png') # Planar map

import yaml


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(image_width, image_height,  *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, cls_names, scores,camera, identities=None, offset=(0,0)):
    pts_src, pts_dst = getKeypoints(str(camera))
    mapperObject = Mapper(PLANAR_MAP, pts_src, pts_dst)

    allMappedPoints = []
    mappedImg = cv2.imread('Mapping/plane.png') #PLANAR_MAP

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]


        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        label = '%d %s %d' % (id, cls_names[i], scores[i])
        label += '%'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        #cv2.rectangle(img, (x1, y1),(x2,y2), color, 3)



        cv2.circle(img, (x1, y1), 3, (255, 0, 0), 3)

        #cv2.circle(img, (x2, y1), 3, (0, 0, 255), 3)
        #cv2.circle(img, (x1, y2), 3, (0, 255, 0), 3)
        cv2.circle(img, (x2, y2), 3, (255, 255, 255), 3)

        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        # Mapping to plane
        x1m, x2m, y1m, y2m, color = mapperObject.mapBoundingBoxPoints(x1,x2,y1,y2, color)


        pTL = np.array([[x1,y1]], dtype='float32')
        pTL = np.array([pTL])
        xTL, yTL = mapperObject.getPoint(pTL)

        pTR = np.array([[x2, y1]], dtype='float32')
        pTR = np.array([pTR])
        xTR, yTR = mapperObject.getPoint(pTR)

        pBL = np.array([[x1, y2]], dtype='float32')
        pBL = np.array([pBL])
        xBL, yBL = mapperObject.getPoint(pBL)

        pBR = np.array([[x2, y2]], dtype='float32')
        pBR = np.array([pBR])
        xBR, yBR = mapperObject.getPoint(pBR)

        cv2.line(mappedImg, (xTL, yTL), (xTR, yTR), color, 2) # left line
        cv2.line(mappedImg, (xTR, yTR), (xBR, yBR), color, 2) # right line
        cv2.line(mappedImg, (xBR, yBR), (xBL, yBL), color, 2) # top line
        cv2.line(mappedImg, (xBL, yBL), (xTL, yTL), color, 2) # bottom line
        """cv2.circle(mappedImg, (x1m, y1m), 3, (255, 0, 0), 3)
        cv2.circle(mappedImg, (x1m, y2m), 3, (0, 0, 255), 3)
        cv2.circle(mappedImg, (x2m, y1m), 3, (0, 255, 0), 3)
        cv2.circle(mappedImg, (x2m, y2m), 3, (255, 255, 255), 3)"""


        #allMappedPoints.append(mappedPoint)


    return img, mappedImg, allMappedPoints, mapperObject


def detect(opt, device,camera, queue=None, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Read Class Name Yaml
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    names = data_dict['names']


################################################ DeepSORT #######################################################

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, 
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, 
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)

    # Initialize
    #if os.path.exists(out):
    #    shutil.rmtree(out)  # delete output folder
    #os.makedirs(out)  # make new output folder

    # Load model
    #google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    #model = torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = False
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    frame = 0
    mappedImg = PLANAR_MAP

    for path, img, im0s, vid_cap in dataset:
        frame += 1

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s


            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                #for c in det[:, -1].unique():
                    #n = (det[:, -1] == c).sum()  # detections per class
                    #s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []
                clses = []

                # Write results
                for *xyxy, conf, cls in det:
                    
                    img_h, img_w, _ = im0.shape  # get image shape
                    
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    clses.append([cls.item()])
                    
                    #if save_img or view_img:  # Add bbox to image
                        #label = '%s %.2f' % (names[int(cls)], conf)
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                clses = torch.Tensor(clses)
                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, clses, im0)
                t3 = time_synchronized()
                
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, 4]
                    clses = outputs[:, 5]
                    scores = outputs[:, 6]
                    stays = outputs[:, 7]
                    outputs[:, 7] = np.ones(outputs.shape[0])*frame
                    im0, mappedImg, mappedPoint, mapperObjects = draw_boxes(im0, bbox_xyxy, [names[i] for i in clses], scores, camera, identities)

                    # Send the boundingboxes (and info such as class and id) out from the function
                    if queue is not None:
                        queue.put(outputs, block=True)
                        print('here')
                    else:
                        yield outputs

                else:
                    yield None
                    mappedImg = PLANAR_MAP
            else:
                yield None
                    # Print time (inference + NMS)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))
            #print('FPS=%.2f' % (1/(t3 - t1)))

            # Comment out if you dont want to step through video
            """   """
            #if cv2.waitKey(0) == 33:
             #   continue
            # Stream results
            if True:
                #numpy_horizontal = np.hstack((im0, mappedImg))
                cv2.imshow(p, im0)
                #cv2.imshow('1', mappedImg)

                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)
        #print('Inference Time = %.2f' % (time_synchronized() - t1))
        #print('FPS=%.2f' % (1/(time_synchronized() - t1)))

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

            # test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--data', type=str, default='yolov5/data/data.yaml', help='data yaml path') # Class names
    parser.add_argument('--source', type=str, default='videos/videoWN.mkv', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.80, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.01, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)
    
    # Select GPU
    device = select_device(args.device)
    import torch
    import torch.backends.cudnn as cudnn
    half = device.type != 'cpu'  # half precision only supported on CUDA

    test = []
    with torch.no_grad():
        out = detect(args, device, 'ME')
        for i in out:
            test.append(i)


def run(path, camera, queue = None):
    print(path)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--data', type=str, default='yolov5/data/data.yaml', help='data yaml path')  # Class names
    parser.add_argument('--source', type=str, default=path, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.1, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    # Select GPU
    device = select_device(args.device)
    import torch
    import torch.backends.cudnn as cudnn

    half = device.type != 'cpu'  # half precision only supported on CUDA

    with torch.no_grad():
        if queue is not None:
            out = detect(args, device, camera, queue)
        else:
            out = detect(args, device, camera)
            return out

        """if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, 4]
        else:
            bbox_xyxy = []
            identities = []"""


