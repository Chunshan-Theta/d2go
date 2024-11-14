# Adapted from https://github.com/facebookresearch/detectron2/
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import json

from PIL import Image
import cv2
from detectron2.utils.visualizer import ColorMode
import torch
import tqdm
from d2go.model_zoo import model_zoo
from d2go.utils.demo_predictor import VisualizationDemo
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from detectron2.data.catalog import MetadataCatalog

from d2go.utils.demo_predictor import DemoPredictor

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(cfg, args):
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="keypoint_rcnn_fbnetv3a_dsmask_C4.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--ckp-file",
        default="../output/model_1064999.pth",
        metavar="FILE",
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--webcam", action="store_true", help="Take inputs from webcam."
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class MyVisualizationDemo(VisualizationDemo):
    def __init__(self, cfg, config_file, runner, instance_mode=ColorMode.IMAGE, parallel=False):
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        model = runner.build_model(cfg, eval_only=True)
        self.predictor = DemoPredictor(model)

def main():
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    # cfg = model_zoo.get_config(args.config_file)

    cfg_file = args.config_file
    runner = "d2go.runner.GeneralizedRCNNRunner"
    runner = model_zoo.create_runner(runner)
    cfg = runner.get_default_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHTS = args.ckp_file

    cfg = setup_cfg(cfg, args)
    # import pdb; pdb.set_trace()
    demo = MyVisualizationDemo(cfg, args.config_file, runner)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            # import pdb; pdb.set_trace()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            logger.info("{}".format(predictions))

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert (
                        len(args.input) == 1
                    ), "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"x264"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()

def batch(cfg_file='../newtrain/faster_rcnn_fbnetv3a_dsmask_C4.yaml', ckp_file='../newtrain/output/model_0289999.pth', dn='../../../mydata/cocoquestion/val2017'):
    mp.set_start_method("spawn", force=True)
    setup_logger(name="batch_fvcore")
    logger = setup_logger()

    # cfg = model_zoo.get_config(args.config_file)
    runner = "d2go.runner.GeneralizedRCNNRunner"
    runner = model_zoo.create_runner(runner)
    cfg = runner.get_default_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHTS = ckp_file
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.01
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        0.01
    )
    cfg.freeze()

    # cfg = setup_cfg(cfg, args)
    # import pdb; pdb.set_trace()
    demo = MyVisualizationDemo(cfg, cfg_file, runner)
    inputs = glob.glob(f'{dn}/*.jp*g')
    for path in tqdm.tqdm(inputs):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        # import pdb; pdb.set_trace()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )
        logger.info("{}".format(predictions))
        os.makedirs(f'{dn}d2go', exist_ok=True)
        out_filename = os.path.join(f'{dn}d2go', os.path.basename(path))
        visualized_output.save(out_filename)
        with open(f'{out_filename}.txt', 'w') as fo:
            # fo.write(str(predictions))
            fo.write(json.dumps(predictions, default=prediction2Dict))

def prediction2Dict(predictions):
    # import pdb; pdb.set_trace()
    boxes = predictions.get_fields()['pred_boxes'].tensor.tolist()
    classes = predictions.get_fields()['pred_classes'].tolist()
    scores = predictions.get_fields()['scores'].tolist()
    return {
        'classes': classes,
        'scores': scores,
        'boxes': boxes
    }


def batchCrop(cfg_file='../newtrain/faster_rcnn_fbnetv3a_dsmask_C4.yaml', ckp_file='../newtrain/output/model_0289999.pth', dn='../../../mydata/cocoquestion/val2017'):
    mp.set_start_method("spawn", force=True)
    setup_logger(name="batch_fvcore")
    logger = setup_logger()

    # cfg = model_zoo.get_config(args.config_file)
    runner = "d2go.runner.GeneralizedRCNNRunner"
    runner = model_zoo.create_runner(runner)
    cfg = runner.get_default_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHTS = ckp_file
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.01
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.01
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        0.01
    )
    cfg.freeze()

    # cfg = setup_cfg(cfg, args)
    # import pdb; pdb.set_trace()
    demo = MyVisualizationDemo(cfg, cfg_file, runner)
    outdir = f'{dn}d2go'
    inputs = glob.glob(f'{dn}/*.jp*g')
    for path in tqdm.tqdm(inputs):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        # import pdb; pdb.set_trace()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )
        logger.info("{}".format(predictions))
        os.makedirs(f'{outdir}', exist_ok=True)
        out_filename = os.path.join(f'{outdir}', os.path.basename(path))
        visualized_output.save(out_filename)
        with open(f'{out_filename}.txt', 'w') as fo:
            fo.write(str(predictions))
        # do crop
        # inst = predictions['instances'][0]
        # import pdb; pdb.set_trace()
        boxes = predictions['instances'].get_fields()['pred_boxes'].tensor.tolist()
        classes = predictions['instances'].get_fields()['pred_classes'].tolist()
        for i in range(len(classes)):
            if classes[i] == 0:
                img = Image.open(path)
                cropped = img.crop(boxes[i])
                os.makedirs(f'{outdir}_crop', exist_ok=True)
                out_filename = os.path.join(f'{outdir}_crop', os.path.basename(path))
                cropped.save(out_filename)
                break

if __name__ == "__main__":
    # batch(cfg_file='../newtrain2/faster_rcnn_fbnetv3a_dsmask_C4.yaml', ckp_file='../newtrain2/output/model_0749999.pth', dn='./mathpix/online2') # 0334999 online，model_0644999， 066499, 0749999
    main()
