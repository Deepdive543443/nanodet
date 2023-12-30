import argparse, os, copy
import pnnx
import onnx
import onnxsim
import torch
nn = torch.nn

from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight
from nanodet.model.arch.one_stage_detector import OneStageDetector
from nanodet.model.head.nanodet_plus_head import NanoDetPlusHead
from nanodet.model.backbone import build_backbone
from nanodet.model.fpn import build_fpn

class LottaStageDetector(OneStageDetector):
    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = self.backbone(x)
        if hasattr(self, "fpn"):
            x = self.fpn(x)
        if hasattr(self, "head"):
            x = self.head.forward_detach(x)
            x1 = x[0]
            x2 = x[1]
            x3 = x[2]
            x4 = x[3]

        return (
            torch.sigmoid(x1[:, :80, ...]), x1[:, 80:, ...],
            torch.sigmoid(x2[:, :80, ...]), x2[:, 80:, ...],
            torch.sigmoid(x3[:, :80, ...]), x3[:, 80:, ...],
            torch.sigmoid(x4[:, :80, ...]), x4[:, 80:, ...]
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="model config file path")
    parser.add_argument("--model", help="model file path")
    parser.add_argument("--save_script", default="script", help="path to images or video")
    parser.add_argument("--model_name", help="name and type of model")
    parser.add_argument("--print_log", type=int, default=0, help="name and type of model")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    load_config(cfg, args.config)
    logger = Logger(0, use_tensorboard=False)

    # model = build_model(cfg.model)
    model = LottaStageDetector(
        backbone_cfg=cfg.model.arch.backbone,
        fpn_cfg=cfg.model.arch.fpn,
        head_cfg=cfg.model.arch.head
    )
    ckpt = torch.load(args.model, map_location=lambda storage, loc: storage)
    load_model_weight(model, ckpt, logger)
    model.to("cpu").eval()

    x = torch.randn(1, 3, 416, 416)
    # print(model(x), len(model(x)))
    for output in model(x):
        print(output.shape)

    os.system('mkdir -p ncnn_model')
    os.system('mkdir -p raw_output')
    pnnx.export(
        model = model,
        ptpath = f"raw_output/{args.model_name}.pt",
        inputs = x,
        pnnxparam = f"raw_output/{args.model_name}.pnnx.param",
        pnnxbin = f"raw_output/{args.model_name}.pnnx.bin",
        pnnxpy = f"raw_output/{args.model_name}_pnnx.py",
        pnnxonnx = f"raw_output/{args.model_name}.pnnx.onnx",
        ncnnparam = f"ncnn_model/{args.model_name}.param",
        ncnnbin = f"ncnn_model/{args.model_name}.bin",
        ncnnpy = f"ncnn_model/{args.model_name}_ncnn.py",
        optlevel = 2
    )

    # torch.onnx.export(
    #     model,
    #     x,
    #     f"raw_output/{args.model_name}.onnx",
    #     verbose=True,
    #     keep_initializers_as_inputs=True,
    #     opset_version=11,
    #     # input_names=["data"],
    #     # output_names=["output"],
    # )

    if not args.print_log:
        os.system('rm debug*')