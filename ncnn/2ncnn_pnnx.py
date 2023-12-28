import argparse, os
import pnnx
import torch
nn = torch.nn

from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

class Script(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward(x)

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

    model = build_model(cfg.model)
    ckpt = torch.load(args.model, map_location=lambda storage, loc: storage)
    load_model_weight(model, ckpt, logger)
    model.to("cpu").eval()
    script = Script(model)

    x = torch.randn(1, 3, 416, 416)
    print(script(x).shape)

    os.system('mkdir -p ncnn_model')
    os.system('mkdir -p raw_output')
    pnnx.export(
        model = script,
        ptpath = f"raw_output/{args.model_name}.pt",
        inputs = x,
        pnnxparam = f"raw_output/{args.model_name}.pnnx.param",
        pnnxbin = f"raw_output/{args.model_name}.pnnx.bin",
        pnnxpy = f"raw_output/{args.model_name}_pnnx.py",
        pnnxonnx = f"raw_output/{args.model_name}.pnnx.onnx",
        ncnnparam = f"ncnn_model/{args.model_name}.param",
        ncnnbin = f"ncnn_model/{args.model_name}.bin",
        ncnnpy = f"ncnn_model/{args.model_name}_ncnn.py",
    )

    if not args.print_log:
        os.system('rm debug*')