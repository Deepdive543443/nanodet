import argparse

import torch
nn = torch.nn

from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

class Script(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.forward(x)
        # x = torch.transpose(x, 1, 2)
        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="model config file path")
    parser.add_argument("--model", help="model file path")
    parser.add_argument("--save_script", default="script", help="path to images or video")
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

    x = torch.randn(1, 3, 320, 320)
    print(script(x).shape)

    script = torch.jit.trace(script, torch.randn(1, 3, 320, 320))
    script.save(f"{args.save_script}.pt")

    torch.onnx.export(script,         # model being run 
    x,       # model input (or a tuple for multiple inputs) 
    f"../ncnn_model/{args.save_script}.onnx",       # where to save the model  
    export_params=True,  # store the trained parameter weights inside the model file 
    opset_version=10,    # the ONNX version to export the model to 
    do_constant_folding=True,  # whether to execute constant folding for optimization 
    input_names = ['modelInput'],   # the model's input names 
    ) 
    print(" ") 
    print('Model has been converted to ONNX')
    # print(model(x).shape)
    # script = torch.jit.trace(model, x)
    # script.save(f"{args.save_script}.pt")