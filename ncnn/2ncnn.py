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
        x = x.permute(0, 2, 1)
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

    script = torch.jit.trace(script, x)
    script.save(f"{args.save_script}.pt")
    # print(model(x).shape)
    # script = torch.jit.trace(model, x)
    # script.save(f"{args.save_script}.pt")