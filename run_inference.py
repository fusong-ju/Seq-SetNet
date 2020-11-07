#!/usr/bin/env python
import os
import sys
import pickle
import torch
import numpy as np
from subprocess import Popen, PIPE


def get_a3m_feat(path):
    prog = os.path.join(os.path.dirname(__file__), "bin/serve_feat")
    process = Popen([prog, path, "3", "0"], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    data = str(stdout, encoding="utf-8").strip().split("\n")
    if len(data) > 10000:
        data = data[:10000]
    x = np.array([np.fromstring(_, dtype=int, sep=" ") for _ in data], dtype=np.int32)
    x = x.reshape(x.shape[0], 8, -1)
    return x


def parse_feature(a3m_path):
    feat = get_a3m_feat(a3m_path)
    return torch.tensor(feat).long()


def load_models(model_dir):
    models = []
    for path in os.listdir(model_dir):
        if path.endswith(".pt"):
            models.append(torch.jit.load(os.path.join(model_dir, path)))
    return models


def inference(models, a3m_path, out_path):
    """
    predict `a3m` with `model`.
    """
    feat = parse_feature(a3m_path)
    with torch.no_grad():
        out = [[] for _ in range(6)]
        for model in models:
            for x, y in zip(out, model(feat)):
                x.append(y.cpu().numpy())

        for i in range(6):
            out[i] = np.mean(out[i], axis=0)
        ss3 = "".join(np.array(list("EHC"))[np.argmax(out[0], axis=-1)])
        ss8 = "".join(np.array(list("BEGHISTC"))[np.argmax(out[1], axis=-1)])
        d = {
            "ss3": ss3,
            "ss8": ss8,
            "phi": np.arctan2(out[2], out[3]),
            "psi": np.arctan2(out[4], out[5]),
        }
        pickle.dump(d, open(out_path, "wb"))


def run_inference(a3m_path, out_path):
    models = load_models(os.path.join(os.path.dirname(__file__), "models"))
    inference(models, a3m_path, out_path)


if __name__ == "__main__":
    a3m_path = sys.argv[1]
    out_path = sys.argv[2]
    run_inference(a3m_path, out_path)
