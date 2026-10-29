import base64
import json

import numpy as np
from tqdm import tqdm


def from_base64(s):
    return np.frombuffer(base64.b64decode(s), np.float32)


def load_car_feat(car_feat_file, fps=1, use_plate=False):
    records = []
    with open(car_feat_file) as file:
        for line in tqdm(file):
            r = json.loads(line)
            r["time"] = r["time"] / float(fps)
            r["car_feature"] = from_base64(r["car_feature"])
            if use_plate and r.get("plate_feature") is not None:
                r["plate_feature"] = from_base64(r["plate_feature"])
            else:
                r["plate_feature"] = None
            records.append(r)

    return records
