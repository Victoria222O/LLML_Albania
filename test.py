import pandas as pd, numpy as np
from detector import Detector

x = pd.read_csv("train-input.csv")
y = pd.read_csv("train-output.csv")

d = Detector()
scores = np.array([d.update(r) for r in x.to_dict("records")])

pred = scores >= 0.5
true = y.is_anomaly.values == 1
tp = (pred & true).sum()
f1 = 2 * tp / (2 * tp + (pred & ~true).sum() + (~pred & true).sum())

rec = {t: pred[(y.anomaly_type == t).values].mean() for t in ["point", "drift", "collective"]}
final = 100 * (0.8 * f1 + 0.05 * rec["point"] + 0.075 * rec["drift"] + 0.075 * rec["collective"])
print("F1:", f1, "recall:", rec, "scor final:", final)

