import os
import json
def calculate_pr(opt):
    test_path = opt.test_pred_save_path

    with open(test_path, 'r') as f:
        test = f.readlines()

    p = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    l = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    c = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    i = 0
    while i < len(test):
        label = json.loads(train[i + 2][:-1])
        pred = json.loads(train[i + 3][:-1])
        for k in range(0, len(label)):
            l[label[k]] += 1
            p[pred[k]] += 1
            if label[k] == pred[k]:
                c[label[k]] += 1
        i += 4

    precision = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    recall = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    f1 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    for key in range(0, 5):
        precision[key] = round(c[key] / p[key], 3)
        recall[key] = round(c[key] / l[key], 3)
        f1[key] = round(2 * precision[key] * recall[key] / (precision[key] + recall[key]), 3)
    print(precision)
    print(recall)
    print(f1)








