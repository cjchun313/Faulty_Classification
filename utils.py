from models.resnet import ResNet18
from torchsummary import summary

import numpy as np
from scipy.stats import gumbel_r, weibull_min
from sklearn.metrics import accuracy_score, confusion_matrix


def freeze_model(model, num_layers=6):
    ct = 0
    for child in model.children():
        if ct < num_layers:
            for param in child.parameters():
                param.requires_grad = False
                # print(ct)
        ct += 1

    return model

def monte_carlo_dropout(model):
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    return model

def compute_confidnce_interval(total_pred, y_pred, mean, interval):
    sample_num = len(total_pred)

    for i in range(sample_num):
        if (total_pred[i] > (mean - interval)) and (total_pred[i] < (mean + interval)):
            continue

        y_pred[i] = 5

    return y_pred


def compute_acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)

def compute_euclidean_distance(a, b):
    return np.linalg.norm(a - b, axis=1)

def compute_logit_distance(y_true, y_logit, class_idx):
    return compute_euclidean_distance(y_logit[y_true == class_idx])

def extract_only_one_class(y_true, y_logit, class_idx):
    return y_logit[y_true == class_idx]

def extract_maximum_samples(x, num=20):
    x = sorted(x)

    return np.array(x[-num:])

def compute_loc_scale(x, mode='weibull'):
    if mode == 'weibull':
        #loc, scale = weibull_min.fit_loc_scale(x, 1, 1)
        _, loc, scale = weibull_min.fit(x)
    elif mode == 'gumbel':
        loc, scale = gumbel_r.fit(x)

    return loc, scale

def compute_pdf(x, loc, scale, mode='weibull'):
    if mode == 'weibull':
        pdf = weibull_min.pdf(x, loc=loc, scale=scale, c=1)
    elif mode == 'gumbel':
        pdf = gumbel_r.pdf(x, loc=loc, scale=scale)

    return pdf

def compute_cdf(x, loc, scale, mode='weibull'):
    if mode == 'weibull':
        cdf = weibull_min.cdf(x, loc=loc, scale=scale, c=1)
    elif mode == 'gumbel':
        cdf = gumbel_r.cdf(x, loc=loc, scale=scale)

    return cdf

def update_logit_from_cdf(out, cdf_0, cdf_1, cdf_2, cdf_3, cdf_4):
    v0 = out[:,0] - (out[:,0] * cdf_0)
    v1 = out[:,1] - (out[:,1] * cdf_1)
    v2 = out[:,2] - (out[:,2] * cdf_2)
    v3 = out[:,3] - (out[:,3] * cdf_3)
    v4 = out[:,4] - (out[:,4] * cdf_4)
    v5 = (out[:,0] * cdf_0) + (out[:,1] * cdf_1) + (out[:,2] * cdf_2) + (out[:,3] * cdf_3) + (out[:,4] * cdf_4)

    v = np.array([v0, v1, v2, v3, v4, v5])

    return np.transpose(v, (1,0))




if __name__ == "__main__":
    model = ResNet18()
    model = monte_carlo_dropout(model)
    summary(model, (3, 224, 224))