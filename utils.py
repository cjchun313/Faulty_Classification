from models.resnet import ResNet18
from torchsummary import summary

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


if __name__ == "__main__":
    model = ResNet18()
    model = monte_carlo_dropout(model)
    summary(model, (3, 224, 224))