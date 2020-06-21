
from models.resnet import ResNet18
from torchsummary import summary

def freeze_model(model, num_layers=6):
    ct = 0
    for child in model.children():
        if ct < num_layers:
            for param in child.parameters():
                param.requires_grad = False
                # print(ct)
        ct += 1

    return model

if __name__ == "__main__":
    model = ResNet18()
    model = freeze_model(model, num_layers=6)
    summary(model, (3, 224, 224))