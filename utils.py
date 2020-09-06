
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

def monte_carlo_dropout(model):
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

    return model

if __name__ == "__main__":
    model = ResNet18()
    model = monte_carlo_dropout(model)
    summary(model, (3, 224, 224))