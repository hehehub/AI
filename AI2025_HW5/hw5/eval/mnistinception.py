import torch.nn as nn
import torchvision.models as models

class MnistInceptionV3(nn.Module):
    '''
    This is a modified version of the InceptionV3 model that outputs 10 classes instead of 1000.
    '''
    def __init__(self, in_channels=3):
        super(MnistInceptionV3, self).__init__()

        self.model = models.inception_v3(pretrained=True)

        # Change the output layer to output 10 classes instead of 1000 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.model(x)