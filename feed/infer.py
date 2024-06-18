import os
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms

def pre_process(image):

    test_transforms = transforms.Compose([transforms.Resize(355), 
                                  transforms.CenterCrop(299),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return test_transforms(image)

def predict(image):

    class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = models.inception_v3(pretrained=True)
    model.aux_logits=False
    num_ftrs = model.fc.in_features

    model.fc = nn.Linear(num_ftrs, len(class_names))

    model.to(device)

    cwd = os.getcwd()
    path_to_model = os.path.join(cwd, 'models/Inception_v3.model')

    model.load_state_dict(torch.load(path_to_model, map_location = torch.device('cpu')))
    model.eval() 

    image = pre_process(image)
    batch = torch.unsqueeze(image, 0)

    output = model(batch.to(device))
    _, pred = torch.max(output, 1)

    label = class_names[pred]

    return label