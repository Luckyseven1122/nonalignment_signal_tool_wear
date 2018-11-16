from data import FixedIntervalDataset
from pytorch_tool_wear.model import ResNetWithROI
import torch
import torch.cuda
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np

fixedIntervalData = FixedIntervalDataset()

x = fixedIntervalData.get_all_loc_x_sample_data()
y = fixedIntervalData.get_all_loc_y_sample_data()

print("""
---------------------------------------
SAMPLE x shape :%s
SAMPLE y shape :%s
---------------------------------------
"""%(x.shape,y.shape))


PREDICT = False
DROPOUT = 0.5
EPOCHS = 500


# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

for DEPTH in [20,18,15]:
    TRAIN_NAME = "change_resnet_SPP_depth_%s_dropout_%s"%(DEPTH,DROPOUT)

    PATH = "%s.torchmodel"%(TRAIN_NAME)

    if not PREDICT:
        model = ResNetWithROI(20).to(device)
        criterion = MSELoss()
        optimizer = Adam(model.parameters())
        for i in range(EPOCHS):
            for j in range(945):
                input = torch.from_numpy(np.array([x[j]]))
                target = torch.from_numpy(np.array([y[j]]))
                output = model(input)
                optimizer.zero_grad()
                loss = criterion(output,target)
                optimizer.step()


        torch.save(model.state_dict(),PATH)
