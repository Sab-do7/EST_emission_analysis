import os
import numpy as np
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

from model import *
from data import create_dataset

np.random.seed(100)
torch.manual_seed(100)

# load data
batch_size = 1  # Set to 1 to create diagnostic images
testdata = create_dataset(datadir='../../images/images_/test')
all_dl = DataLoader(testdata, batch_size=batch_size, shuffle=True)
progress = tqdm(enumerate(all_dl), total=len(all_dl))

# load model
model.load_state_dict(torch.load(
    'classification.model', map_location=torch.device('cpu')))
model.eval()

# implant hooks for resnet layers
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.relu.register_forward_hook(get_activation('conv1'))
model.layer1.register_forward_hook(get_activation('layer1'))
model.layer2.register_forward_hook(get_activation('layer2'))
model.layer3.register_forward_hook(get_activation('layer3'))
model.layer4.register_forward_hook(get_activation('layer4'))

# Create output directory for saved plots
save_path = './output_images'
os.makedirs(save_path, exist_ok=True)

# run through test data set
true = 0
false = 0
for i, batch in progress:
    x, y = batch['img'].float().to(device), batch['lbl'].float().to(device)

    output = model(x)
    prediction = 1 if output[0] > 0 else 0

    if prediction == 1 and y[0] == 1:
        res = 'true_pos'
        true += 1
    elif prediction == 0 and y[0] == 0:
        res = 'true_neg'
        true += 1
    elif prediction == 0 and y[0] == 1:
        res = 'false_neg'
        false += 1
    elif prediction == 1 and y[0] == 0:
        res = 'false_pos'
        false += 1

    if batch_size == 1:
        # create plot
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(1, 3))

        # rgb plot
        ax1.imshow(0.2+1.5*(np.dstack([x[0][3], x[0][2], x[0][1]])-
                    np.min([x[0][3].numpy(),
                            x[0][2].numpy(),
                            x[0][1].numpy()]))/
                   (np.max([x[0][3].numpy(),
                            x[0][2].numpy(),
                            x[0][1].numpy()])-
                    np.min([x[0][3].numpy(),
                            x[0][2].numpy(),
                            x[0][1].numpy()])),
                   origin='upper')
        ax1.set_title({'true_pos': 'True Positive',
                       'true_neg': 'True Negative',
                       'false_pos': 'False Positive',
                       'false_neg': 'False Negative'}[res],
                      fontsize=8)
        ax1.set_ylabel('RGB', fontsize=8)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # false color plot
        # RGB = (Aerosols; Water Vapor; SWIR1)
        ax2.imshow(0.2+(np.dstack([x[0][0], x[0][9], x[0][10]])-
                    np.min([x[0][0].numpy(),
                            x[0][9].numpy(),
                            x[0][10].numpy()]))/
                   (np.max([x[0][0].numpy(),
                            x[0][9].numpy(),
                            x[0][10].numpy()])-
                    np.min([x[0][0].numpy(),
                            x[0][9].numpy(),
                            x[0][10].numpy()])),
                   origin='upper')
        ax2.set_ylabel('False Color', fontsize=8)
        ax2.set_xticks([])
        ax2.set_yticks([])

        # layer2 activations plot
        map_layer2 = ax3.imshow(activation['layer2'].sum(axis=(0, 1)),
                                vmin=50, vmax=150)
        ax3.set_ylabel('Layer2', fontsize=8)
        ax3.set_xticks([])
        ax3.set_yticks([])

        f.subplots_adjust(0.05, 0.02, 0.95, 0.9, 0.05, 0.05)

        # Generate a filename using batch index `i` to ensure unique file names
        file_name = f"{res}_{i}_eval.png"
        plt.savefig(os.path.join(save_path, file_name), dpi=200, bbox_inches="tight")
        plt.close()
        
        # Confirm saving of each plot
        print(f"Saved plot for batch {i} as {file_name}")

print('Test set accuracy:', true / (true + false))
