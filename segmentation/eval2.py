import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, jaccard_score
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from model_unet import *
from data import create_dataset
np.random.seed(3)
torch.manual_seed(3)

# Create a directory to save evaluation images
output_dir = 'evaluation_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load data
valdata = create_dataset(datadir='../../images/images_/test',
                         seglabeldir='../../segmentation_labels')
batch_size = 1  # 1 to create diagnostic images, any value otherwise
all_dl = DataLoader(valdata, batch_size=batch_size, shuffle=True)
progress = tqdm(enumerate(all_dl), total=len(all_dl))

# Load model
model.load_state_dict(torch.load(
    'segmentation.model', map_location=torch.device('cpu')))
model.eval()

# Define loss function
loss_fn = nn.BCEWithLogitsLoss()

# Run through test data
all_ious = []
all_accs = []
all_arearatios = []

for i, batch in progress:
    x, y = batch['img'].float().to(device), batch['fpt'].float().to(device)
    idx = batch['idx']
    output = model(x)

    # Obtain binary prediction map
    pred = np.zeros(output.shape)
    pred[output >= 0] = 1

    # Calculate IoU, with handling for zero division
    cropped_iou = []
    for j in range(y.shape[0]):
        y_flat = y[j].flatten().detach().numpy()
        pred_flat = pred[j][0].flatten()
        if np.sum(pred_flat) != 0 and np.sum(y_flat) != 0:
            z = jaccard_score(y_flat, pred_flat)
            cropped_iou.append(z)
    all_ious.extend(cropped_iou)

    # Calculate image-wise accuracy for this batch
    y_bin = np.array(np.sum(y.detach().numpy(), axis=(1, 2)) != 0).astype(int)
    prediction = np.array(np.sum(pred, axis=(1, 2, 3)) != 0).astype(int)
    all_accs.append(accuracy_score(y_bin, prediction))

    # Calculate smoke areas and area ratios
    output_binary = np.zeros(output.shape)
    output_binary[output.cpu().detach().numpy() >= 0] = 1
    area_pred = np.sum(output_binary, axis=(1, 2, 3))
    area_true = np.sum(y.cpu().detach().numpy(), axis=(1, 2))
    arearatios = [(area_pred[k] / area_true[k]) if area_true[k] != 0 else 1
                  for k in range(len(area_pred))]
    all_arearatios.extend(arearatios)

    if batch_size == 1:
        # Determine result type (TP, TN, FP, FN) for labeling
        res = ('true_pos' if prediction == 1 and y_bin == 1 else
               'true_neg' if prediction == 0 and y_bin == 0 else
               'false_neg' if prediction == 0 and y_bin == 1 else
               'false_pos')

        # Create diagnostic plot
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(1, 3))

        # RGB plot
        ax1.imshow(0.2 + 1.5 * (np.dstack([x[0][3], x[0][2], x[0][1]]) -
                                np.min([x[0][3].numpy(),
                                        x[0][2].numpy(),
                                        x[0][1].numpy()])) /
                   (np.max([x[0][3].numpy(),
                            x[0][2].numpy(),
                            x[0][1].numpy()]) -
                    np.min([x[0][3].numpy(),
                            x[0][2].numpy(),
                            x[0][1].numpy()])),
                   origin='upper')
        ax1.set_title({'true_pos': 'True Positive',
                       'true_neg': 'True Negative',
                       'false_pos': 'False Positive',
                       'false_neg': 'False Negative'}[res],
                      fontsize=8)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # False color plot
        ax2.imshow(0.2 + (np.dstack([x[0][0], x[0][9], x[0][10]]) -
                          np.min([x[0][0].numpy(),
                                  x[0][9].numpy(),
                                  x[0][10].numpy()])) /
                   (np.max([x[0][0].numpy(),
                            x[0][9].numpy(),
                            x[0][10].numpy()]) -
                    np.min([x[0][0].numpy(),
                            x[0][9].numpy(),
                            x[0][10].numpy()])),
                   origin='upper')
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Segmentation ground truth and prediction
        ax3.imshow(y[0], cmap='Reds', alpha=0.3)
        ax3.imshow(pred[0][0], cmap='Greens', alpha=0.3)
        ax3.set_xticks([])
        ax3.set_yticks([])

        # Calculate IoU for the image and add annotation
        this_iou = jaccard_score(y[0].flatten().detach().numpy(),
                                 pred[0][0].flatten()) if np.sum(y[0].detach().numpy()) != 0 else 0
        ax3.annotate(f"IoU={this_iou:.2f}", xy=(5, 15), fontsize=8)
        f.subplots_adjust(0.05, 0.02, 0.95, 0.9, 0.05, 0.05)

        # Save the plot to the designated folder
        img_name = f"{res}_{os.path.basename(batch['imgfile'][0]).replace('.tif', '_eval.png').replace(':', '_')}"
        f.tight_layout()
        plt.savefig(os.path.join(output_dir, img_name), dpi=200)

        plt.close()

# Report average metrics
print('iou:', len(all_ious), np.average(all_ious))
print('accuracy:', len(all_accs), np.average(all_accs))
print('mean area ratio:', len(all_arearatios), np.average(all_arearatios),
      np.std(all_arearatios) / np.sqrt(len(all_arearatios) - 1))
