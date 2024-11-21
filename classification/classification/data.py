
import os
import numpy as np
from matplotlib import pyplot as plt
import rasterio as rio
import torch
from torchvision import transforms

# set random seeds
torch.manual_seed(3)
np.random.seed(3)

class SmokePlumeDataset():

    def __init__(self, datadir=None, mult=1, transform=None,
                 balance='upsample'):

        self.datadir = datadir
        self.transform = transform

        self.imgfiles = []  # list of image files
        self.labels = []    # list of image file labels

        self.positive_indices = []  # list of indices for positive examples
        self.negative_indices = []  # list of indices for negative examples

        # read in image file names
        idx = 0
        for root, dirs, files in os.walk(datadir):
            for filename in files:
                if not filename.endswith('.tif'):
                    # ignore files that or not GeoTIFFs
                    continue
                self.imgfiles.append(os.path.join(root, filename))
                if 'positive' in root:
                    # positive example (smoke plume present)
                    self.labels.append(True)
                    self.positive_indices.append(idx)
                    idx += 1
                elif 'negative' in root:
                    # negative example (no smoke plume present)
                    self.labels.append(False)
                    self.negative_indices.append(idx)
                    idx += 1

        # turn lists into arrays
        self.imgfiles = np.array(self.imgfiles)
        self.labels = np.array(self.labels)
        self.positive_indices = np.array(self.positive_indices)
        self.negative_indices = np.array(self.negative_indices)

        # balance sample, if desired
        if balance == 'downsample':
            self.balance_downsample()
        elif balance == 'upsample':
            self.balance_upsample()

        # increase data set size by factor `mult`
        if mult > 1:
            self.imgfiles = np.array([*self.imgfiles] * mult)
            self.labels = np.array([*self.labels] * mult)
            self.positive_indices = np.array([*self.positive_indices] * mult)
            self.negative_indices = np.array([*self.negative_indices] * mult)


    def __len__(self):
        """Returns length of data set."""
        return len(self.imgfiles)

    def balance_downsample(self):
        """Balance data set (same number of positive/negative samples by
        downsampling the negative class (which is the larger of the two).
        The resulting negative sample will include a random subset of
        the original sample."""
        subsample_idc = np.ravel([
            self.positive_indices,
            self.negative_indices[
                np.random.randint(0, len(self.negative_indices),
                                  len(self.positive_indices))]]).astype(int)

        # adjust other class attributes accordingly
        self.imgfiles = self.imgfiles[subsample_idc]
        self.labels = self.labels[subsample_idc]
        self.positive_indices = np.arange(0, len(self.labels), 1)[
            self.labels == True]
        self.negative_indices = np.arange(0, len(self.labels), 1)[
            self.labels == False]

    def balance_upsample(self):
        """Balance data set (same number of positive/negative samples by
        upsampling the positive class (which is the smaller of the two).
        The resulting positive sample will include a duplicates of a random
        subset of the original sample."""

        subsample_idc = np.ravel([
            self.positive_indices[
                np.random.randint(0, len(self.positive_indices),
                                  len(self.negative_indices)-
                                  len(self.positive_indices))]]).astype(int)

        # adjust other class attributes accordingly
        self.imgfiles = np.concatenate((self.imgfiles,
                                        self.imgfiles[subsample_idc]), axis=0)
        self.labels = np.concatenate((self.labels,
                                      self.labels[subsample_idc]),
                                     axis=0)

        self.positive_indices = np.arange(0, len(self.labels), 1)[
            self.labels == True]
        self.negative_indices = np.arange(0, len(self.labels), 1)[
            self.labels == False]


    def __getitem__(self, idx):
        """Read in image data, preprocess, and apply transformations."""

        # read in data file
        imgfile = rio.open(self.imgfiles[idx])
        imgdata = np.array([imgfile.read(i) for i in
                            [1,2,3,4,5,6,7,8,9,10,12,13]])
        # skip band 11 (Sentinel-2 Band 10, Cirrus) as it does not contain
        # useful information in the case of Level-2A data products

        # force image shape to be 120 x 120 pixels
        if imgdata.shape[1] != 120:
            newimgdata = np.empty((12, 120, imgdata.shape[2]))
            newimgdata[:, :imgdata.shape[1], :] = imgdata[:, :imgdata.shape[1], :]
            newimgdata[:, imgdata.shape[1]:, :] = imgdata[:, imgdata.shape[1]-1:, :]
            imgdata = newimgdata
        if imgdata.shape[2] != 120:
            newimgdata = np.empty((12, 120, 120))
            newimgdata[:, :, :imgdata.shape[2]] = imgdata[:, :, :imgdata.shape[2]]
            newimgdata[:, :, imgdata.shape[2]:] = imgdata[:, :, imgdata.shape[2]-1:]
            imgdata = newimgdata

        sample = {'idx': idx,
                  'img': imgdata,
                  'lbl': self.labels[idx],
                  'imgfile': self.imgfiles[idx]}

        # apply transformation
        if self.transform:
            sample = self.transform(sample)

        return sample

    def display(self, idx, offset=0.2, scaling=1.5):
        """Helper method to display a given example from the data set with
        index `idx`. Only RGB channels are displayed. 
        
        :param idx: (int) image index to be displayed
        :param offset: (float) constant scaling offset (on a range [0,1])
        :param scaling: (float) scaling factor
        :return: `matplotlib.pyplot.figure` object
        """

        imgdata = self[idx]['img'].numpy()

        # scale image data
        imgdata = offset+scaling*(
            np.dstack([imgdata[3], imgdata[2], imgdata[1]])-
            np.min([imgdata[3], imgdata[2], imgdata[1]]))/ \
                (np.max([imgdata[3], imgdata[2], imgdata[1]])-
                 np.min([imgdata[3], imgdata[2], imgdata[1]]))

        f, ax = plt.subplots(1, 1, figsize=(3, 3))
        ax.imshow((imgdata-np.min(imgdata, axis=(0, 1)))/
                   (np.max(imgdata, axis=(0, 1))-
                    np.min(imgdata, axis=(0, 1))))

        return f  

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        """
        :param sample: sample to be converted to Tensor
        :return: converted Tensor sample
        """

        out = {'idx': sample['idx'],
               'img': torch.from_numpy(sample['img'].copy()),
               'lbl': sample['lbl'],
               'imgfile': sample['imgfile']}

        return out

class Normalize(object):
    """Normalize pixel values to zero mean and range [-1, +1] measured in
    standard deviations."""
    def __init__(self):
        self.channel_means = np.array(
            [809.2, 900.5, 1061.4, 1091.7, 1384.5, 1917.8,
             2105.2, 2186.3, 2224.8, 2346.8, 1901.2, 1460.42])
        self.channel_stds = np.array(
            [441.8, 624.7, 640.8, 718.1, 669.1, 767.5,
             843.3, 947.9, 882.4, 813.7, 716.9, 674.8])

    def __call__(self, sample):
        """
        :param sample: sample to be normalized
        :return: normalized sample
        """

        sample['img'] = (sample['img']-self.channel_means.reshape(
            sample['img'].shape[0], 1, 1))/self.channel_stds.reshape(
            sample['img'].shape[0], 1, 1)

        return sample

class Randomize(object):
    """Randomize image orientation including rotations by integer multiples of
       90 deg, (horizontal) mirroring, and (vertical) flipping."""

    def __call__(self, sample):
        """
        :param sample: sample to be randomized
        :return: randomized sample
        """
        imgdata = sample['img']

        # mirror horizontally
        mirror = np.random.randint(0, 2)
        if mirror:
            imgdata = np.flip(imgdata, 2)
        # flip vertically
        flip = np.random.randint(0, 2)
        if flip:
            imgdata = np.flip(imgdata, 1)
        # rotate by [0,1,2,3]*90 deg
        rot = np.random.randint(0, 4)
        imgdata = np.rot90(imgdata, rot, axes=(1,2))

        return {'idx': sample['idx'],
                'img': imgdata.copy(),
                'lbl': sample['lbl'],
                'imgfile': sample['imgfile']}

class RandomCrop(object):
    """Randomly crop 90x90 pixel image (from 120x120)."""

    def __call__(self, sample):
        """
        :param sample: sample to be cropped
        :return: randomized sample
        """
        imgdata = sample['img']

        x, y = np.random.randint(0, 30, 2)

        return {'idx': sample['idx'],
                'img': imgdata.copy()[:, y:y+90, x:x+90],
                'lbl': sample['lbl'],
                'imgfile': sample['imgfile']}

def create_dataset(*args, apply_transforms=True, **kwargs):
    """Create a dataset; uses same input parameters as SmokePlumeDataset.
    :param apply_transforms: if `True`, apply available transformations
    :return: data set"""
    if apply_transforms:
        data_transforms = transforms.Compose([
            Normalize(),
            RandomCrop(),
            Randomize(),
            ToTensor()
           ])
    else:
        data_transforms = None

    data = SmokePlumeDataset(*args, **kwargs, transform=data_transforms)

    return data
