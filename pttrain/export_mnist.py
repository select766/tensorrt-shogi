# MNISTデータを単純なバイナリにエクスポート

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def main():
    ds = datasets.MNIST('data', train=False, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ]))
    images = []
    labels = []
    for i in range(len(ds)):
        image, label = ds[i]  # Tensor,int
        images.append(image.numpy())
        labels.append(label)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    images.tofile("data/mnist_test_images.bin")
    labels.tofile("data/mnist_test_labels.bin")


if __name__ == '__main__':
    main()
