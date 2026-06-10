import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets
from model import ALOCC, DEVICE
from Metrics import calculate_metrics
from utils import timer
from shutil import copyfile


class CIFAR(Dataset):
    data_root = 'dataset/cifar10'
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True)

    def __init__(
        self,
        train=True,
        test_count=540,
        specific=None,
        out_class_scale=1,
        per_out_class_count=None,
        noise_std=0.155 * 2,
        verbose=False,
    ):
        dataset = CIFAR.train_dataset if train else CIFAR.test_dataset
        targets = torch.tensor(dataset.targets)
        if specific is None:
            raise ValueError("specific 类别不能为空")

        if train:
            indices = torch.arange(len(dataset))[targets == specific]
        else:
            specific_mask = (targets == specific)
            specific_indices = torch.where(specific_mask)[0]
            rand_perm = torch.randperm(len(specific_indices))
            indices = specific_indices[rand_perm[:test_count]]
            if per_out_class_count is None:
                out_class_count = int(test_count * out_class_scale)
                non_specific_indices = torch.where(~specific_mask)[0]
                non_perm = torch.randperm(len(non_specific_indices))
                indices = torch.cat([indices, non_specific_indices[non_perm[:out_class_count]]])
            else:
                for i in range(10):
                    if i != specific:
                        class_indices = torch.where(targets == i)[0]
                        class_perm = torch.randperm(len(class_indices))
                        indices = torch.cat([indices, class_indices[class_perm[:per_out_class_count]]])

        indices_np = indices.cpu().numpy()
        imgs_np = dataset.data[indices_np]  # (N, H, W, C), uint8
        imgs = torch.tensor(imgs_np, dtype=torch.float32, device=DEVICE).permute(0, 3, 1, 2)
        imgs = imgs / 127.5 - 1.0
        noisy_imgs = torch.clamp(imgs + torch.randn_like(imgs, device=DEVICE) * noise_std, -1.0, 1.0)
        labels = targets[indices].to(DEVICE, dtype=torch.int32, non_blocking=True)
        self.data = TensorDataset(imgs, noisy_imgs, labels)
        if verbose:
            print(f"Loaded {len(self.data)} CIFAR images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def extend(self, other: TensorDataset):
        self.data = TensorDataset(*[
            torch.cat([self.data.tensors[i], other.tensors[i]])
            for i in range(len(self.data.tensors))
        ])


@timer
def train(specific_class, checkpoint_dir, epoch=40, step=5, r_alpha=0.2):
    data_dataset = CIFAR(specific=specific_class)
    data_loader = DataLoader(data_dataset, batch_size=128, shuffle=True)
    model = ALOCC(c_dim=3, gf_dim=16, df_dim=16, in_h=32, out_h=32)
    model._train(data_loader=data_loader, epoch=epoch, step=step, checkpoint_dir=checkpoint_dir, r_alpha=r_alpha)
    test_dataset = CIFAR(train=False, test_count=800, specific=specific_class, per_out_class_count=200)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    best_index, best_metrics = 0, [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(step, epoch + step, step):
        model._load_checkpoint(f'./checkpoint/cifar10/{specific_class}/epoch_{i}.pth')
        metrics = calculate_metrics(model, test_loader, inner_class=specific_class)
        if metrics[1] > best_metrics[1]:
            best_index, best_metrics = i, metrics
    copyfile(f'./checkpoint/cifar10/{specific_class}/epoch_{best_index}.pth', f'./checkpoint/cifar10/{specific_class}/best.pth')
    print(
        f"Class {specific_class} -> F1: {best_metrics[0]:.3f}, Acc: {best_metrics[1]:.3f}, EER: {best_metrics[2]:.3f}, "
        f"AUC: {best_metrics[3]:.3f}, SSIM-IC: {best_metrics[4]:.3f}, SSIM-OC: {best_metrics[5]:.3f}, "
        f"VIF-IC: {best_metrics[6]:.3f}, VIF-OC: {best_metrics[7]:.3f}"
    )
    del model, data_loader, data_dataset, test_dataset, test_loader


def test_single_class(specific_class):
    data_dataset = CIFAR(train=False, test_count=400, specific=specific_class, out_class_scale=1)
    data_loader = DataLoader(data_dataset, batch_size=256, shuffle=False)
    model = ALOCC(c_dim=3, gf_dim=16, df_dim=16, in_h=32, out_h=32)
    model._load_checkpoint(f'./checkpoint/cifar10/{specific_class}/best.pth')
    f1, acc, eer, auc, ssim_ic, ssim_oc, vif_ic, vif_oc = calculate_metrics(model, data_loader=data_loader, inner_class=specific_class)
    print(
        f"Class {specific_class} -> F1: {f1:.3f}, Acc: {acc:.3f}, EER: {eer:.3f}, AUC: {auc:.3f}, "
        f"SSIM-IC: {ssim_ic:.3f}, SSIM-OC: {ssim_oc:.3f}, VIF-IC: {vif_ic:.3f}, VIF-OC: {vif_oc:.3f}"
    )


def test_per_class(specific_class):
    data_dataset = CIFAR(train=False, test_count=800, specific=specific_class, per_out_class_count=200)
    data_loader = DataLoader(data_dataset, batch_size=256, shuffle=False)
    model = ALOCC(c_dim=3, gf_dim=16, df_dim=16, in_h=32, out_h=32)
    model._load_checkpoint(f'./checkpoint/cifar10/{specific_class}/best.pth')
    f1, acc, eer, auc, ssim_ic, ssim_oc, vif_ic, vif_oc = calculate_metrics(model, data_loader=data_loader, inner_class=specific_class)
    print(
        f"Class {specific_class} -> F1: {f1:.3f}, Acc: {acc:.3f}, EER: {eer:.3f}, AUC: {auc:.3f}, "
        f"SSIM-IC: {ssim_ic:.3f}, SSIM-OC: {ssim_oc:.3f}, VIF-IC: {vif_ic:.3f}, VIF-OC: {vif_oc:.3f}"
    )


if __name__ == '__main__':
    epoch = 40
    step = 1
    specific_class = 0
    for specific_class in range(10):
        train(specific_class=specific_class, checkpoint_dir=f'./checkpoint/cifar10/{specific_class}', epoch=epoch, step=step)
    for specific_class in range(10):
        test_per_class(specific_class)