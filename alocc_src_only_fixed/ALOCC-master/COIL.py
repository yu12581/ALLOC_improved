import numpy as np
import torch, os, glob
import matplotlib.pyplot as plt
from torchvision import transforms, io
from torch.utils.data import DataLoader, Dataset, TensorDataset
from shutil import copyfile
from model import ALOCC, ALOCC_LOSS
from Metrics import calculate_metrics
from utils import timer, DEVICE

class COIL(Dataset):
    dataset = [
        glob.glob(os.path.join("./dataset/coil-100", f"obj{i}_*.png"))
        for i in range(0, 101)
    ]
    def __init__(self, specific=None, train=True, test_count=72, out_class_scale=1, per_out_class_count=None):
        self.data = None
        dataset = COIL.dataset
        normal_files = np.random.choice(dataset[specific], size=test_count, replace=False)
        if not train: # 测试集：内类从Normal读取前1k个，外类从Abnormal读取前1k个
            # 内类（Normal）- 标签为specific
            normal_files = np.random.choice(normal_files, size=test_count, replace=False)
            abnormal_files = []
            # 外类（Abnormal）- 标签为0
            if per_out_class_count is None:
                for i in range(1, 101):
                    if i != specific:
                        abnormal_files.extend(dataset[i])
                abnormal_files = np.random.choice(abnormal_files, size=int(test_count*out_class_scale), replace=False)
            else:
                for i in range(1, 101):
                    if i != specific:
                        abnormal_files.extend(np.random.choice(dataset[i], per_out_class_count, replace=False))
            self.extend_from_files(abnormal_files, 0)
        self.extend_from_files(normal_files, specific)

    def extend_from_files(self, files, label, noise_std=0.155*2):
        count = len(files)
        imgs = torch.stack([
            io.read_image(f).float() 
            for f in files
        ]).to(DEVICE, non_blocking=True) / 127.5 - 1.0
        labels = torch.full((count,), label, dtype=torch.int32, device=DEVICE)
        noisy_imgs = torch.clamp(imgs + torch.randn_like(imgs, device=DEVICE) * noise_std, -1.0, 1.0)
        data = TensorDataset(imgs, noisy_imgs, labels)
        self.data = TensorDataset(*[
            torch.cat([self.data.tensors[i], data.tensors[i]]) 
            for i in range(len(data.tensors))
        ]) if self.data else data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train(specific, checkpoint_dir, epoch=40, step=4, r_alpha=0.2):
    data_dataset = COIL(specific=specific, test_count=60, train=True)
    data_loader = DataLoader(data_dataset, batch_size=6, shuffle=True)
    model = ALOCC(c_dim=3, gf_dim=16, df_dim=16, in_h=128, out_h=128)
    model._train(data_loader=data_loader, epoch=epoch, step=step, checkpoint_dir=checkpoint_dir, r_alpha=r_alpha)
    test_dataset = COIL(train=False, test_count=72, specific=specific, per_out_class_count=1)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    best_index, best_metrics = 0, [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(step, epoch + step, step):
        model._load_checkpoint(f'./checkpoint/coil-100/{specific}/epoch_{i}.pth')
        metrics = calculate_metrics(model, test_loader, inner_class=specific)
        if metrics[4] > best_metrics[4]:
            best_index, best_metrics = i, metrics
    copyfile(f'./checkpoint/coil-100/{specific}/epoch_{best_index}.pth', f'./checkpoint/coil-100/{specific}/best.pth')
    print(
        f"Obj {specific} -> F1: {best_metrics[0]:.3f}, Acc: {best_metrics[1]:.3f}, EER: {best_metrics[2]:.3f}, "
        f"AUC: {best_metrics[3]:.3f}, SSIM-IC: {best_metrics[4]:.3f}, SSIM-OC: {best_metrics[5]:.3f}, "
        f"VIF-IC: {best_metrics[6]:.3f}, VIF-OC: {best_metrics[7]:.3f}"
    )
    del model, data_loader, data_dataset, test_dataset, test_loader
    
def train_loss(specific, checkpoint_dir, epoch=40, step=4):
    data_dataset = COIL(specific=specific, test_count=60, train=True)
    data_loader = DataLoader(data_dataset, batch_size=6, shuffle=True)
    model = ALOCC_LOSS(c_dim=3, gf_dim=16, df_dim=16, in_h=128, out_h=128)
    model._train(data_loader=data_loader, outclass_loader=outclass_loader, epoch=epoch, step=step, checkpoint_dir=checkpoint_dir)
    test_dataset = COIL(train=False, test_count=72, specific=specific, per_out_class_count=1)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    best_index, best_metrics = 0, [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(step, epoch + step, step):
        model._load_checkpoint(f'./checkpoint/coil-100/{specific}/epoch_{i}.pth')
        metrics = calculate_metrics(model, test_loader, inner_class=specific)
        if metrics[4] > best_metrics[4]:
            best_index, best_metrics = i, metrics
    copyfile(f'./checkpoint/coil-100/{specific}/epoch_{best_index}.pth', f'./checkpoint/coil-100/{specific}/best.pth')
    print(
        f"Obj {specific} -> F1: {best_metrics[0]:.3f}, Acc: {best_metrics[1]:.3f}, EER: {best_metrics[2]:.3f}, "
        f"AUC: {best_metrics[3]:.3f}, SSIM-IC: {best_metrics[4]:.3f}, SSIM-OC: {best_metrics[5]:.3f}, "
        f"VIF-IC: {best_metrics[6]:.3f}, VIF-OC: {best_metrics[7]:.3f}"
    )
    del model, data_loader, data_dataset, test_dataset, test_loader

def test_single_class(specific):
    data_dataset = COIL(train=False, test_count=72, specific=specific, out_class_scale=1)
    data_loader = DataLoader(data_dataset, batch_size=256, shuffle=False)
    model = ALOCC(c_dim=3, gf_dim=16, df_dim=16, in_h=128, out_h=128)
    model._load_checkpoint(f'./checkpoint/coil-100/{specific}/best.pth')
    f1, acc, eer, auc, ssim_ic, ssim_oc, vif_ic, vif_oc = calculate_metrics(model, data_loader=data_loader, inner_class=specific)
    print(
        f"Obj {specific} -> F1: {f1:.3f}, Acc: {acc:.3f}, EER: {eer:.3f}, AUC: {auc:.3f}, "
        f"SSIM-IC: {ssim_ic:.3f}, SSIM-OC: {ssim_oc:.3f}, VIF-IC: {vif_ic:.3f}, VIF-OC: {vif_oc:.3f}"
    )


def test_per_class(specific):
    data_dataset = COIL(train=False, specific=specific, per_out_class_count=1)
    data_loader = DataLoader(data_dataset, batch_size=256, shuffle=False)
    model = ALOCC(c_dim=3, gf_dim=16, df_dim=16, in_h=128, out_h=128)
    model._load_checkpoint(f'./checkpoint/coil-100/{specific}/best.pth')
    f1, acc, eer, auc, ssim_ic, ssim_oc, vif_ic, vif_oc = calculate_metrics(model, data_loader=data_loader, inner_class=specific)
    print(
        f"Obj {specific} -> F1: {f1:.3f}, Acc: {acc:.3f}, EER: {eer:.3f}, AUC: {auc:.3f}, "
        f"SSIM-IC: {ssim_ic:.3f}, SSIM-OC: {ssim_oc:.3f}, VIF-IC: {vif_ic:.3f}, VIF-OC: {vif_oc:.3f}"
    )

def show(specific):
    start_index = specific * 72
    end_index = start_index + 72
    image_list = COIL.dataset[start_index:end_index]
    num_images = 72
    # 计算布局，例如接近正方形的布局
    rows, cols = 8, 9
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    # 如果子图数量是1，则axes不是数组，需要处理这种情况
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    # 展平axes数组以便于迭代
    axes = axes.ravel()
    for i in range(num_images):
        ax = axes[i]
        ax.imshow(image_list[i])
        ax.axis('off') # 不显示坐标轴
    # 隐藏多余的子图
    for i in range(num_images, len(axes)):
        fig.delaxes(axes[i])
    plt.tight_layout() # 自动调整子图参数,使之填充整个图像区域
    plt.show()

if __name__ == '__main__':
    epoch = 40
    step = 5
    specific = 14
    # for i in range(99, 0, -1):
    #     show(i)
    train(specific=specific, checkpoint_dir=f'./checkpoint/coil-100/{specific}', epoch=epoch, step=step)
    # for specific in range(1, 101):
    #     train(specific=specific, checkpoint_dir=f'./checkpoint/coil-100/{specific}', epoch=epoch, step=step)
    # for specific in range(1, 101):
    #     test_per_class(specific)