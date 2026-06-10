import numpy as np
import os
import torch
from model import ALOCC, ALOCC_LOSS
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, TensorDataset
from Metrics import calculate_metrics, show_class_metrics
from utils import timer, DEVICE, set_random_seed
from shutil import copyfile

class MNIST(Dataset):
    data_root = 'dataset/mnist'
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True)

    def __init__(self, train=True, count=10000, specific=None, out_class_scale=1, per_out_class_count=None, noise_std=0.155*2, verbose=False):
        set_random_seed()
        torch.backends.cudnn.deterministic = True
        if train:
            dataset = MNIST.train_dataset
            if per_out_class_count is None:
                class_indices = torch.where(dataset.targets == specific)[0]
                take = min(int(count), int(class_indices.numel()))
                indices = class_indices[torch.randperm(int(class_indices.numel()))[:take]]
            else:
                indices = []
                for i in range(10):
                    if i != specific:
                        class_indices = torch.where(dataset.targets == i)[0]
                        class_selected = class_indices[torch.randperm(len(class_indices))[:per_out_class_count]]
                        indices.append(class_selected)
                indices = torch.cat(indices)
        else:
            dataset = MNIST.test_dataset
            labels = dataset.targets
            specific_mask = (labels == specific)
            specific_indices = torch.where(specific_mask)[0]
            indices = specific_indices[torch.randperm(len(specific_indices))[:count]]
            if per_out_class_count is None:
                out_class_count = int(count * out_class_scale)
                non_specific_indices = torch.where(~specific_mask)[0]
                indices = torch.cat([
                    indices,
                    non_specific_indices[torch.randperm(len(non_specific_indices))[:out_class_count]]
                ])
            else:
                for i in range(10):
                    if i != specific:
                        class_indices = torch.where(labels == i)[0]
                        class_selected = class_indices[torch.randperm(len(class_indices))[:per_out_class_count]]
                        indices = torch.cat([indices, class_selected])
        # 向量化处理
        imgs = dataset.data[indices].float().to(DEVICE, non_blocking=True)
        labels = dataset.targets[indices].to(DEVICE, dtype=torch.int32, non_blocking=True)
        imgs = (imgs / 127.5 - 1.0).unsqueeze(1)  # 添加通道维度
        noisy_imgs = torch.clamp(imgs + torch.randn_like(imgs, device=DEVICE) * noise_std, -1.0, 1.0)
        self.data = TensorDataset(imgs, noisy_imgs, labels)
        if verbose:
            print(f"Loaded {len(self.data)} MNIST images")
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def extend(self, other: TensorDataset):
        self.data = TensorDataset(*[
            torch.cat([self.data.tensors[i], other.tensors[i]]) 
            for i in range(len(self.data.tensors))
        ])
        
    def extend_from_indices(self, imgs, label, noise_std=0.155*2):
        set_random_seed()
        imgs = imgs.float().to(DEVICE, non_blocking=True)
        labels = torch.full((len(imgs),), label, device=DEVICE, dtype=torch.int32)
        imgs = (imgs / 127.5 - 1.0).unsqueeze(1)  # 添加通道维度
        noisy_imgs = torch.clamp(imgs + torch.randn_like(imgs, device=DEVICE) * noise_std, -1.0, 1.0)
        data = TensorDataset(imgs, noisy_imgs, labels)
        self.extend(data)
#测试
def test(model, specific, checkpoint_dir='checkpoint/mnist_loss', epoch=20, step=2, dataloader=None, find_best=False, verbose=False):
    best_index, best_metrics = 0, [0,0,0,0,0,0,0,0,0,0]
    for i in range(step, epoch + step, step):
        model._load_checkpoint(os.path.join(checkpoint_dir, f'{i}.pth'))
        metrics = calculate_metrics(model, dataloader, inner_class=specific, verbose=verbose)
        if metrics[1] > best_metrics[1]:
            best_index, best_metrics = i, metrics
    if find_best:
        copyfile(os.path.join(checkpoint_dir, f'{best_index}.pth'), os.path.join(checkpoint_dir, f'best.pth'))
        print(
            f"Epoch {best_index} (Best) -> F1: {best_metrics[0]:.3f}, Acc: {best_metrics[1]:.3f}, EER: {best_metrics[2]:.3f}, AUC: {best_metrics[3]:.3f}, "
            f"SSIM-IC: {best_metrics[4]:.3f}, SSIM-OC: {best_metrics[5]:.3f}, "
            f"VIF-IC: {best_metrics[6]:.3f}, VIF-OC: {best_metrics[7]:.3f}, "
            f"GMSD-IC: {best_metrics[8]:.3f}, GMSD-OC: {best_metrics[9]:.3f}"
        )        

# 训练
def train(specific, checkpoint_dir, epoch=40, step=5, find_best=False):
    # print(f"数字 {specific}: 开始训练ALOCC")
    batch_size = 128
    train_count = (MNIST.train_dataset.targets == specific).sum().item() // batch_size * batch_size
    data_dataset = MNIST(specific=specific, count=train_count)
    data_loader = DataLoader(data_dataset, batch_size=batch_size, shuffle=True)
    model = ALOCC(in_h=28, out_h=28)
    model._train(data_loader=data_loader, epoch=epoch, step=step, checkpoint_dir=checkpoint_dir)
    # 选取最好的Epoch
    print(f"ALOCC: ", end='')
    test_dataset = MNIST(train=False, specific=specific, count=1600, out_class_scale=0.5)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    test(model, specific=specific, checkpoint_dir=checkpoint_dir, epoch=epoch, step=step, dataloader=test_loader, find_best=find_best)
    # 清空缓存
    del model, data_loader, data_dataset, test_dataset, test_loader
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
def train_loss(specific, checkpoint_dir, epoch=40, step=5, oc_cnt=600, find_best=False):
    # print(f"数字 {specific}: 开始训练ALOCC-LOSS模型 -> OC = {oc_cnt}")
    # 训练
    batch_size = 128
    train_count = (MNIST.train_dataset.targets == specific).sum().item() // batch_size * batch_size
    data_dataset = MNIST(train=True, specific=specific, count=train_count)
    data_loader = DataLoader(data_dataset, batch_size=batch_size, shuffle=True)
    model = ALOCC_LOSS(in_h=28, out_h=28)
    # print(outclass_files)
    oc_cnt = max(batch_size, (oc_cnt // (9*batch_size)) * batch_size)
    outclass_dataset = MNIST(train=True, count=0, specific=specific, per_out_class_count=oc_cnt)
    outclass_loader = DataLoader(outclass_dataset, batch_size=batch_size, shuffle=True)
    model._train(data_loader=data_loader, outclass_loader=outclass_loader, epoch=epoch, step=step, checkpoint_dir=checkpoint_dir)
    # 测试
    print(f"ALOCC_LOSS {oc_cnt // 512 * 10}: ", end='')
    test_dataset = MNIST(train=False, specific=specific, count=1600, out_class_scale=0.5)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    test(model, specific=specific, checkpoint_dir=checkpoint_dir, epoch=epoch, step=step, dataloader=test_loader, find_best=find_best)
    # 清空缓存
    del model, data_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def test_single_class(specific_number):
    data_dataset = MNIST(train=False, count=400, specific=specific_number, out_class_scale=1)
    data_loader = DataLoader(data_dataset, batch_size=256, shuffle=False)
    model = ALOCC(c_dim=1, gf_dim=16, df_dim=16, in_h=28, out_h=28)
    model._load_checkpoint(f'./checkpoint/mnist/{specific_number}/best.pth')
    f1, acc, eer, auc, ssim_ic, ssim_oc, vif_ic, vif_oc = calculate_metrics(model, data_loader=data_loader, inner_class=specific_number)
    print(
        f"Num {specific_number} -> F1: {f1:.3f}, Acc: {acc:.3f}, EER: {eer:.3f}, AUC: {auc:.3f}, "
        f"SSIM-IC: {ssim_ic:.3f}, SSIM-OC: {ssim_oc:.3f}, "
        f"VIF-IC: {vif_ic:.3f}, VIF-OC: {vif_oc:.3f}"
    )

def test_per_class(specific_number):
    data_dataset = MNIST(train=False, count=800, specific=specific_number, per_out_class_count=200)
    data_loader = DataLoader(data_dataset, batch_size=256, shuffle=False)
    model = ALOCC(c_dim=1, gf_dim=16, df_dim=16, in_h=28, out_h=28)
    model._load_checkpoint(f'./checkpoint/mnist/{specific_number}/best.pth')
    f1, acc, eer, auc, ssim_ic, ssim_oc, vif_ic, vif_oc = calculate_metrics(model, data_loader=data_loader, inner_class=specific_number)
    print(
        f"Num {specific_number} -> F1: {f1:.3f}, Acc: {acc:.3f}, EER: {eer:.3f}, AUC: {auc:.3f}, "
        f"SSIM-IC: {ssim_ic:.3f}, SSIM-OC: {ssim_oc:.3f}, "
        f"VIF-IC: {vif_ic:.3f}, VIF-OC: {vif_oc:.3f}"
    )
    
if __name__ == '__main__':
    epoch = 40
    step = 1
    specific = 2
    # 训练    
    # train(specific=specific, checkpoint_dir=f'./checkpoint/mnist/{specific}', epoch=epoch, step=step, find_best=True)
    # train_loss(specific=specific, checkpoint_dir=f'./checkpoint/mnist_loss/{specific}', epoch=epoch, step=step, oc_cnt=5400, find_best=True)
    
    # specific = 5
    # train(specific=specific, checkpoint_dir=f'./checkpoint/mnist/{specific}', epoch=epoch, step=step, find_best=True)
    # train_loss(specific=specific, checkpoint_dir=f'./checkpoint/mnist_loss/{specific}', epoch=epoch, step=step, oc_cnt=5400, find_best=True)
    
    for specific in range(10):
        print(f'数字 {specific} 开始训练...')
        train(specific=specific, checkpoint_dir=f'./checkpoint/mnist/{specific}', epoch=epoch, step=step, find_best=True)
        for oc_cnt in range(5120, 51200//2 + 1, 5120):
            train_loss(specific=specific, checkpoint_dir=f'./checkpoint/mnist_loss/{specific}', epoch=epoch, step=step, oc_cnt=oc_cnt, find_best=True)
        
    # for specific in range(10):
    #     print(f'Number {specific}')
    #     model = ALOCC(in_h=28, out_h=28)
    #     test_dataset = MNIST(train=False, specific=specific, count=1600, out_class_scale=0.5)
    #     test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    #     test(model, specific=specific, checkpoint_dir=f'./checkpoint/mnist/{specific}', epoch=epoch, step=step, dataloader=test_loader, find_best=True, verbose=False)
    #     test(model, specific=specific, checkpoint_dir=f'./checkpoint/mnist_loss/{specific}', epoch=epoch, step=step, dataloader=test_loader, find_best=True, verbose=False)
    # 测试
    # for specific in range(10):
    #     test_per_class(specific)
