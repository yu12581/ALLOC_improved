import numpy as np
import os, glob, torch, random
from torch.utils.data import DataLoader
from model import ALOCC, ALOCC_LOSS, ALOCC_LOSS_CLS
from torch.utils.data import Dataset, TensorDataset
from Metrics import calculate_metrics, show_class_metrics
from utils import timer, DEVICE, set_random_seed
from shutil import copyfile
import torchvision

class UCSD(Dataset):
    data_root = 'dataset/UCSD/'
    train_files = glob.glob(os.path.join(data_root, 'Train2/*.png'))
    test_normal_files = glob.glob(os.path.join(data_root, 'Test2/Normal/*.png'))
    test_abnormal_files = glob.glob(os.path.join(data_root, 'Test2/Abnormal/*.png'))
    rng = np.random.RandomState(13)
    train_files = rng.choice(train_files, size=len(train_files), replace=False)
    test_normal_files = rng.choice(test_normal_files, size=len(test_normal_files), replace=False)
    test_abnormal_files = rng.choice(test_abnormal_files, size=len(test_abnormal_files), replace=False)
    def __init__(self, train=True, test_count=400, out_class_scale=0.5):
        self.data = None
        if train: # 训练集：从dataset/UCSD/Train/下面读取所有.png文件（都是Normal）
            train_files = self.train_files[:test_count]
            self.extend_from_files(train_files, 0)
        else: # 测试集：内类从Normal读取前1k个，外类从Abnormal读取前1k个
            # 内类（Normal）- 标签为0
            inner_class_count = int(test_count * (1 - out_class_scale))
            normal_files = self.test_normal_files[:inner_class_count]
            # 外类（Abnormal）- 标签为1
            out_class_count = int(test_count * out_class_scale)
            abnormal_files = self.test_abnormal_files[:out_class_count]
            
            self.extend_from_files(normal_files, 0)
            self.extend_from_files(abnormal_files, 1)

    def extend_from_files(self, files, label, noise_std=0.155*2):
        set_random_seed()
        cnt = len(files)
        if cnt > 0:
            # print(f"找到 {cnt} 个图片")
            imgs = torch.stack([
                torchvision.io.read_image(f, torchvision.io.ImageReadMode.GRAY).float() 
                for f in files
            ]).to(DEVICE, non_blocking=True) / 127.5 - 1.0
            labels = torch.full((cnt,), label, dtype=torch.int32, device=DEVICE)
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
    
def test(model, checkpoint_dir='checkpoint/ucsd', epoch=20, step=2, dataloader=None, find_best=False):
    best_index, best_metrics = 0, [0,0,0,0,0,0,0,0,0,0]
    for i in range(step, epoch + step, step):
        model._load_checkpoint(os.path.join(checkpoint_dir, f'{i}.pth'))
        metrics = calculate_metrics(model, dataloader, inner_class=0, verbose=True)
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
# 训练函数
def train(checkpoint_dir='checkpoint/ucsd', epoch=20, step=2, find_best=False):
    print("开始训练UCSD模型..."); 
    # 训练
    data_dataset = UCSD(train=True, test_count=5000)
    data_loader = DataLoader(data_dataset, batch_size=64, shuffle=False)
    model = ALOCC(in_h=45, out_h=45, lr=0.000001)
    model._train(data_loader=data_loader, epoch=epoch, step=step, checkpoint_dir=checkpoint_dir)
    # 测试
    test_dataset = UCSD(train=False, test_count=400, out_class_scale=0.5)
    test_loader = DataLoader(test_dataset, batch_size=400, shuffle=False)
    test(model, checkpoint_dir=checkpoint_dir, epoch=epoch, step=step, dataloader=test_loader, find_best=find_best)
    # 清空缓存
    del model, data_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def train_loss(checkpoint_dir='checkpoint/ucsd_loss', epoch=20, step=2, oc_cnt=20, find_best=False):
    print(f"开始训练ALOCC-LOSS模型: UCSD数据集 -> OC = {oc_cnt}"); 
    # 训练
    data_dataset = UCSD(train=True, test_count=5000)
    data_loader = DataLoader(data_dataset, batch_size=64, shuffle=False)
    model = ALOCC_LOSS(in_h=45, out_h=45, lr=0.000001)
    # print(outclass_files)
    outclass_dataset = UCSD(test_count=0)
    outclass_files = np.resize(UCSD.test_abnormal_files[200:200+oc_cnt], 5000)
    outclass_dataset.extend_from_files(outclass_files, 1)
    outclass_loader = DataLoader(outclass_dataset, batch_size=64, shuffle=False)
    model._train(data_loader=data_loader, outclass_loader=outclass_loader, epoch=epoch, step=step, checkpoint_dir=checkpoint_dir)
    # 测试
    test_dataset = UCSD(train=False, test_count=400, out_class_scale=0.5)
    test_loader = DataLoader(test_dataset, batch_size=400, shuffle=False)
    test(model, checkpoint_dir=checkpoint_dir, epoch=epoch, step=step, dataloader=test_loader, find_best=find_best)
    # 清空缓存
    del model, data_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def train_loss_cls(checkpoint_dir='checkpoint/ucsd_loss_cls', epoch=20, step=2, oc_cnt=20, find_best=False):
    print(f"开始训练ALOCC-LOSS-CLS模型: UCSD数据集 -> OC = {oc_cnt}"); 
    # 训练
    data_dataset = UCSD(train=True, test_count=5000)
    data_loader = DataLoader(data_dataset, batch_size=64, shuffle=False)
    model = ALOCC_LOSS_CLS(in_h=45, out_h=45, lr=0.000001, classify=True)
    # print(outclass_files)
    outclass_dataset = UCSD(test_count=0)
    outclass_files = np.resize(UCSD.test_abnormal_files[200:200+oc_cnt], 5000)
    outclass_dataset.extend_from_files(outclass_files, 1)
    outclass_loader = DataLoader(outclass_dataset, batch_size=64, shuffle=False)
    model._train(data_loader=data_loader, outclass_loader=outclass_loader, epoch=epoch, step=step, checkpoint_dir=checkpoint_dir)
    # 测试
    test_dataset = UCSD(train=False, test_count=400, out_class_scale=0.5)
    test_loader = DataLoader(test_dataset, batch_size=400, shuffle=False)
    test(model, checkpoint_dir=checkpoint_dir, epoch=epoch, step=step, dataloader=test_loader, find_best=find_best)
    # 清空缓存
    del model, data_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
# 测试函数
def test_normal(checkpoint_dir='checkpoint/ucsd', epoch=20, step=2, test_count=400):
    model = ALOCC(in_h=45, out_h=45)
    test_dataset = UCSD(train=False, test_count=test_count, out_class_scale=0.5)
    test_loader = DataLoader(test_dataset, batch_size=test_count, shuffle=False)
    test(model, checkpoint_dir=checkpoint_dir, epoch=epoch, step=step, dataloader=test_loader)

def test_loss(checkpoint_dir='checkpoint/ucsd_loss', epoch=20, step=2, test_count=400):
    model = ALOCC_LOSS(in_h=45, out_h=45)
    test_dataset = UCSD(train=False, test_count=test_count, out_class_scale=0.5)
    test_loader = DataLoader(test_dataset, batch_size=test_count, shuffle=False)
    test(model, checkpoint_dir=checkpoint_dir, epoch=epoch, step=step, dataloader=test_loader)
    
def test_loss_cls(checkpoint_dir='checkpoint/ucsd_loss_cls', epoch=20, step=2, test_count=400):
    model = ALOCC_LOSS_CLS(in_h=45, out_h=45, classify=True)
    test_dataset = UCSD(train=False, test_count=test_count, out_class_scale=0.5)
    test_loader = DataLoader(test_dataset, batch_size=test_count, shuffle=False)
    test(model, checkpoint_dir=checkpoint_dir, epoch=epoch, step=step, dataloader=test_loader)

if __name__ == '__main__':
    epoch = 10  # 增加训练epoch数量
    step = 1    # 每10个epoch保存一次    
    
    # 训练模型
    train(epoch=epoch, step=step)
    # test_normal(epoch=epoch, step=step)
    # 测试模型
    for oc_cnt in range(20, 120, 20):
        train_loss(epoch=epoch, step=step, oc_cnt=oc_cnt)
        # test_loss(epoch=epoch, step=step)
        
    for oc_cnt in range(20, 120, 20):
        train_loss_cls(epoch=epoch, step=step, oc_cnt=oc_cnt)
