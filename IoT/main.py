import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from data.dataloader import RGB_Dataset
# 按照老师的要求，模型命名为自己的名字
from model import Liyutong
from tqdm import tqdm
import cv2

def train(model, train_loader, optimizer, criterion, epochs=100):
    best_mIoU = 0.0
    loss_history = []  # 用于记录每一轮的损失
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Lyt", ncols=140)
        for i, data_batch in enumerate(progress_bar):
            images = data_batch['image']
            labels = data_batch['gt']
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)  # 计算损失

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss_avg = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")
        loss_history.append(loss_avg)
    if epoch +1 == epochs:
        torch.save(model.state_dict(), 'latest_model.pth')

    return loss_history

def compute_iou(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()

def test(model, test_loader):
    # model.load_state_dict(torch.load('latest_model.pth'))
    model.eval()
    total_iou = 0
    total_samples = 0
    progress_bar = tqdm(test_loader, desc="Lyt", ncols=140)
    with torch.no_grad():
        for i, data_batch in enumerate(progress_bar):
            images = data_batch['image']
            labels = data_batch['gt']
            images, labels = images.cuda(), labels.cuda()
            shape = data_batch['shape']
            preds = model(images)
            iou = compute_iou(preds, labels)
            total_iou += iou
            total_samples += 1
        miou = total_iou / total_samples
        print("李禹佟的物联网大作业预测!")
        print(f"测试 mIoU 值为: {miou:.4f}")

def save_loss_history(loss_history, filename='loss_history.log'):

    with open(filename, 'w') as f:
        for epoch, loss in enumerate(loss_history, start=1):
            f.write(f'Epoch {epoch:03d}: {loss:.6f}\n')


if __name__ == '__main__':
    # data_root
    data_root = r"C:\Users\lyttt\Desktop\ORSSD"
    # train data
    train_dataset = RGB_Dataset(root=data_root , sets=['train'], img_size=224, mode='train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=8,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)
    # test data
    test_dataset = RGB_Dataset(data_root , ['test'], 224, mode='Test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)

    # model
    liyutong = Liyutong().cuda()
    # loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(liyutong.parameters(), lr=1e-4)
    loss = train(liyutong, train_loader, optimizer, criterion)
    save_loss_history(loss, 'train_loss.log')

    test(liyutong, test_loader)

