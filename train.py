import argparse
import os
from math import log10

import pandas as pd
from sklearn.model_selection import train_test_split
import torch 
import torch.optim as optim
import torch.utils.data
from torch.utils.data import dataset
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from data_utils import TrainDatasetFromList, ValDatasetFromList, display_transform
from loss import GeneratorLoss
from discriminator_network import Discriminator
from generator_network import Generator

# Speed up Pytorch https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/

if __name__ == '__main__':
    
    CROP_SIZE = 80 # (crop size by crop size)
    UPSCALE_FACTOR = 4
    NUM_EPOCHS = 100
    ACCUMULATION_STEPS=100

    # Make sure the bit depth is 24, 8 = Gray scale
    df = pd.read_pickle('data/dataset_files.pickle')
    df = df[(df['channels']==3) & (df['width']>100) & (df['height']>100)]
    train_df, val_df = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)
    train_filenames = train_df['filename'].tolist()
    val_filenames = val_df['filename'].tolist()
    if (not os.path.exists('data/dataset.pt')):
        train_set = TrainDatasetFromList(train_filenames, crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR)
        val_set = ValDatasetFromList(val_filenames, upscale_factor=UPSCALE_FACTOR)
        data_to_save = {'train_dataset':train_set,"val_dataset":val_set}
        torch.save(data_to_save,'data/dataset.pt')
    else:
        datasets = torch.load('data/dataset.pt')
        train_set = datasets['train_dataset']
        val_set = datasets['val_dataset']

    train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)
    
    netG = Generator(UPSCALE_FACTOR)
    print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))
    
    generator_criterion = GeneratorLoss()
    
    if torch.cuda.is_available():
        netG.cuda()
        netD.cuda()
        generator_criterion.cuda()
    
    optimizerG = optim.Adam(netG.parameters())
    optimizerD = optim.Adam(netD.parameters())
    
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
    dscaler = torch.cuda.amp.GradScaler() # Creates once at the beginning of training #* Discriminator 
    gscaler = torch.cuda.amp.GradScaler() #* Generator

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    
        netG.train()
        netD.train()
        i=0
        for data, target in train_bar:
            with torch.cuda.amp.autocast():        # Mix precision
                g_update_first = True
                batch_size = data.size(0)
                running_results['batch_sizes'] += batch_size
        
                ############################
                # (1) Update D network: maximize D(x)-1-D(G(z))
                ###########################
                netD.zero_grad()
                real_img = Variable(target, requires_grad=False)
                if torch.cuda.is_available():
                    real_img = real_img.cuda()
                z = Variable(data)
                if torch.cuda.is_available():
                    z = z.cuda()
                fake_img = netG(z)

                real_out = netD(real_img).mean()    # Discriminator Takes in the real image and predicts whether it's real
                fake_out = netD(fake_img).mean()    # Discriminator takes in the fake image and predicts if it's fake
                d_loss = 1 - real_out + fake_out    # Minimizing the loss would mean real_out=1 and fake out = 0. so it knows the real image it knows the fake image
                dscaler.scale(d_loss/ACCUMULATION_STEPS).backward(retain_graph=True)

                # d_loss.backward(retain_graph=True)
                # optimizerD.step()
                ############################
                # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
                ###########################
                netG.zero_grad()
                g_loss = generator_criterion(fake_out.detach(), fake_img, real_img.detach())
                gscaler.scale(g_loss/ACCUMULATION_STEPS).backward()

                # g_loss.backward()
                
                fake_img = netG(z)
                fake_out = netD(fake_img).mean()
                #optimizerG.step()
                i+=1
                if (i % ACCUMULATION_STEPS) == 0 or i==len(train_loader):
                     dscaler.step(optimizerD)
                     dscaler.update()
                     gscaler.step(optimizerG)                     
                     gscaler.update()

                
                # loss for current batch before optimization 
                running_results['g_loss'] += g_loss.item() * batch_size
                running_results['d_loss'] += d_loss.item() * batch_size
                running_results['d_score'] += real_out.item() * batch_size
                running_results['g_score'] += fake_out.item() * batch_size
        
                train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                    epoch, NUM_EPOCHS, running_results['d_loss'] / running_results['batch_sizes'],
                    running_results['g_loss'] / running_results['batch_sizes'],
                    running_results['d_score'] / running_results['batch_sizes'],
                    running_results['g_score'] / running_results['batch_sizes']))
    
        netG.eval()
        out_path = 'training_results/SRF_' + str(UPSCALE_FACTOR) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path, exist_ok=True)
        
        with torch.no_grad():
            val_bar = tqdm(val_loader)
            val_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                val_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)
        
                batch_mse = ((sr - hr) ** 2).data.mean()
                val_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                val_results['ssims'] += batch_ssim * batch_size
                val_results['psnr'] = 10 * log10((hr.max()**2) / (val_results['mse'] / val_results['batch_sizes']))
                val_results['ssim'] = val_results['ssims'] / val_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        val_results['psnr'], val_results['ssim']))
        
                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1
    
        # save model parameters
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (UPSCALE_FACTOR, epoch))
        # save loss\scores\psnr\ssim
        results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
        results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
        results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
        results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
        results['psnr'].append(val_results['psnr'])
        results['ssim'].append(val_results['ssim'])
    
        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(UPSCALE_FACTOR) + '_train_results.csv', index_label='Epoch')
