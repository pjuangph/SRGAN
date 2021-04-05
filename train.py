import argparse
import os
from math import log10
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import torch 
import torch.optim as optim
import torch.utils.data
from torch.utils.data import dataset
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import pytorch_ssim
from data_utils import TrainDatasetFromList, ValDatasetFromList, display_transform,dataframe_find_similar_images
from loss import GeneratorLoss
from discriminator_network import Discriminator
from generator_network import Generator

# Speed up Pytorch https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/

def get_args_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    parser = argparse.ArgumentParser('Set SuperResolution GANS', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)    
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--crop_size', default=80, type=float, help='Image crop size in pixels, square')
    parser.add_argument('--upscale_factor', default=4, type=float, help='upscale image')
    parser.add_argument('--resume',default=True,type=str2bool, help='resume from checkpoint file')
    parser.add_argument('--validation_epoch', default=50, type=int, help='At what epoch do we run the validation')
    parser.add_argument('--output_dir',default='epochs',type=str, help='folder where to save checkpoints')
    return parser

def main(args):
    if (not os.path.exists('data/dataset.pt')):
        # Make sure the bit depth is 24, 8 = Gray scale
        df = pd.read_pickle('data/dataset_files.gzip')
        df = df[(df['width']>100) & (df['height']>100)]
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        _,val_similar = dataframe_find_similar_images(val_df,batch_size=args.batch_size)

        # Create the train dataset 
        train_filenames = train_df['filename'].tolist()        
        train_set = TrainDatasetFromList(train_filenames, crop_size=args.crop_size, upscale_factor=args.upscale_factor)

        val_sets = list() 
        for val_df in val_similar:
            val_filenames = val_df['filename'].tolist()
            val_set = ValDatasetFromList(val_filenames, upscale_factor=args.upscale_factor)
            val_sets.append(val_set)

        train_sampler = torch.utils.data.RandomSampler(train_set)
        val_sampler = torch.utils.data.SequentialSampler(val_set)
        data_to_save = {'train_dataset':train_set,"val_datasets":val_sets, 'train_sampler':train_sampler, 'val_sampler':val_sampler}
        torch.save(data_to_save,'data/dataset.pt')
    else:
        datasets = torch.load('data/dataset.pt')
        train_set = datasets['train_dataset']
        val_sets = datasets['val_datasets']
        train_sampler = datasets['train_sampler']
        val_sampler = datasets['val_sampler']

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=args.num_workers,sampler=train_sampler)
    val_loaders= list()
    for val_set in val_sets:
        val_loaders.append(DataLoader(dataset=val_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False))
    
    netG = Generator(args.upscale_factor)
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
    start_epoch = 1    
    if args.resume:
        import glob
        netG_files = glob.glob(os.path.join(args.output_dir, 'netG_epoch_%d_*.pth' % (args.upscale_factor))) 
        netD_files = glob.glob(os.path.join(args.output_dir, 'netD_epoch_%d_*.pth' % (args.upscale_factor)))
        if (len(netG_files)>0):
            netG_file = max(netG_files, key=os.path.getctime)
            netD_file = max(netD_files, key=os.path.getctime)
            netG.load_state_dict(torch.load(netG_file))
            netD.load_state_dict(torch.load(netD_file))
            start_epoch = len(netG_files)

    for epoch in range(start_epoch, args.epochs + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        dscaler = torch.cuda.amp.GradScaler() # Creates once at the beginning of training #* Discriminator 
        gscaler = torch.cuda.amp.GradScaler() #* Generator
        netG.train()
        netD.train()
        for data, target in train_bar:
            with torch.cuda.amp.autocast():        # Mix precision
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
                

                # d_loss.backward(retain_graph=True)
                # optimizerD.step()
                ############################
                # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
                ###########################    
                netG.zero_grad()
                g_loss = generator_criterion(fake_out.detach(), fake_img, real_img.detach())
                
                
            dscaler.scale(d_loss).backward(retain_graph=True)
            gscaler.scale(g_loss).backward()

            dscaler.step(optimizerD)
            dscaler.update()
            gscaler.step(optimizerG)                     
            gscaler.update()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            # loss for current batch before optimization 
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, args.epochs, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))
    
         # save model parameters
        torch.save(netG.state_dict(), os.path.join(args.output_dir, 'netG_epoch_%d_%d.pth' % (args.upscale_factor, epoch)))
        torch.save(netD.state_dict(), os.path.join(args.output_dir, 'netD_epoch_%d_%d.pth' % (args.upscale_factor, epoch)))

        if epoch % args.validation_epoch == 0 and epoch != 0:
            netG.eval()
            with torch.no_grad():
                val_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
                val_images = []
                for i in trange(len(val_loaders), desc='Running validation'):
                    val_loader = val_loaders[i]
                    for val_lr, val_hr_restore, val_hr in val_loader:
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
                        # val_bar.set_description(
                        #     desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        #         val_results['psnr'], val_results['ssim']))
                        
                        # convert the validation images
                        val_hr_restore_squeeze = val_hr_restore.squeeze(0)
                        hr_squeeze = hr.data.cpu().squeeze(0)
                        sr_squeeze = sr.data.cpu().squeeze(0)
                        for b in range(batch_size):
                            val_hr = val_hr_restore_squeeze[b]
                            hr_temp = hr_squeeze[b]
                            sr_temp = sr_squeeze[b]
                            val_images.extend([display_transform()(val_hr), display_transform()(hr_temp),display_transform()(sr_temp)])
                
                val_images = torch.stack(val_images)
                val_images = torch.chunk(val_images, val_images.size(0) // 15)
                val_save_bar = tqdm(val_images, desc='[saving training results]')
                index = 1
                for image in val_save_bar:
                    image = utils.make_grid(image, nrow=3, padding=5)
                    utils.save_image(image, os.path.join(args.output_dir,'epoch_%d_upscale_%d_index_%d.png' % (epoch, args.upscale_factor, index)))
                    index += 1


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
            data_frame.to_csv(out_path + 'srf_' + str(args.upscale_factor) + '_train_results.csv', index_label='Epoch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
