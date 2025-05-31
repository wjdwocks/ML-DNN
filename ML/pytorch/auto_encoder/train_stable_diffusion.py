import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from Diffusion_Model import MNISTDiffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse

def create_mnist_dataloaders(batch_size,image_size=224,num_workers=4):
    
    preprocess=transforms.Compose([transforms.Resize(224),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)



def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',  type = str,help = 'define checkpoint path',default='results/steps_00001876.pt')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=9)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')

    args = parser.parse_args()

    return args


def main(args):
    device="cpu" if args.cpu else "cuda"
    train_dataloader,test_dataloader=create_mnist_dataloaders(batch_size=args.batch_size,image_size=28)
    model=MNISTDiffusion(timesteps=args.timesteps,
                image_size=28,
                in_channels=1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4]).to(device)

    #torchvision ema setting
    #https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer=AdamW(model.parameters(),lr=args.lr)
    scheduler=OneCycleLR(optimizer,args.lr,total_steps=args.epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
    loss_fn=nn.MSELoss(reduction='mean')

    #load checkpoint
    if args.ckpt:
        ckpt=torch.load(args.ckpt) # .pt로 끝나는 모델을 불러올 것인가? 하는것임.
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(ckpt["model"])

    global_steps=1876
    for i in range(args.epochs):
        model.train()
        # 전체 Training Dataset을 배치 단위로 불러들임.
        for j,(image,target) in enumerate(train_dataloader):
            # 원본 이미지와 동일한 크기의 정규 분포 노이즈 생성. 
            noise=torch.randn_like(image).to(device) 
            # 이미지와 노이즈를 gpu로 이동
            image=image.to(device)
            # 모델이 주어진 noisy image로부터 노이즈를 예측함(Denoising learning)
            pred=model(image,noise)
            # 예측한 노이즈와 실제 노이즈 간의 차이를 Loss로 계산
            loss=loss_fn(pred,noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() # Learning Rate 갱신을 위한 코드
            # EMA(step 마다 일정 주기로 moving average된 파라미터로 모델을 보관함)
            if global_steps%args.model_ema_steps==0:
                model_ema.update_parameters(model)
            global_steps+=1
            # 로그 출력 조건에 맞는 경우 현재 상태 출력
            if j%args.log_freq==0:
                print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i+1,args.epochs,j,len(train_dataloader),
                                                                    loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
        # ----한 Epoch이 종료되고 난 뒤 ----
        # 현재 모델과 EMA모델의 State-Dict 저장
        ckpt={"model":model.state_dict(),
                "model_ema":model_ema.state_dict()}
        # 폴더가 없다면 만들고, 파일 저장
        os.makedirs("results",exist_ok=True)
        torch.save(ckpt,"results/steps_{:0>8}.pt".format(global_steps))
        # EMA 모델로 샘플 이미지 생성.
        model_ema.eval()
        samples=model_ema.module.sampling(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device)
        save_image(samples,"results/steps_{:0>8}.png".format(global_steps),nrow=int(math.sqrt(args.n_samples)))

if __name__=="__main__":
    args=parse_args()
    main(args)