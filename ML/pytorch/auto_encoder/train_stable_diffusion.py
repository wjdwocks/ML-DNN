import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import random_split
from Diffusion_Model import StableDiffusion
from utils import ExponentialMovingAverage
from PIL import Image
import os
import math
import argparse


# 비율 유지 padding + resize
def pad_and_resize(img, size=224):
    w, h = img.size
    max_side = max(w, h)
    padded = Image.new("RGB", (max_side, max_side), color=(255, 255, 255))
    padded.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
    return padded.resize((size, size))


def create_dataloaders(batch_size, image_size=224, num_workers=4, data_root="anime_images", train_ratio=0.9):
    # 전처리 정의
    preprocess = transforms.Compose([
        transforms.Lambda(lambda img: pad_and_resize(img, size=image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [0, 1] → [-1, 1]
    ])

    # 전체 데이터셋 로드
    full_dataset = ImageFolder(root=data_root, transform=preprocess)
    
    # idx → text 매핑
    idx_to_text = {v: k for k, v in full_dataset.class_to_idx.items()}

    # train / test split
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # DataLoader 반환
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        idx_to_text  # ← label index를 prompt 텍스트로 매핑하는 데 필요
    )



def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=2)    
    parser.add_argument('--epochs',type = int,default=100)
    parser.add_argument('--ckpt',  type = str,help = 'define checkpoint path',default='anime_results/steps_00037339.pt')
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
    train_dataloader,test_dataloader, idx_to_text=create_dataloaders(batch_size=args.batch_size,image_size=224)
    model=StableDiffusion(timesteps=args.timesteps,
                image_size=224,
                in_channels=3,
                base_dim=args.model_base_dim,
                dim_mults=[2,4]).to(device)
    
    #idx_to_text 보기
    print('idx_to_text.items() : ', idx_to_text.items())

    # 미리 각 label에 대한 text embedding을 만들어서 넣어둠.
    # 1. Tokenizer 및 TextEncoder 로드
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_encoder.eval()  # 학습 안 함
    for p in text_encoder.parameters():
        p.requires_grad = False
        
    # 2. 각 label index에 대한 text embedding 생성 및 캐싱
    label_emb_dict = {}
    for idx, label_text in idx_to_text.items():
        # ① 텍스트 → 토큰
        inputs = tokenizer([label_text], return_tensors="pt", padding=True, truncation=True).to(device)

        # ② 텍스트 → 임베딩
        with torch.no_grad():
            text_emb = text_encoder(**inputs).last_hidden_state  # [1, L, D]

        # ③ 첫 배치 차원 제거 → [L, D] 형태로 저장
        label_emb_dict[idx] = text_emb.mean(dim=1).squeeze(0)  # [1, D]
    
    ############################

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

    global_steps=0
    for i in range(args.epochs):
        model.train()
        # 전체 Training Dataset을 배치 단위로 불러들임.
        for j,(image,label) in enumerate(train_dataloader):
            # 원본 이미지와 동일한 크기의 정규 분포 노이즈 생성. 
            noise=torch.randn_like(image).to(device) 
            # 이미지와 노이즈를 gpu로 이동
            image=image.to(device)
            # label index를 보고, text_emb vector로 변환.
            text_emb = torch.stack([label_emb_dict[l.item()] for l in label]).to(device)  # [B, L, D]
            # 모델이 주어진 noisy image로부터 노이즈를 예측함(Denoising learning)
            pred=model(image, noise, text_emb)
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
        os.makedirs("anime_results",exist_ok=True)
        torch.save(ckpt,"anime_results/steps_{:0>8}.pt".format(i))
        # EMA 모델로 샘플 이미지 생성.
        model_ema.eval()
        
        # prompt 생성 및 embedding
        prompt = tokenizer(['higurashi'], return_tensors="pt", padding=True, truncation=True).to(device)
        prompt_emb = text_encoder(**prompt).last_hidden_state  # [1, L, D]
        prompt_emb = prompt_emb.mean(dim=1).squeeze(0)
        
        samples=model_ema.module.sampling(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device, text_emb = prompt_emb)
        save_image(samples,"anime_results/anime_steps_{:0>8}.png".format(i),nrow=int(math.sqrt(args.n_samples)))

if __name__=="__main__":
    args=parse_args()
    main(args)