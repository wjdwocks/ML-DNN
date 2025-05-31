import torch.nn as nn
import torch
import math
from UNet import Unet
from tqdm import tqdm

class StableDiffusion(nn.Module):
    def __init__(self,image_size,in_channels,time_embedding_dim=256,timesteps=1000,base_dim=32,dim_mults= [1, 2, 4, 8]):
        # time step : 몇 번에 걸쳐 노이즈를 추가/제거할지 정함.
        # beta, alpha, alphas_cumprod : 각 시간 t에서 사용할 노이즈의 분산 or 감쇠 정도를 미리 계산해둔 값.
        # register_buffer : 학습 중엔 사용되지만 학습되는 파라미터는 아닌 텐서를 등록 ????
        # time_embedding_dim : 현재 몇 번째 time step인지 알려주기 위해 정수 t를 256차원의 벡터로 인코딩할거임.
        super().__init__()
        self.timesteps=timesteps
        self.in_channels=in_channels
        self.image_size=image_size

        betas=self._cosine_variance_schedule(timesteps)

        alphas=1.-betas
        alphas_cumprod=torch.cumprod(alphas,dim=-1)

        # 이 Register_buffer는 학습되진 않지만 모델에 저장되어야 하는 값들을 처리할 때 사용된다.
        self.register_buffer("betas",betas)
        self.register_buffer("alphas",alphas)
        self.register_buffer("alphas_cumprod",alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod",torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",torch.sqrt(1.-alphas_cumprod))

        self.model=Unet(timesteps,time_embedding_dim,in_channels,in_channels,base_dim,dim_mults)

    def forward(self,x,noise, text_emb):
        # x:NCHW
        t=torch.randint(0,self.timesteps,(x.shape[0],)).to(x.device) # 배치에 있는 각 이미지마다, 0부터 (timesteps - 1) 사이의 랜덤한 timestep t를 하나씩 샘플링
        x_t=self._forward_diffusion(x,t,noise) # x: 원본 이미지 (x₀), t: 랜덤 timestep, noise: 정규분포 노이즈 → 이를 이용해 forward diffusion 공식을 따라 x_t 생성
        pred_noise=self.model(x_t,t, text_emb) # x_t (노이즈가 낀 이미지)와 시점 t를 넣어서, 그 안에 섞인 노이즈 ε를 예측함
        # 여기서 x_t는 이밎 x_0와 e가 합쳐진 결과고, t는 그게 어느 시점인지를 알려주기 때문에, x_t안에 e가 어떻게 들어갔는지 학습할 수 있음.
        return pred_noise       
    '''
    t=0이면 거의 원본 (x₀ ≈ xₜ), 노이즈는 거의 없음
    t=500이면 원본과 노이즈 반반
    t=999면 거의 순수 노이즈
    '''

    @torch.no_grad()
    def sampling(self,n_samples,clipped_reverse_diffusion=True,device="cuda", text_emb = None):
        x_t=torch.randn((n_samples,self.in_channels,self.image_size,self.image_size)).to(device)
        for i in tqdm(range(self.timesteps-1,-1,-1),desc="Sampling"):
            noise=torch.randn_like(x_t).to(device)
            t=torch.tensor([i for _ in range(n_samples)]).to(device)

            if clipped_reverse_diffusion:
                x_t=self._reverse_diffusion_with_clip(x_t,t,noise, text_emb)
            else:
                x_t=self._reverse_diffusion(x_t,t,noise, text_emb)

        x_t=(x_t+1.)/2. #[-1,1] to [0,1]

        return x_t
    
    def _cosine_variance_schedule(self,timesteps,epsilon= 0.008):
        steps=torch.linspace(0,timesteps,steps=timesteps+1,dtype=torch.float32)
        f_t=torch.cos(((steps/timesteps+epsilon)/(1.0+epsilon))*math.pi*0.5)**2
        betas=torch.clip(1.0-f_t[1:]/f_t[:timesteps],0.0,0.999)

        return betas

    def _forward_diffusion(self,x_0,t,noise):
        # x_t = sqrt(a_t) * x_0 + sqrt(1-a_t) * e : 이 공식으로 노이즈를 생성하는 함수임.
        assert x_0.shape==noise.shape
        #q(x_{t}|x_{t-1})
        return self.sqrt_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*x_0 + self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_0.shape[0],1,1,1)*noise


    @torch.no_grad()
    def _reverse_diffusion(self,x_t,t,noise, text_emb):
        '''
        p(x_{t-1}|x_{t})-> mean,std

        pred_noise-> pred_mean and pred_std
        '''
        pred=self.model(x_t,t, text_emb=text_emb)

        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        sqrt_one_minus_alpha_cumprod_t=self.sqrt_one_minus_alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        mean=(1./torch.sqrt(alpha_t))*(x_t-((1.0-alpha_t)/sqrt_one_minus_alpha_cumprod_t)*pred)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            std=0.0

        return mean+std*noise 


    @torch.no_grad()
    def _reverse_diffusion_with_clip(self,x_t,t,noise): 
        '''
        p(x_{0}|x_{t}),q(x_{t-1}|x_{0},x_{t})->mean,std

        pred_noise -> pred_x_0 (clip to [-1.0,1.0]) -> pred_mean and pred_std
        '''
        pred=self.model(x_t,t)
        alpha_t=self.alphas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        alpha_t_cumprod=self.alphas_cumprod.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        beta_t=self.betas.gather(-1,t).reshape(x_t.shape[0],1,1,1)
        
        x_0_pred=torch.sqrt(1. / alpha_t_cumprod)*x_t-torch.sqrt(1. / alpha_t_cumprod - 1.)*pred
        x_0_pred.clamp_(-1., 1.)

        if t.min()>0:
            alpha_t_cumprod_prev=self.alphas_cumprod.gather(-1,t-1).reshape(x_t.shape[0],1,1,1)
            mean= (beta_t * torch.sqrt(alpha_t_cumprod_prev) / (1. - alpha_t_cumprod))*x_0_pred +\
                    ((1. - alpha_t_cumprod_prev) * torch.sqrt(alpha_t) / (1. - alpha_t_cumprod))*x_t

            std=torch.sqrt(beta_t*(1.-alpha_t_cumprod_prev)/(1.-alpha_t_cumprod))
        else:
            mean=(beta_t / (1. - alpha_t_cumprod))*x_0_pred #alpha_t_cumprod_prev=1 since 0!=1
            std=0.0

        return mean+std*noise 
    