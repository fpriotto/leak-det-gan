"""
Arquivo: src/training/train_gan.py
Descrição: Loops de treinamento customizados. Contém a WGANTrainer (sua arquitetura principal com WGAN-GP)
           e a CGANTrainer (Vanilla GAN) para os testes de ablação da dissertação.
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T


def calc_grad_penalty(D, real, fake, cond, lambda_gp, device):
    """Penalidade de Gradiente para o Crítico Temporal (Condicional)."""
    alpha = torch.rand(real.size(0), 1, 1, device=device)
    inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_inter = D(inter, cond)
    grad = torch.autograd.grad(
        outputs=d_inter,
        inputs=inter,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(grad.size(0), -1)
    return lambda_gp * ((grad.norm(2, dim=1) - 1) ** 2).mean()


def calc_grad_penalty_spec(D, real, fake, lambda_gp, device):
    """Penalidade de Gradiente para o Crítico Espectral (Não condicional)."""
    alpha = torch.rand(real.size(0), 1, 1, device=device)
    inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_inter = D(inter)
    grad = torch.autograd.grad(
        outputs=d_inter,
        inputs=inter,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(grad.size(0), -1)
    return lambda_gp * ((grad.norm(2, dim=1) - 1) ** 2).mean()


class WGANTrainer:
    """Classe responsável por orquestrar o treinamento da WGAN-GP (Single ou Dual Critic)."""

    def __init__(self, G, D_time, D_spec, opt_G, opt_Dt, opt_Ds, cfg, device):
        self.G = G
        self.D_time = D_time
        self.D_spec = D_spec
        self.opt_G = opt_G
        self.opt_Dt = opt_Dt
        self.opt_Ds = opt_Ds
        self.cfg = cfg
        self.device = device

        self.use_spec_critic = self.cfg.get("use_spectral_critic", True)
        self.spec_layer = T.Spectrogram(n_fft=256, hop_length=128, power=2).to(device)

    def train_step(self, real_signals, conds):
        batch_size = real_signals.size(0)

        # 1. TREINO DOS CRÍTICOS (D)
        for _ in range(self.cfg["n_critic"]):
            z = torch.randn(batch_size, self.cfg["z_dim"], device=self.device)
            fake_signals = self.G(z, conds).detach()

            d_real_t = self.D_time(real_signals, conds).mean()
            d_fake_t = self.D_time(fake_signals, conds).mean()
            gp_t = calc_grad_penalty(
                self.D_time,
                real_signals,
                fake_signals,
                conds,
                self.cfg["lambda_gp"],
                self.device,
            )
            loss_Dt = d_fake_t - d_real_t + gp_t

            self.opt_Dt.zero_grad()
            loss_Dt.backward()
            self.opt_Dt.step()

            if self.use_spec_critic:
                spec_real = torch.log1p(self.spec_layer(real_signals.squeeze(1)))
                spec_fake = torch.log1p(self.spec_layer(fake_signals.squeeze(1)))
                d_real_s = self.D_spec(spec_real).mean()
                d_fake_s = self.D_spec(spec_fake).mean()
                gp_s = calc_grad_penalty_spec(
                    self.D_spec,
                    spec_real,
                    spec_fake,
                    self.cfg["lambda_gp"],
                    self.device,
                )
                loss_Ds = d_fake_s - d_real_s + gp_s

                self.opt_Ds.zero_grad()
                loss_Ds.backward()
                self.opt_Ds.step()
            else:
                loss_Ds = torch.tensor(0.0)

        # 2. TREINO DO GERADOR (G)
        z = torch.randn(batch_size, self.cfg["z_dim"], device=self.device)
        fake_signals = self.G(z, conds)

        adv_loss_t = -self.D_time(fake_signals, conds).mean()
        if self.use_spec_critic:
            spec_fake = torch.log1p(self.spec_layer(fake_signals.squeeze(1)))
            adv_loss_s = -self.D_spec(spec_fake).mean()
        else:
            adv_loss_s = torch.tensor(0.0)

        loss_G = adv_loss_t + adv_loss_s

        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

        return (
            loss_Dt.item(),
            loss_Ds.item() if self.use_spec_critic else 0.0,
            loss_G.item(),
        )


class CGANTrainer:
    """Classe responsável pelo treinamento da Vanilla Conditional GAN (BCE Loss).
    Usada como baseline para provar a superioridade da WGAN-GP.
    """

    def __init__(self, G, D_time, opt_G, opt_Dt, cfg, device):
        self.G = G
        self.D_time = D_time  # Atuando como Discriminador Clássico
        self.opt_G = opt_G
        self.opt_Dt = opt_Dt
        self.cfg = cfg
        self.device = device

        # Função de perda para GAN Clássica (1 = Real, 0 = Falso)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_step(self, real_signals, conds):
        batch_size = real_signals.size(0)

        # Labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # ---------------------
        # 1. TREINO DO DISCRIMINADOR (D)
        # ---------------------
        z = torch.randn(batch_size, self.cfg["z_dim"], device=self.device)
        fake_signals = self.G(
            z, conds
        ).detach()  # Detach pro gradiente não ir pro Gerador

        # Predições e Loss
        d_real = self.D_time(real_signals, conds)
        loss_D_real = self.criterion(d_real, real_labels)

        d_fake = self.D_time(fake_signals, conds)
        loss_D_fake = self.criterion(d_fake, fake_labels)

        loss_D = (loss_D_real + loss_D_fake) / 2

        self.opt_Dt.zero_grad()
        loss_D.backward()
        self.opt_Dt.step()

        # ---------------------
        # 2. TREINO DO GERADOR (G)
        # ---------------------
        z = torch.randn(batch_size, self.cfg["z_dim"], device=self.device)
        fake_signals = self.G(z, conds)

        # O Gerador quer que o Discriminador classifique o Falso(0) como Real(1)
        d_fake_g = self.D_time(fake_signals, conds)
        loss_G = self.criterion(d_fake_g, real_labels)

        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

        # Retorna 0.0 para a loss do espectro para manter compatibilidade com os logs
        return loss_D.item(), 0.0, loss_G.item()
