import torch
import torch.nn as nn
import torchaudio.transforms as T


def _grad_penalty(D, real, fake, cond, lambda_gp, device):
    alpha = torch.rand(real.size(0), 1, 1, device=device)
    inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    grad = torch.autograd.grad(
        outputs=D(inter, cond),
        inputs=inter,
        grad_outputs=torch.ones(real.size(0), 1, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0].view(real.size(0), -1)
    return lambda_gp * ((grad.norm(2, dim=1) - 1) ** 2).mean()


def _grad_penalty_spec(D, real, fake, lambda_gp, device):
    alpha = torch.rand(real.size(0), 1, 1, device=device)
    inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    grad = torch.autograd.grad(
        outputs=D(inter),
        inputs=inter,
        grad_outputs=torch.ones(real.size(0), 1, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0].view(real.size(0), -1)
    return lambda_gp * ((grad.norm(2, dim=1) - 1) ** 2).mean()


class WGANTrainer:
    def __init__(self, G, D_time, D_spec, opt_G, opt_Dt, opt_Ds, cfg, device):
        self.G = G
        self.D_time = D_time
        self.D_spec = D_spec
        self.opt_G = opt_G
        self.opt_Dt = opt_Dt
        self.opt_Ds = opt_Ds
        self.cfg = cfg
        self.device = device
        self.use_spec = cfg.get("use_spectral_critic", True)
        self.spec_layer = T.Spectrogram(n_fft=256, hop_length=128, power=2).to(device)

    def train_step(self, real_signals, conds):
        bs = real_signals.size(0)

        for _ in range(self.cfg["n_critic"]):
            z = torch.randn(bs, self.cfg["z_dim"], device=self.device)
            fake = self.G(z, conds).detach()

            loss_Dt = (
                self.D_time(fake, conds).mean()
                - self.D_time(real_signals, conds).mean()
                + _grad_penalty(self.D_time, real_signals, fake, conds, self.cfg["lambda_gp"], self.device)
            )
            self.opt_Dt.zero_grad()
            loss_Dt.backward()
            self.opt_Dt.step()

            if self.use_spec:
                spec_real = torch.log1p(self.spec_layer(real_signals.squeeze(1)))
                spec_fake = torch.log1p(self.spec_layer(fake.squeeze(1)))
                loss_Ds = (
                    self.D_spec(spec_fake).mean()
                    - self.D_spec(spec_real).mean()
                    + _grad_penalty_spec(self.D_spec, spec_real, spec_fake, self.cfg["lambda_gp"], self.device)
                )
                self.opt_Ds.zero_grad()
                loss_Ds.backward()
                self.opt_Ds.step()
            else:
                loss_Ds = torch.tensor(0.0)

        z = torch.randn(bs, self.cfg["z_dim"], device=self.device)
        fake = self.G(z, conds)
        loss_G = -self.D_time(fake, conds).mean()
        if self.use_spec:
            loss_G = loss_G - self.D_spec(torch.log1p(self.spec_layer(fake.squeeze(1)))).mean()

        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

        return loss_Dt.item(), loss_Ds.item() if self.use_spec else 0.0, loss_G.item()


class CGANTrainer:
    """Vanilla conditional GAN with BCE loss — used as ablation baseline."""

    def __init__(self, G, D_time, opt_G, opt_Dt, cfg, device):
        self.G = G
        self.D_time = D_time
        self.opt_G = opt_G
        self.opt_Dt = opt_Dt
        self.cfg = cfg
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()

    def train_step(self, real_signals, conds):
        bs = real_signals.size(0)
        real_labels = torch.ones(bs, 1, device=self.device)
        fake_labels = torch.zeros(bs, 1, device=self.device)

        z = torch.randn(bs, self.cfg["z_dim"], device=self.device)
        fake = self.G(z, conds).detach()
        loss_D = (
            self.criterion(self.D_time(real_signals, conds), real_labels)
            + self.criterion(self.D_time(fake, conds), fake_labels)
        ) / 2
        self.opt_Dt.zero_grad()
        loss_D.backward()
        self.opt_Dt.step()

        z = torch.randn(bs, self.cfg["z_dim"], device=self.device)
        loss_G = self.criterion(self.D_time(self.G(z, conds), conds), real_labels)
        self.opt_G.zero_grad()
        loss_G.backward()
        self.opt_G.step()

        return loss_D.item(), 0.0, loss_G.item()
