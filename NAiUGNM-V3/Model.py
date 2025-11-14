# ==================================================
# TRAINING UTILITY FUNCTIONS
# ==================================================
import torch
import re
import os


def extract_epoch(filename):
    match = re.search(r"epoch_(\d+)", filename)
    return int(match.group(1)) if match else -1


def save_checkpoint(checkpoint_dir, epoch, generator, discriminator, optimizerG, optimizerD,
                    G_losses, D_losses, fixed_noise, training_config, max_checkpoints=4):
    """
    Save complete training state for resuming training seamlessly.

    Args:
        checkpoint_dir: Directory to save checkpoints
        epoch: Current epoch number
        generator: Generator model
        discriminator: Discriminator model
        optimizerG: Generator optimizer
        optimizerD: Discriminator optimizer
        G_losses: List of generator losses
        D_losses: List of discriminator losses
        fixed_noise: Fixed noise tensor for visualization
        training_config: Dict with training configuration (batch_size, lr, etc.)
        max_checkpoints: Maximum number of checkpoints to keep
    """
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save({
        "epoch": epoch,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "optimizerG_state_dict": optimizerG.state_dict(),
        "optimizerD_state_dict": optimizerD.state_dict(),
        "G_losses": G_losses,
        "D_losses": D_losses,
        "fixed_noise": fixed_noise,
        "training_config": training_config,
    }, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")

    # Keep only last N checkpoints to save disk space
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    checkpoints = sorted(checkpoints, key=extract_epoch)
    while len(checkpoints) > max_checkpoints:
        old_ckpt = os.path.join(checkpoint_dir, checkpoints[0])
        os.remove(old_ckpt)
        print(f"🗑️  Removed old checkpoint: {old_ckpt}")
        checkpoints.pop(0)


def load_latest_checkpoint(checkpoint_dir, generator, discriminator, optimizerG, optimizerD, device):
    """
    Load the latest checkpoint and restore complete training state.

    Returns:
        tuple: (start_epoch, G_losses, D_losses, fixed_noise, training_config)
               Returns (0, [], [], None, {}) if no checkpoint found
    """
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    checkpoints = sorted(checkpoints, key=extract_epoch)

    if checkpoints:
        latest_ckpt = os.path.join(checkpoint_dir, checkpoints[-1])
        checkpoint = torch.load(latest_ckpt, map_location=device)

        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        optimizerG.load_state_dict(checkpoint["optimizerG_state_dict"])
        optimizerD.load_state_dict(checkpoint["optimizerD_state_dict"])

        start_epoch = checkpoint["epoch"]
        G_losses = checkpoint.get("G_losses", [])
        D_losses = checkpoint.get("D_losses", [])
        fixed_noise = checkpoint.get("fixed_noise", None)
        training_config = checkpoint.get("training_config", {})

        print(f"✅ Resuming from checkpoint: {latest_ckpt} (epoch {start_epoch})")
        print(f"   Loaded {len(G_losses)} loss entries")
        return start_epoch, G_losses, D_losses, fixed_noise, training_config
    else:
        print("🚀 Starting training from scratch.")
        return 0, [], [], None, {}

# GENERIC FUNCTIONS

def save_checkpoint_generic(checkpoint_dir, epoch, state_dict, max_checkpoints=4):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")

    # Add epoch to state_dict if not present
    if 'epoch' not in state_dict:
        state_dict['epoch'] = epoch

    torch.save(state_dict, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")

    # Clean old checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir)
                   if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    checkpoints = sorted(checkpoints, key=extract_epoch)

    while len(checkpoints) > max_checkpoints:
        old_ckpt = os.path.join(checkpoint_dir, checkpoints[0])
        os.remove(old_ckpt)
        print(f"🗑️  Removed old checkpoint: {old_ckpt}")
        checkpoints.pop(0)


def load_checkpoint_generic(checkpoint_dir, device='cpu'):
    checkpoints = [f for f in os.listdir(checkpoint_dir)
                   if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    checkpoints = sorted(checkpoints, key=extract_epoch)

    if checkpoints:
        latest_ckpt = os.path.join(checkpoint_dir, checkpoints[-1])
        checkpoint = torch.load(latest_ckpt, map_location=device)
        print(f"✅ Loaded checkpoint: {latest_ckpt} (epoch {checkpoint.get('epoch', '?')})")
        return checkpoint
    else:
        print("🚀 No checkpoint found, starting from scratch")
        return {}


# ==================================================
# DCGAN MODEL DEFINITION
# ==================================================
import torch.nn as nn


# Generator

class DCGAN_Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=3, feature_g=64):
        super(DCGAN_Generator, self).__init__()
        self.block1 = self._block(z_dim, feature_g * 8, stride=(1, 1), padding=(0, 0))
        self.block2 = self._block(feature_g * 8, feature_g * 4, stride=(2, 2), padding=(1, 1))
        self.block3 = self._block(feature_g * 4, feature_g * 2, stride=(2, 2), padding=(1, 1))
        self.block4 = self._block(feature_g * 2, feature_g * 1, stride=(2, 2), padding=(1, 1))
        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                feature_g, img_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False
            ),
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, (4, 4), stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.final(x)


# Discriminator

class DCGAN_Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=64, negative_slope=0.2):
        super(DCGAN_Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                img_channels, feature_d, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False
            ),
            nn.LeakyReLU(negative_slope),
        )
        self.block2 = self._block(feature_d, feature_d * 2, stride=(2, 2), padding=(1, 1))
        self.block3 = self._block(feature_d * 2, feature_d * 4, stride=(2, 2), padding=(1, 1))
        self.block4 = self._block(feature_d * 4, feature_d * 8, stride=(2, 2), padding=(1, 1))
        self.final = nn.Sequential(
            nn.Conv2d(feature_d * 8, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0), bias=False),
            nn.Sigmoid(),
        )

    def _block(self, in_channels, out_channels, stride, padding, negative_slope=0.2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (4, 4), stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.final(x).view(-1, 1).squeeze(1)


# ==================================================
# DCGAN Dataset
# ==================================================
import torchvision
from torchvision import transforms
from pathlib import Path


class DCGAN_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.files = [p for p in self.root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = str(self.files[idx])
        img = torchvision.io.read_image(path).float() / 255.0  # C x H x W, float [0,1]
        if self.transform is not None:
            pil = transforms.ToPILImage()(img)
            return self.transform(pil)
        return img


# ============================================
# CycleGAN Models
# ============================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class CycleGAN_Generator(nn.Module):
    def __init__(self, img_channels=3, num_residuals=6):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 64, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residuals)]
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, img_channels, 7, 1, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.residuals(x)
        return self.decoder(x)


class CycleGAN_Discriminator(nn.Module):
    def __init__(self, img_channels=3):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 1, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# ============================================
# CycleGAN Dataset
# ============================================

class CycleGAN_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.root_A = Path(root_A)
        self.root_B = Path(root_B)
        self.transform = transform

        self.files_A = sorted([p for p in self.root_A.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        self.files_B = sorted([p for p in self.root_B.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

        self.len_A = len(self.files_A)
        self.len_B = len(self.files_B)

    def __len__(self):
        return max(self.len_A, self.len_B)

    def __getitem__(self, idx):
        img_A = torchvision.io.read_image(str(self.files_A[idx % self.len_A])).float() / 255.0
        img_B = torchvision.io.read_image(str(self.files_B[idx % self.len_B])).float() / 255.0

        if self.transform:
            img_A = self.transform(transforms.ToPILImage()(img_A))
            img_B = self.transform(transforms.ToPILImage()(img_B))

        return img_A, img_B
