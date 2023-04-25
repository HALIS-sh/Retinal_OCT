import torch
import torchvision.utils as vutils
import os

def checkpoint(epoch, G_XtoY, G_YtoX, D_X, D_Y):
    """Save the model parameters for each epoch"""
    checkpoint_dir = "F:/CycleGAN_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
    torch.save({
        "G_XtoY": G_XtoY.state_dict(),
        "G_YtoX": G_YtoX.state_dict(),
        "D_X": D_X.state_dict(),
        "D_Y": D_Y.state_dict()
    }, checkpoint_path)


def save_samples(epoch, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=4, sample_dir='F:/samples_cyclegan'):

    # generate fake images
    fake_X = G_YtoX(fixed_Y)
    fake_Y = G_XtoY(fixed_X)

    # # concatenate real and fake images
    # X = torch.cat([fixed_X, fake_X], dim=0)
    # Y = torch.cat([fixed_Y, fake_Y], dim=0)

    # Save translated X images
    fake_X = fake_X.clone().cpu().data
    grid_fake_X = vutils.make_grid(fake_X, nrow=batch_size, normalize=True, scale_each=True)
    vutils.save_image(grid_fake_X, f'{sample_dir}/sample-{epoch}-Y-X.png')

    # Save translated Y images
    fake_Y = fake_Y.clone().cpu().data
    grid_fake_Y = vutils.make_grid(fake_Y, nrow=batch_size, normalize=True, scale_each=True)
    vutils.save_image(grid_fake_Y, f'{sample_dir}/sample_{epoch}-X-Y.png')

    # # create grid of images
    # grid_X = vutils.make_grid(X, nrow=batch_size, normalize=True, scale_each=True)
    # grid_Y = vutils.make_grid(Y, nrow=batch_size, normalize=True, scale_each=True)
    # 
    # 
    # # save the grid of images
    # vutils.save_image(grid_X, f'{sample_dir}/sample-{epoch}-Y-X.png')
    # vutils.save_image(grid_Y, f'{sample_dir}/sample_{epoch}-X-Y.png')

