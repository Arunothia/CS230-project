import random, torch, os, numpy as np
import config
from torchvision.utils import save_image

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("===> Saving Checkpoint")
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("===> Loading Checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Reset learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def val(gen_F, gen_P, disc_F, disc_P, mse, L1, val_loader, idx, epoch, folder):
    loss = AverageMeter('Loss', ':.4e')
    loss_piano = AverageMeter('Gen_piano', ':.4e')
    loss_flute = AverageMeter('Gen_flute', ':.4e')
    loss_id_piano = AverageMeter('Identity_piano', ':.4e')
    loss_id_flute = AverageMeter('Identity_flute', ':.4e')
    loss_cyc_piano = AverageMeter('Cycle_piano', ':.4e')
    loss_cyc_flute = AverageMeter('Cycle_flute', ':.4e')
    progress = ProgressMeter(
        len(val_loader),
        [loss, loss_piano, loss_flute, loss_id_piano, loss_id_flute, loss_cyc_piano, loss_cyc_flute],
        prefix='Val: ')
    gen_F.eval(), gen_P.eval(), disc_F.eval(), disc_P.eval()
    for e, (flute, piano) in enumerate(val_loader):
        flute, piano = flute.to(config.DEVICE), piano.to(config.DEVICE)
        with torch.no_grad():
            piano_fake = gen_F(piano)
            flute_fake = gen_P(flute)
            save_image(flute, folder + f"/Val_flute_{epoch}_{idx}.png")
            save_image(piano, folder + f"/Val_piano_{epoch}_{idx}.png")
            save_image(flute_fake, folder + f"/Val_fake_flute{epoch}_{idx}.png")
            save_image(piano_fake, folder + f"/Val_fake_piano{epoch}_{idx}.png")

            # Adversarial Loss for both generators
            D_P_fake = disc_P(piano_fake)
            D_F_fake = disc_F(flute_fake)
            loss_G_F = mse(D_F_fake, torch.ones_like(D_F_fake))
            loss_G_P = mse(D_P_fake, torch.ones_like(D_P_fake))
            loss_piano.update(loss_G_P, 1)
            loss_flute.update(loss_G_F, 1)

            # Cycle Loss
            cycle_piano = gen_P(flute_fake)
            cycle_flute = gen_F(piano_fake)
            cycle_piano_loss = L1(piano, cycle_piano)
            cycle_flute_loss = L1(flute, cycle_flute)
            loss_cyc_piano.update(cycle_piano_loss, 1)
            loss_cyc_flute.update(cycle_flute_loss, 1)

            # Identity Loss
            identity_flute = gen_F(flute)
            identity_piano = gen_P(piano)
            identity_piano_loss = L1(piano, identity_piano)
            identity_flute_loss = L1(flute, identity_flute)
            loss_id_piano.update(identity_piano_loss, 1)
            loss_id_flute.update(identity_flute_loss, 1)

            # Overall Generator Loss
            G_loss =  (
                loss_G_F +
                loss_G_P +
                cycle_flute_loss * config.LAMBDA_CYCLE +
                cycle_piano_loss * config.LAMBDA_CYCLE +
                identity_flute_loss * config.LAMBDA_IDENTITY +
                identity_piano_loss * config.LAMBDA_IDENTITY
            )

            loss.update(G_loss, 1)
            progress.display(e)

    gen_F.train(), gen_P.train(), disc_F.train(), disc_P.train()