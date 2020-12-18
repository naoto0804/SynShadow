from models import create_model
from models.networks.base.sync_batchnorm import DataParallelWithCallback


class GANTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.model = create_model(opt)
        if len(opt.gpu_ids) > 0:
            self.model = DataParallelWithCallback(
                self.model, device_ids=opt.gpu_ids)
            self.model_on_one_gpu = self.model.module
        else:
            self.model_on_one_gpu = self.model

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def get_latest_losses(self):
        if self.opt.no_adv:
            return {**self.g_losses}
        else:
            return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    # def update_learning_rate(self, epoch):
    #     self.update_learning_rate(epoch)

    def save(self, epoch):
        self.model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################
    def update_learning_rate_per_epoch(self, epoch):
        if self.opt.lr_decay_policy == "pix2pix":
            if epoch > self.opt.nepoch:
                lrd = self.opt.lr / self.opt.nepoch_decay
                new_lr = self.old_lr - lrd
            else:
                new_lr = self.old_lr
        else:
            raise NotImplementedError

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            if not self.opt.no_adv:
                for param_group in self.optimizer_D.param_groups:
                    param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr

    def update_learning_rate_per_iter(self, iter):
        decay_rate = 0.9
        if self.opt.lr_decay_policy == "poly":
            if iter == self.opt.niter:
                return
            mult = (1 - iter / self.opt.niter) ** decay_rate
            new_lr = self.opt.lr * mult
            self.optimizer_G.param_groups[0]['lr'] = 2 * new_lr
            self.optimizer_G.param_groups[1]['lr'] = new_lr
            # print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
