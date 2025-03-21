import numpy as np
import torch
from .base_model import BaseModel
from . import networks
import os
from .rcf_model import RCF
from util.image_pool import ImagePool

class GLCTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_rec', type=float, default=2.5, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_idt', type=float, default=5.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_local', type=float, default=5.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_grad', type=float, default=0.1, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--grad_layers', type=str, default='0,3,6,10,14', help='compute NCE loss on which layers')
        parser.add_argument('--style_dim', type=int, default=8, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--grad_interval_min', type=float, default=0.05, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--grad_interval_max', type=float, default=0.10, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--noise_std', type=float, default=1.0, help='compute NCE loss on which layers')
        parser.add_argument('--tag', type=str, default='debug', help='compute NCE loss on which layers')
        parser.set_defaults(no_html=True, pool_size=0)  # no image pooling
        opt, _ = parser.parse_known_args()
        if opt.phase != 'test':
            model_id = '%s' % opt.tag
            model_id += '/' + os.grad.basename(opt.dataroot.strip('/')) + '_%s' % opt.direction
            model_id += '/lam%s_layers%s_dim%d_rec%d_idt%s_pool%d_noise%s_kl%s' % \
                        (opt.lambda_grad, opt.grad_layers, opt.style_dim, opt.lambda_rec, opt.lambda_idt, opt.pool_size,
                         opt.noise_std, opt.lambda_kl)
            parser.set_defaults(name=model_id)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G_rec', 'G_idt', 'G_kl', 'G_grad', 'd1', 'd2']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.grad_layers = [int(i) for i in self.opt.grad_layers.split(',')]
        for l in self.grad_layers:
            self.loss_names += ['energy_%d' % l]

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)

        self.netM = TEMPModel(opt, self.device).to(self.device)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids,
                                      opt)
        # 固定的rcf
        self.rcf_net = RCF().cuda()
        self.rcf_net.load_state_dict(torch.load("./pth/bsds500_pascal_model.pth"))
        self.rcf_net.eval() 
        self.d_A = torch.zeros([1]).to(self.device)
        self.d_B = torch.ones([1]).to(self.device)

        # print(self.netG)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type,
                                          opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # print(self.netD)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.per_loss_func = torch.nn.MSELoss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netM.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def optimize_parameters(self):
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss() + self.compute_rcf_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_grads = input['A_grads' if AtoB else 'B_grads']

    def forward(self):
        self.mu, self.latent_A, self.latent_B, self.rec_A, self.fake_B, self.idt_B, \
        self.real_A_rcf, self.fake_B_rcf, self.netG = self.netM(self.real_A, self.real_B)
        with torch.no_grad():
            self.real_A_rcf_fix = self.rcf_net(self.real_A)[0]

        xy = torch.cat([self.real_A, self.rec_A], dim=0)
        latents_xy = self.netG(xy, mode='encode')
        if self.isTrain and self.opt.noise_std > 0:
            noise = torch.ones_like(latents_xy).normal_(mean=0, std=self.opt.noise_std)
            latents_xy = latents_xy + noise
        self.latent_x, self.latent_y = latents_xy.chunk(2, dim=0)
        ds_xy = torch.cat([self.d_A, self.d_B, self.d_B], 0).unsqueeze(-1)
        latents_xy = torch.cat([self.latent_x, self.latent_x, self.latent_y], 0)
        with torch.no_grad():
            images_xy = self.netG((latents_xy, ds_xy), mode='decode')
            self.x_t, self.y_t, self.xy_t = images_xy.chunk(3, dim=0)


    @torch.no_grad()
    def single_forward(self):
        latent = self.netG(self.real_A, mode='encode')
        out = self.netG((latent, self.d_B), mode='decode')

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B_pool.query(self.fake_B.detach())
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True).mean()
        # self.loss_G_global = self.criterionIdt(self.rec_A, self.real_A).mean()
        self.loss_G_global = self.criterionIdt(self.real_A, self.x_t).mean() + self.criterionIdt(self.rec_A, self.y_t).mean()

        # loss_G_global = a * Lper(x) + b * Lper(y)
        self.loss_G_idt = self.criterionIdt(self.idt_B, self.real_B).mean()
        self.loss_d1 = torch.tensor(0.)
        self.loss_d2 = torch.tensor(0.)
        for l in self.grad_layers:
            setattr(self, 'loss_energy_%d' % l, 0)
        if self.opt.noise_std > 0:
            self.loss_G_kl = torch.pow(self.mu, 2).mean()
        else:
            self.loss_G_kl = 0
        self.loss_n_dots = 0
        if self.opt.lambda_grad > 0:
            self.loss_G_grad = self.compute_grad_losses()
        else:
            self.loss_G_grad = torch.tensor(0.)
        self.loss_G = self.opt.lambda_GAN * self.loss_G_GAN + \
                      self.opt.lambda_rec * self.loss_G_global + \
                      self.opt.lambda_idt * self.loss_G_idt + \
                      self.opt.lambda_kl * self.loss_G_kl + \
                      self.opt.lambda_grad * self.loss_G_grad
        return self.loss_G

    def compute_rcf_loss(self):
        # real_A_rcf_fix = self.rcf_net(self.real_A)[0]
        # real_A_rcf = self.netR(self.real_A)[0]
        # fake_B_rcf = self.netR(self.fake_B)[0]

        self.loss_edge_l1 = self.criterionIdt(self.real_A_rcf_fix, self.real_A_rcf).mean()
        self.loss_edge_l2 = self.criterionIdt(self.real_A_rcf_fix, self.fake_B_rcf).mean()
        self.loss_R = (self.loss_edge_l1 + self.loss_edge_l2) * self.opt.lambda_local
        return self.loss_R

    def compute_grad_losses(self):
        d1_center = torch.ones(len(self.latent_A)).to(self.device).uniform_(0, 1)
        interval = torch.ones(len(self.latent_A)).to(self.device).uniform_(self.opt.grad_interval_min,
                                                                           self.opt.grad_interval_max)
        d1 = (d1_center + interval).clamp(0, 1)
        d2 = (d1_center - interval).clamp(0, 1)
        latents = torch.cat([self.latent_A, self.latent_A], 0)
        ds = torch.cat([d1, d2], 0).unsqueeze(-1)
        features = self.netG((latents, ds), layers=self.grad_layers, mode='decode_and_extract')

        # dummy for outputs only
        self.loss_d1 = torch.mean(d1)
        self.loss_d2 = torch.mean(d2)

        loss_grad = 0
        for id, feats in enumerate(features):
            x_d1, x_d2 = torch.chunk(feats, 2, dim=0)
            jacobian = (x_d1 - x_d2) / (torch.maximum(d1 - d2, torch.ones_like(d1) * 0.1))
            energy = (jacobian ** 2).mean()
            setattr(self, 'loss_energy_%d' % self.grad_layers[id], energy.item())
            loss_grad += energy
        loss_grad = loss_grad / len(features)
        return loss_grad

    @torch.no_grad()
    def compute_whole_grad_length(self):

        small_int = 0.1
        linsp = np.linspace(0, 1 - small_int, int(1 / small_int))
        losses = [0 for i in range(5)]
        for d2 in linsp:
            d1 = d2 + small_int
            d1 = torch.tensor([d1]).to(self.device).float()
            d2 = torch.tensor([d2]).to(self.device).float()
            latents = torch.cat([self.latent_A, self.latent_A], 0)
            ds = torch.cat([d1, d2], 0).unsqueeze(-1)
            features = self.netG((latents, ds), layers=self.grad_layers, mode='decode_and_extract')
            loss_grad = 0
            for id, feats in enumerate(features):
                x_d1, x_d2 = torch.chunk(feats, 2, dim=0)
                jacobian = (x_d1 - x_d2) / (torch.maximum(d1 - d2, torch.ones_like(d1) * 0.1))
                energy = (jacobian ** 2).mean()
                loss_grad += energy
                losses[id] += energy.item() / 10
        return losses

    @torch.no_grad()
    def interpolation(self, x_a, x_b):
        self.netG.eval()
        if self.opt.direction == 'AtoB':
            x = x_a
        else:
            x = x_b
        interps = []
        for i in range(min(x.size(0), 8)):
            h_a = self.netG(x[i].unsqueeze(0), mode='encode')
            d = 0.2
            local_interps = []
            # local_interps.append(x[i].unsqueeze(0))

            while d < 1.:
                d_t = torch.tensor([d]).to(x_a.device).unsqueeze(-1)
                local_interps.append(self.netG((h_a, d_t), mode='decode'))
                d += 0.2
            local_interps = torch.cat(local_interps, 0)
            interps.append(local_interps)
        self.netG.train()
        return interps

    @torch.no_grad()
    def translate(self, x):
        self.netG.eval()
        h = self.netG(x, mode='encode')
        out = self.netG((h, self.d_B), mode='decode')
        self.netG.train()
        return out

    @torch.no_grad()
    def sample(self, x_a, x_b):
        self.netG.eval()
        if self.opt.direction == 'BtoA':
            x_a, x_b = x_b, x_a
        x_a_recon, x_b_recon, x_ba, x_ab = [], [], [], []
        for i in range(x_a.size(0)):
            h_a = self.netG(x_a[i].unsqueeze(0), mode='encode')
            h_b = self.netG(x_b[i].unsqueeze(0), mode='encode')
            x_a_recon.append(self.netG((h_a, self.d_A), mode='decode'))
            x_b_recon.append(self.netG((h_b, self.d_B), mode='decode'))
            x_ab.append(self.netG((h_a, self.d_B), mode='decode'))
        x_a_recon, x_b_recon = torch.cat(x_a_recon), torch.cat(x_b_recon)
        x_ab = torch.cat(x_ab)
        self.netG.train()
        return x_a, x_a_recon, x_ab, x_b, x_b_recon


class TEMPModel(torch.nn.Module):
    def __init__(self, opt, device):
        super(TEMPModel, self).__init__()
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
                                      opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, opt.gpu_ids,
                                      opt)
        self.netR = RCF().cuda()
        self.opt = opt
        self.d_A = torch.zeros([1]).to(device)
        self.d_B = torch.ones([1]).to(device)

    def forward(self, x, y):
        self.real_A = x
        self.real_B = y
        real = torch.cat([self.real_A, self.real_B], dim=0)
        latents = self.netG(real, mode='encode')
        noise = torch.ones_like(latents).normal_(mean=0, std=self.opt.noise_std)
        mu = latents
        latents = latents + noise
        latent_A, latent_B = latents.chunk(2, dim=0)
        ds = torch.cat([self.d_A, self.d_B, self.d_B], 0).unsqueeze(-1)
        latents = torch.cat([latent_A, latent_A, latent_B], 0)
        images = self.netG((latents, ds), mode='decode')
        rec_A, fake_B, idt_B = images.chunk(3, dim=0)  # self.rec_A 重建结果A

        # RCF
        real_A_rcf = self.netR(self.real_A)[0]
        fake_B_rcf = self.netR(fake_B)[0]

        return mu, latent_A, latent_B, rec_A, fake_B, idt_B, real_A_rcf, fake_B_rcf, self.netG

