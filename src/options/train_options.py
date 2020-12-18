from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=1000,
                            help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000,
                            help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=100,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true', default=True,
                            help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--tf_log', action='store_true', default=True,
                            help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for training
        parser.add_argument('--continue_train', action='store_true',
                            help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--nepoch', type=int, default=100,
                            help='# of epoch at starting learning rate. This is NOT the total #epochs. Totla #epochs is nepoch + nepoch_decay')
        parser.add_argument('--nepoch_decay', type=int, default=0,
                            help='# of epoch to linearly decay learning rate to zero')
        parser.add_argument('--niter', type=int, default=0,
                            help="only used for poly learning schedule. If this is specified, nepoch is automatically set.")
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float,
                            default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float,
                            default=0.9, help='momentum term of adam')
        parser.add_argument('--no_TTUR', default=True,
                            action='store_true', help='Use TTUR training scheme')
        parser.add_argument('--no_adv', action='store_true',
                            help='no adversarial learning')

        # the default values for beta1 and beta2 differ by TTUR option
        opt, _ = parser.parse_known_args()
        if opt.no_TTUR:
            parser.set_defaults(beta1=0.5, beta2=0.999)

        parser.add_argument('--lr', type=float, default=0.0002,
                            help='initial learning rate for adam')
        parser.add_argument('--lr_decay_policy', type=str, default='pix2pix',
                            choices=["pix2pix", "poly"])
        parser.add_argument('--G_steps_per_D', type=int, default=1,
                            help='number of generator iterations per discriminator iterations.')

        # for discriminators
        parser.add_argument('--ndf', type=int, default=64,
                            help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float,
                            default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_vgg', type=float,
                            default=10.0, help='weight for vgg loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true',
                            help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true',
                            help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--gan_mode', type=str,
                            default='hinge', help='(ls|original|hinge)')
        parser.add_argument(
            '--netD', type=str, default='nlayer', help='(nlayer|multiscale|image|global|dhan)')
        # parser.add_argument('--lambda_kld', type=float, default=0.05)
        parser.add_argument('--finetune_from', type=str, default=None)

        self.isTrain = True
        return parser
