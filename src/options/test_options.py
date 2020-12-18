from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='../results')
        parser.add_argument('--which_epoch', type=str, default='latest',
                            help='which epoch to load?')

        parser.set_defaults(phase='test', no_flip=True,
                            serial_batches=True, batch_size=1)
        self.isTrain = False
        return parser
