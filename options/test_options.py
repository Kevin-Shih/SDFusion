from .base_options import BaseOptions

class TestOptions(BaseOptions):

    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = False

        self.phase = 'test'

class TestMyDataOptions(BaseOptions):
    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed

    def init_model_args(
            self,
            ckpt_path='saved_ckpt/sdfusion-img2shape.pth',
            vq_ckpt_path='saved_ckpt/vqvae-snet-all.pth',
        ):
        import os
        import sys
        from utils import util
        if not self.initialized:
            self.initialize()
        
        cmd = ' '.join(sys.argv)
        print(f'python {cmd}')
        
        self.opt = self.parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{self.opt.gpu_ids}"

        self.opt.isTrain = False
        self.opt.phase = 'test'
        self.opt.device = 'cuda'
        self.opt.name = 'SDFusionImage2ShapeOption'
        self.opt.gpu_ids_str = str(self.opt.gpu_ids)
        self.opt.rank = 0
        self.opt.nThreads = 4
        self.opt.distributed = False
        self.opt.max_dataset_size = 10000000
        self.opt.ddim_eta = 0.15
        self.opt.uc_scale = 2.
        if self.seed is not None:
            util.seed_everything(self.seed)

        self.opt.model = 'sdfusion-img2shape'
        self.opt.df_cfg = 'configs/sdfusion-img2shape.yaml'
        self.opt.ckpt = ckpt_path
        self.opt.vq_ckpt = vq_ckpt_path
        return self.opt