import os
import argparse
import scipy.io as sio
import dataloader
from scipy.interpolate import interpn
from medipy.metrics import dice
from model_vm import UNet
from losses import *

class Tester(object):
    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self._build_graph()

    def _build_graph(self):
        atlas = np.load('data/atlas_norm.npz')
        img1 = atlas['vol']
        img1_seg = atlas['seg']
        img1 = np.reshape(img1, img1.shape + (1,))

        img2, img2_seg = dataloader.load_nii_by_name('data/test_vol.nii.gz', 'data/test_seg.nii.gz')

        self.img1_seg = img1_seg # shape(h, w, c)
        self.img2_seg = img2_seg
        self.vol_size = img1_seg.shape
        self.images = np.array([img1, img2]) # shape(2, h, w, c, 1)
        self.images_tf = tf.expand_dims(tf.convert_to_tensor(self.images, dtype=tf.float32), axis=0) # shape(1, 2, h, w, c, 1)

        self.model = UNet(name='unet')
        self.flow, self.y = self.model(self.images_tf[:, 0], self.images_tf[:, 1])

        self.saver = tf.train.Saver()
        if self.args.resume is not None:
            print(f'Loading learned model from checkpoint {self.args.resume}')
            self.saver.restore(self.sess, self.args.resume)
        else:
            print('!!! Test with un-learned model !!!')
            self.sess.run(tf.global_variables_initializer())

    def test(self):

        # get Dice
        flow_final = self.sess.run(self.flow)

        labels = sio.loadmat('data/labels.mat')['labels'][0]  # Anatomical labels we want to evaluate
        vals, _ = dice(self.img2_seg, self.img1_seg, labels=labels, nargout=2)
        print(np.mean(vals))

        # Warp segments with flow
        flow = np.zeros([1, self.vol_size[0],self.vol_size[1], self.vol_size[2], 3])
        flow[0, :, :, :, 1] = flow_final[0, :, :, :, 0]
        flow[0, :, :, :, 0] = flow_final[0, :, :, :, 1]
        flow[0, :, :, :, 2] = flow_final[0, :, :, :, 2]

        xx = np.arange(self.vol_size[1])
        yy = np.arange(self.vol_size[0])
        zz = np.arange(self.vol_size[2])
        grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)
        sample = flow[0, :, :, :, :] + grid
        sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
        warp_seg = interpn((yy, xx, zz), self.img2_seg, sample, method='nearest', bounds_error=False,
                           fill_value=0)
        val, _ = dice(warp_seg, self.img1_seg, labels=labels, nargout=2)
        print(np.mean(val))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None,
                        help='Learned parameter checkpoint file [None]')

    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    tester = Tester(args)
    tester.test()
