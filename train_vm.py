import argparse
import time
from model_vm import UNet
from Dataset import *
from losses import *
from utils import show_progress


class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        self._build_graph()

    def _build_graph(self):
        shared_keys = ['dataset_dir', 'batch_size', 'num_parallel_calls']
        shared_args = {}
        for key in shared_keys:
            shared_args[key] = getattr(self.args, key)

        # Input data
        with tf.name_scope('Data'):
            tset = BaseDataset_3D(train_or_val='train', **shared_args)
            self.images = tset.get_element()
            self.image0, self.image1 = self.images[:, 0], self.images[:, 1]

            vset = BaseDataset_3D(train_or_val='val', **shared_args)
            self.images_v, self.initializer_v = vset.get_element()
            self.image0_v, self.image1_v = self.images_v[:, 0], self.images_v[:, 1]

            self.num_batches = len(tset.samples)//self.args.batch_size
            self.num_batches_v = len(vset.samples)//self.args.batch_size

        # Model inference
        model = UNet(name='unet')

        self.flow, self.y = model(self.image0, self.image1) # image0 image1 (fixed and moving)
        self.flow_v, self.y_v = model(self.image0_v, self.image1_v, reuse=True)

        # Loss calculation
        with tf.name_scope('Loss'):

            weights_l2 = Grad(self.flow)
            weights_l2_v = Grad(self.flow_v)

            # self.epe = mse(self.image0, self.y)
            # self.epe_v = mse(self.image0_v, self.y_v)
            self.epe = ncc(self.image0, self.y)
            self.epe_v = ncc(self.image0_v, self.y_v)

            self.loss = self.epe + self.args.gamma * weights_l2
            self.loss_v = self.epe_v + self.args.gamma * weights_l2_v

            self.weights_l2 = weights_l2



        # Gradient descent optimization
        with tf.name_scope('Optimize'):
            self.global_step = tf.train.get_or_create_global_step()
            lr = self.args.lr
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss, var_list=model.vars)
            with tf.control_dependencies([self.optimizer]):
                self.optimizer = tf.assign_add(self.global_step, 1)

        # Load learned model
        self.saver = tf.train.Saver(model.vars, max_to_keep=50)
        self.sess.run(tf.global_variables_initializer())
        if self.args.resume is not None:
            print(f'Loading learned model from checkpoint {self.args.resume}')
            self.saver.restore(self.sess, self.args.resume)

        tf.summary.FileWriter('./logs', graph=self.sess.graph)
                    
    def train(self):
        train_start = time.time()
        for e in range(self.args.num_epochs):
            for i in range(self.num_batches):
                time_s = time.time()
                _, loss, epe, reg_loss = self.sess.run([self.optimizer, self.loss, self.epe, self.weights_l2])

                if i%20 == 0:
                    batch_time = time.time() - time_s
                    kwargs = {'loss':loss, 'reg_loss':reg_loss, 'epe':epe, 'batch time':batch_time}
                    show_progress(e+1, i+1, self.num_batches, **kwargs)

            loss_vals, epe_vals, reg_vals = [], [], []
            self.sess.run([self.initializer_v])
            for i in range(self.num_batches_v):
                image0_v, image1_v, flows_val, loss_val, epe_val, reg_val \
                  = self.sess.run([self.image0_v, self.image1_v, self.flow_v,
                                   self.loss_v, self.epe_v, self.weights_l2])
                loss_vals.append(loss_val)
                epe_vals.append(epe_val)
                reg_vals.append(reg_val)
                
            g_step = self.sess.run(self.global_step)
            print(f'\r{e+1} epoch validation, loss: {np.mean(loss_vals)}, reg_loss:{np.mean(reg_vals)}, epe: {np.mean(epe_vals)}'\
                  + f', global step: {g_step}, elapsed time: {time.time()-train_start} sec.')



            if not os.path.exists('./model'):
                os.mkdir('./model')
            self.saver.save(self.sess, f'./model/model_{e+1}.ckpt')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dd', '--dataset_dir', type=str, required=True,
                        help = 'Directory containing target dataset')
    parser.add_argument('-e', '--num_epochs', type=int, default=100,
                        help = '# of epochs [100]')
    parser.add_argument('-b', '--batch_size', type=int, default=1,
                        help = 'Batch size [4]')
    parser.add_argument('-nc', '--num_parallel_calls', type=int, default=1,
                        help = '# of parallel calls for data loading [2]')

    parser.add_argument('--lr', type = float, default = 1e-4,
                        help = 'Learning rate [1e-4]')
    parser.add_argument('--gamma', type = float, default = 1.5,
                        help = 'Coefficient for weight decay [4e-4]')
    parser.add_argument('-r', '--resume', type = str,
                        default = None,
                        help = 'Learned parameter checkpoint file [None]')

    args = parser.parse_args()
    for key, item in vars(args).items():
        print(f'{key} : {item}')

    # os.environ['CUDA_VISIBLE_DEVICES'] = input('Input utilize gpu-id (-1:cpu) : ')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    trainer = Trainer(args)
    trainer.train()
