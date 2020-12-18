import time
from io import BytesIO
from pathlib import Path

import scipy

from . import util, webhtml


class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name

        root = Path(opt.checkpoints_dir)
        if self.tf_log:
            import tensorflow as tf
            self.tf = tf
            self.log_dir = root / opt.name / 'logs'
            self.writer = tf.summary.FileWriter(str(self.log_dir))

        if self.use_html:
            self.web_dir = root / opt.name / 'web'
            self.img_dir = self.web_dir / 'images'
            print('create web directory %s...' % str(self.web_dir))
            self.web_dir.mkdir(parents=True, exist_ok=True)
            self.img_dir.mkdir(parents=True, exist_ok=True)

        if opt.isTrain:
            self.log_name = root / opt.name / 'loss_log.txt'
            with self.log_name.open(mode='a') as log_file:
                now = time.strftime("%c")
                log_file.write('====== Training Loss (%s) ====== \n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step):

        # convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)

        if self.tf_log:  # show images in tensorboard output
            img_summaries = []
            for label, image_numpy in visuals.items():
                # Write the image to a string
                s = BytesIO()
                if len(image_numpy.shape) >= 4:
                    image_numpy = image_numpy[0]
                image_numpy = image_numpy.squeeze()  # for grayscale
                scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                # Create an Image object
                img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(
                ), height=image_numpy.shape[0], width=image_numpy.shape[1])
                # Create a Summary value
                img_summaries.append(
                    self.tf.Summary.Value(tag=label, image=img_sum))

            # Create and write Summary
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

        if self.use_html:  # save images to a html file
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = \
                            self.img_dir / 'epoch%.3d_iter%.3d_%s_%d.png' % (epoch, step, label, i)
                        util.save_image(image_numpy[i], str(img_path))
                else:
                    img_path = \
                        self.img_dir / 'epoch%.3d_iter%.3d_%s.png' % (epoch, step, label)
                    if len(image_numpy.shape) >= 4:
                        image_numpy = image_numpy[0]
                    image_numpy = image_numpy.squeeze()  # for grayscale
                    util.save_image(image_numpy, str(img_path))

            # update website
            webpage = webhtml.HTML(
                self.web_dir, 'Experiment name = %s' % self.name, refresh=5)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = 'epoch%.3d_iter%.3d_%s_%d.png' % (
                                n, step, label, i)
                            ims.append(img_path)
                            txts.append(label+str(i))
                            links.append(img_path)
                    else:
                        img_path = 'epoch%.3d_iter%.3d_%s.png' % (
                            n, step, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims)/2.0))
                    webpage.add_images(
                        ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(
                        ims[num:], txts[num:], links[num:], width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                value = value.mean().float()
                summary = self.tf.Summary(
                    value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            # print(v)
            # if v != 0:
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            tile = self.opt.batch_size > 8
            t = util.tensor2im(t, normalize=False, tile=tile)
            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        visuals = self.convert_visuals_to_numpy(visuals)

        image_dir = webpage.get_image_dir()
        name = Path(image_path[0]).with_suffix('').name

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s/%s.png' % (label, name)
            save_path = image_dir / image_name
            util.save_image(image_numpy, str(save_path), create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
