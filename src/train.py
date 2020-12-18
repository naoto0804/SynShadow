import sys
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.gan_trainer import GANTrainer

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)

# create trainer for our model
trainer = GANTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        trainer.run_generator_one_step(data_i)

        # train discriminator
        if not opt.no_adv and i % opt.G_steps_per_D == 0:
            trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(
                epoch, iter_counter.epoch_iter,
                losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(
                losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            generated = trainer.get_latest_generated()

            visual_list = [
                ('input', data_i['input']), ('target_attn', data_i['mask']),
                ('target_img', data_i['target'])
            ]
            visual_list = trainer.model.module.add_visualization(
                    data_i, generated, visual_list)
            visuals = OrderedDict(visual_list)
            visualizer.display_current_results(
                visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

        if opt.model == 'detect':
            current_iter = iter_counter.total_steps_so_far // opt.batch_size
            trainer.update_learning_rate_per_iter(current_iter)

    if opt.model != 'detect':
        trainer.update_learning_rate_per_epoch(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')
