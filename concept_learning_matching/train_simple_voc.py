import os
import torch
import torch.optim as optim
import time
from datetime import datetime
from pathlib import Path
# from concept_learning_matching.concept_ds import PerceptPairsDataset
# from concept_learning_matching.concept_ds_synthetic_matches import PerceptPairsSynDataset
from concept_learning_matching.concept_ds_mixmatch import PerceptPairsDatasetMix
from concept_learning_matching.concept_ds_mixt import PerceptPairsDatasetMix2
from torch.utils.tensorboard import SummaryWriter
from concept_learning_matching.displacement_layer import Displacement

from concept_learning_matching.utils.permutation_loss import CrossEntropyLoss
from concept_learning_matching.utils.evaluation_metric import matching_accuracy
from concept_learning_matching.utils.hungarian import hungarian
from concept_learning_matching.utils.config import cfg
from concept_learning_matching.PCA.model import Net
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Line2D

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_grad_flow(named_parameters, writer):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    fig, ax = plt.subplots()
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.0010, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()
    fig.tight_layout()
    writer.add_figure(tag=f'gradient_flow_custom_attention',
                      figure=fig)


def train_eval_model(model,
                     criterion,
                     optimizer,
                     dataloader_1,
                     tfboard_writer,
                     num_epochs=150,
                     resume=False,
                     total_nr=0,
                     start_epoch=0):
    print('Start training...')

    since = time.time()
    dataset_size = len(dataloader_1)
    lap_solver = hungarian


    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
    #                                            milestones=cfg.TRAIN.LR_STEP,
    #                                            gamma=cfg.TRAIN.LR_DECAY,
    #                                            last_epoch=cfg.TRAIN.START_EPOCH - 1)

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()  # Set model to training mode

        print('lr = ' + ', '.join(['{:.2e}'.format(x['lr']) for x in optimizer.param_groups]))

        epoch_loss = 0.0
        epoch_acc = 0.0
        running_loss = 0.0
        running_since = time.time()
        iter_num = 0

        # Iterate over data.
        for counter, inputs in enumerate(dataloader_1):
            emb_src = inputs.x1.to(device)
            emb_tgt = inputs.x2.to(device)
            perm_mat = inputs.train_y.unsqueeze(0).to(device)
            a_src = inputs.edge_attr1.to(device)
            a_tgt = inputs.edge_attr2.to(device)
            pos_matches = inputs.all_pos
            n1_gt = inputs.n1_gt.to(device)
            n2_gt = inputs.n2_gt.to(device)
            edge_index_1 = inputs.edge_index1.to(device)
            edge_index_2 = inputs.edge_index2.to(device)

            iter_num = iter_num + 1

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # forward
                s_pred = model(emb_src, emb_tgt, a_src, a_tgt, edge_index_1, edge_index_2)

                multi_loss = []
                loss = criterion(s_pred, perm_mat, pos_matches)

                # backward + optimize
                try:
                    loss.backward()
                except:
                    print('e')
                # print(
                #     [(m[0], m[1].grad.min().item(), m[1].grad.max().item()) for m in list(model.named_parameters())
                #      if
                #      m[1].grad is not None])
                total_nr+=1
                acc, _, __ = matching_accuracy(lap_solver(s_pred, n1_gt, n2_gt), perm_mat, n1_gt)
                if not len(pos_matches) == 0:
                    if torch.isnan(acc):
                        print(acc)
                    else:
                        epoch_acc = epoch_acc + acc
                if counter % 5000 == 0:
                    plot_grad_flow(model.named_parameters(), tfboard_writer)
                epoch_loss += loss.item() * perm_mat.size(0)

                if counter % 1000 == 0 and counter !=0:
                    print(epoch_loss / (counter + 1))
                    print(epoch_acc / (counter + 1))
                    tfboard_writer.add_scalar('Avg/acc', epoch_acc / (counter + 1), total_nr)
                    tfboard_writer.add_scalar('Avg/loss', epoch_loss / (counter + 1), total_nr)
                    print(
                        [(m[0], m[1].grad.min().item(), m[1].grad.max().item()) for m in list(model.named_parameters())
                         if
                         m[1].grad is not None])

                optimizer.step()

                # tfboard writer
                loss_dict = {'loss_{}'.format(i): l.item() for i, l in enumerate(multi_loss)}
                loss_dict['loss'] = loss.item()
                tfboard_writer.add_scalar('Iter/loss', loss.item(), total_nr)
                # tfboard_writer.add_scalar('Iter/acc', acc, total_nr)

                # statistics
                running_loss += loss.item() * perm_mat.size(0)

                if iter_num % cfg.STATISTIC_STEP == 0:
                    running_speed = cfg.STATISTIC_STEP * perm_mat.size(0) / (time.time() - running_since)
                    print('Epoch {:<4} Iteration {:<4} {:>4.2f}sample/s Loss={:<8.4f}'
                          .format(epoch, iter_num, running_speed, running_loss / cfg.STATISTIC_STEP / perm_mat.size(0)))
                    tfboard_writer.add_scalars(
                        'speed',
                        {'speed': running_speed},
                        epoch * cfg.TRAIN.EPOCH_ITERS + iter_num
                    )
                    running_loss = 0.0
                    running_since = time.time()

        epoch_loss = epoch_loss / dataset_size
        if epoch % 2 == 0:
            torch.save(model.state_dict(), f'pre_temp_100temproi_pca_gm_frames_mixt_5gnn_200ep_fix_bug_adam_1.0e-3_500_{epoch}.pth')
        tfboard_writer.add_scalar('Epoch/loss', epoch_loss /dataset_size, epoch)
        tfboard_writer.add_scalar('Epoch/acc', epoch_acc / dataset_size, epoch)

        # save_model(model, str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
        # torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        print()

        # # Eval in each epoch
        # accs = eval_model(model, dataloader['test'])
        # acc_dict = {"{}".format(cls): single_acc for cls, single_acc in zip(dataloader['train'].dataset.classes, accs)}
        # acc_dict['average'] = torch.mean(accs)
        # tfboard_writer.add_scalars(
        #     'Eval acc',
        #     acc_dict,
        #     (epoch + 1) * cfg.TRAIN.EPOCH_ITERS
        # )

        # scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'
          .format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

    return model, total_nr


def main():
    import importlib

    torch.manual_seed(cfg.RANDOM_SEED)
    # ds_1 = PerceptPairsDatasetMix2(os.path.join(os.getcwd(), 'caches_ds_mixtc1'), 'correct_percepts')
    ds_2 = PerceptPairsDatasetMix(os.path.join(os.getcwd(), 'caches_ds_mixtc2'), 'correct_percepts2')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model.load_state_dict(torch.load('new_adj_pca_gm_frames_temp_2gnn_200ep_fix_bug_1.0e-3_500_14.pth',map_location=device))

    model = model.to(device)

    if cfg.TRAIN.LOSS_FUNC == 'perm':
        criterion = CrossEntropyLoss()
    else:
        raise ValueError('Unknown loss function {}'.format(cfg.TRAIN.LOSS_FUNC))

    optimizer = optim.SGD(model.parameters(), lr=1.0e-3)


    if not Path(cfg.OUTPUT_PATH).exists():
        Path(cfg.OUTPUT_PATH).mkdir(parents=True)

    now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tfboardwriter = SummaryWriter('graph/pre_temp_100temproi_pca_gm_frames_mixt_5gnn_200ep_fix_bug_adam_1.0e-3_500')

    model, counter = train_eval_model(
        model, criterion, optimizer,
        ds_2, tfboardwriter, num_epochs=200,
        resume=cfg.TRAIN.START_EPOCH != 0,
        start_epoch=cfg.TRAIN.START_EPOCH)
    # model, counter = train_eval_model(
    #     model, criterion, optimizer,
    #     ds_2, tfboardwriter, num_epochs=30,
    #     resume=cfg.TRAIN.START_EPOCH != 0,
    #     start_epoch=31,  total_nr=counter)
    torch.save(model.state_dict(), f'pre_temp_100temproi_pca_gm_frames_mixt_5gnn_200ep_fix_bug_adam_1.0e-3_500.pth')
    return


if __name__ == '__main__':
    main()



