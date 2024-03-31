#!/usr/bin/env python
# coding: utf-8
import sys
import torch
import tqdm
import time
import warnings
import os.path as osp
import torch.nn as nn
from torch import optim
from termcolor import colored


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from in_out.arguments import parse_arguments
from in_out.neural_net_oriented import load_scan_related_data, load_referential_data
from in_out.neural_net_oriented import compute_auxiliary_data, trim_scans_per_referit3d_data
from in_out.pt_datasets.listening_dataset import make_data_loaders
from utils import set_gpu_to_zero_position, create_logger, seed_training_code
from utils.tf_visualizer import Visualizer
from models.MiKASA_transformer import MiKASA_transformer
from models.MiKASA_transformer_utils import single_epoch_train, evaluate_on_dataset
from utils.models_utils import load_state_dicts, save_state_dicts
from analysis.deepnet_predictions import analyze_predictions
from transformers import BertTokenizer, BertModel


def log_train_test_information():
        """Helper logging function.
        Note uses "global" variables defined below.
        """
        logger.info('Epoch:{}'.format(epoch))
        for phase in ['train', 'test']:
            if phase == 'train':
                meters = train_meters
            else:
                meters = test_meters
            info = '{}: Total-Loss {:.4f}, Listening-Acc {:.4f}'.format(phase, meters[phase + '_total_loss'], meters[phase + '_referential_acc'])
            if config.obj_cls_alpha > 0:
                info += ', Object-Clf-Acc: {:.4f}'.format(meters[phase + '_object_cls_acc'])
                info += ', Object-Clf-Acc-Post: {:.4f}'.format(meters[phase + '_post_object_cls_acc'])

            if config.lang_cls_alpha > 0:
                info += ', Text-Clf-Acc: {:.4f}'.format(meters[phase + '_txt_cls_acc'])            
            logger.info(info)            
            logger.info('{}: Epoch-time {:.3f}'.format(phase, timings[phase]))
        logger.info('Best so far {:.3f} (@epoch {})'.format(best_test_acc, best_test_epoch))
        logger.info('--------------------------------------------------------------')
        
if __name__ == '__main__':
    
    # Parse arguments
    args, config = parse_arguments()
    # Read the scan related information
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(config.scannet_file)
    # Read the linguistic data of ReferIt3D
    referit_data = load_referential_data(config, config.referit3D_file, scans_split)
    # Prepare data & compute auxiliary meta-information.
    all_scans_in_dict = trim_scans_per_referit3d_data(referit_data, all_scans_in_dict)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict, config)
    data_loaders = make_data_loaders(config, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb, mode=args.mode)
    # Prepare GPU environment
    set_gpu_to_zero_position(config.gpu)  # Pnet++ seems to work only at "gpu:0"

    device = torch.device('cuda')
    seed_training_code(config.random_seed)

    # Losses:
    criteria = dict()
    # Prepare the Listener
    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class
    pad_idx = class_to_idx['pad']
    tokenizer = BertTokenizer.from_pretrained(config.bert_pretrain_path)
    model = MiKASA_transformer(config, n_classes, ignore_index=pad_idx)    
    model = model.to(device)
    print(model)
    
    param_list=[
            {'params':model.language_encoder.parameters(),'lr':config.init_lr*0.1},
            {'params':model.object_encoder.parameters(), 'lr':config.init_lr},
            {'params':model.post_obj_enc.parameters(), 'lr':config.init_lr*0.1},
            {'params':model.obj_feature_mapping.parameters(), 'lr': config.init_lr},
            {'params':model.box_feature_mapping.parameters(), 'lr': config.init_lr},
            {'params':model.fusion_net.parameters(), 'lr': config.init_lr*0.1},
            {'params':model.language_clf.parameters(), 'lr': config.init_lr},
            {'params':model.object_clf.parameters(), 'lr': config.init_lr},
            {'params':model.post_object_clf.parameters(), 'lr': config.init_lr},
            {'params':model.fusion_clf.parameters(), 'lr': config.init_lr},
        ]
    if config.optimizer.type == "adamw":
        optimizer = optim.AdamW(param_list,lr=config.init_lr, weight_decay=config.optimizer.weight_decay)
    else:
        optimizer = optim.Adam(param_list,lr=config.init_lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,config.optimizer.steps, gamma=config.optimizer.gamma)

    start_training_epoch = 1
    best_test_acc = -1
    best_test_epoch = -1
    last_test_acc = -1
    last_test_epoch = -1

    if args.resume_path:
        warnings.warn('Resuming assumes that the BEST per-val model is loaded!')
        # perhaps best_test_acc, best_test_epoch, best_test_epoch =  unpickle...
        loaded_epoch = load_state_dicts(args.resume_path, map_location=device, model=model)
        print('Loaded a model stopped at epoch: {}.'.format(loaded_epoch))
        if not config.fine_tune:
            print('Loaded a model that we do NOT plan to fine-tune.')
            load_state_dicts(args.resume_path, optimizer=optimizer, lr_scheduler=lr_scheduler)
            start_training_epoch = loaded_epoch + 1
            start_training_epoch = 0
            best_test_epoch = loaded_epoch
            best_test_acc = 0
            print('Loaded model had {} test-accuracy in the corresponding dataset used when trained.'.format(
                best_test_acc))
        else:
            print('Parameters that do not allow gradients to be back-propped:')
            ft_everything = True
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    print(name)
                    exist = False
            if ft_everything:
                print('None, all wil be fine-tuned')
            # if you fine-tune the previous epochs/accuracy are irrelevant.
            dummy = config.max_train_epochs + 1 - start_training_epoch
            print('Ready to *fine-tune* the model for a max of {} epochs'.format(dummy))
            
        
    # Training.
    if args.mode == 'train':
        train_vis = Visualizer(args.tensorboard_dir)
        logger = create_logger(args.log_dir)
        logger.info('Starting the training. Good luck!')
        logger.info(config)

        with tqdm.trange(start_training_epoch, config.max_train_epochs + 1, desc='epochs') as bar:
            timings = dict()
            for epoch in bar:
                print("cnt_lr", lr_scheduler.get_last_lr())
                # Train:
                tic = time.time()
                train_meters = single_epoch_train(model, data_loaders['train'], criteria, optimizer,
                                                  device, pad_idx, config=config, tokenizer=tokenizer,epoch=epoch)
                toc = time.time()
                timings['train'] = (toc - tic) / 60

                # Evaluate:
                tic = time.time()
                test_meters = evaluate_on_dataset(model, data_loaders['test'], criteria, device, pad_idx, config=config, tokenizer=tokenizer)
                toc = time.time()
                timings['test'] = (toc - tic) / 60

                eval_acc = test_meters['test_referential_acc']

                last_test_acc = eval_acc
                last_test_epoch = epoch

                lr_scheduler.step()

                save_state_dicts(osp.join(args.checkpoint_dir, 'last_model.pth'),
                                     epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)

                if best_test_acc < eval_acc:
                    logger.info(colored('Test accuracy, improved @epoch {}'.format(epoch), 'green'))
                    best_test_acc = eval_acc
                    best_test_epoch = epoch

                    save_state_dicts(osp.join(args.checkpoint_dir, 'best_model.pth'),
                                     epoch, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)
                else:
                    logger.info(colored('Test accuracy, did not improve @epoch {}'.format(epoch), 'red'))

                log_train_test_information()
                train_meters.update(test_meters)
                train_vis.log_scalars({k: v for k, v in train_meters.items() if '_acc' in k}, step=epoch,
                                      main_tag='acc')
                train_vis.log_scalars({k: v for k, v in train_meters.items() if '_loss' in k},
                                      step=epoch, main_tag='loss')

                bar.refresh()

        with open(osp.join(args.checkpoint_dir, 'final_result.txt'), 'w') as f_out:
            f_out.write(('Best accuracy: {:.4f} (@epoch {})'.format(best_test_acc, best_test_epoch)))
            f_out.write(('Last accuracy: {:.4f} (@epoch {})'.format(last_test_acc, last_test_epoch)))

        logger.info('Finished training successfully.')

    elif args.mode == 'evaluate':

        out_file = osp.join(args.checkpoint_dir, 'test_result.txt')
        res = analyze_predictions(model, data_loaders['test'].dataset, class_to_idx, pad_idx, device,
                                  config, out_file=out_file,tokenizer=tokenizer)
        print(res)