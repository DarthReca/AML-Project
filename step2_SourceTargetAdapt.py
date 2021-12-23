
import torch
from torch import nn
from optimizer_helper import get_optim_and_scheduler
from itertools import cycle
import numpy as np


#### Implement Step2
"""
    Method is to be completed. It's supposed to predict orientation and class of each image, 
    along with class and orientation loss and then do the final evaluation computing:
    OS: accuracy in recognizing known category
    UNK: accuracy in recognizing unknown category
    HOS: harmonic (~= average ) between the 2

    Parameters
    ---------
        args : data-type -> {epochs_step2 : int, weight_RotTask_step2 : float}
            weight_RotTask_step2 is the alpha2 parameter (go to page 6 of project description)
        feature_extractor
            only used for calling it's train() and eval() methods
        rot_cls
            stands for rotation classifier. only used for calling it's train() and eval() methods
        obj_cls
            stands for rotation classifier. only used for calling it's train() and eval() methods
        source_loader : list<data-type> -> {data_source, class_l_source, _, _}
            contains all the images of the source (data_source) and their assigned class (class_l_source)
        target_loader_train: list<data-type> -> {data_target, _, data_target_rot, rot_l_target}
            contains the target image, the rotated target image and (?) the loss of the rotated target image (?)
        target_loader_eval : list<data-type> -> {data_source, class_l_source, _, _}
            for each image contains: image, associated class
        optimizer
            only used for calling it's zero_grad() and step() methods
        device
            CPU or GPU ?
    Returns
    -------
        none
"""
def _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,optimizer,device):

    #train the 2 classifiers and the feature extractor
    criterion = nn.CrossEntropyLoss()
    feature_extractor.train()
    obj_cls.train()
    rot_cls.train()

    #make target_loader_train iterable
    target_loader_train = cycle(target_loader_train)

    #loop over source_loader AND target_loader_train at the same time
    for it, (data_source, class_l_source, _, _) in enumerate(source_loader):
        (data_target, _, data_target_rot, rot_l_target) = next(target_loader_train)

        #move tensors to GPU
        data_source, class_l_source  = data_source.to(device), class_l_source.to(device)
        data_target, data_target_rot, rot_l_target  = data_target.to(device), data_target_rot.to(device), rot_l_target.to(device)

        #set gradients of optimized tensors to 0
        optimizer.zero_grad()

        #TODO: compute the loss of the class and the rotation classification tasks
        class_loss = ....
        rot_loss = ....

        loss = class_loss + args.weight_RotTask_step2*rot_loss

        #backpropagation
        loss.backward()

        #update network's parameters
        optimizer.step()

        #TODO: store predicted class and rotation#TODO: compute the loss of the class and the rotation classification tasks
        _, cls_pred = ...
        _, rot_pred = ...

    #TODO: compute accuracy on class and rotation predictions
    acc_cls = ...
    acc_rot = ...

    print("Class Loss %.4f, Class Accuracy %.4f,Rot Loss %.4f, Rot Accuracy %.4f" % (class_loss.item(), acc_cls, rot_loss.item(), acc_rot))


    #### Implement the final evaluation step, computing OS*, UNK and HOS
    feature_extractor.eval()
    obj_cls.eval()
    rot_cls.eval()

    #deactivate autograd engine for speedup during evaluation phase
    with torch.no_grad():
        for it, (data, class_l,_,_) in enumerate(target_loader_eval):


"""
    Method retrieves the optimizer and the scheduler 
    and then iteratively calls _do_epoch() and scheduler.step() methods for each epoch

    Parameters
    ---------
        args : data-type -> {epochs_step2 : int}
            epoch_step2 is the number of epochs to do in step2
        feature_extractor
            only passed through
        rot_cls
            only passed through. stands for rotation classifier. 
        obj_cls
            only passed through. stands for object classifier
        source_loader 
            only passed through
        target_loader_train
            only passed through
        target_loader_eval : list<data-type> -> {data, class_l, data_rot, rot_l}
            only passed through. for each image contains: image, loss for the image, rotated image, loss for the rotated image (?)        
        device
            only passed through. CPU or GPU ?
    Returns
    -------
        none
"""
def step2(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,device):
    optimizer, scheduler = get_optim_and_scheduler(feature_extractor,rot_cls,obj_cls, args.epochs_step2, args.learning_rate, args.train_all)


    for epoch in range(args.epochs_step2):
        _do_epoch(args,feature_extractor,rot_cls,obj_cls,source_loader,target_loader_train,target_loader_eval,optimizer,device)
        scheduler.step()