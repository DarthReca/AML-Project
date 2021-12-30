
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
            contains the target image, the rotated target image and (?) the label of the rotated target image (?)
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

    # Correct prediction counters
    correct_classes = 0
    correct_rotations = 0
    
    #loop over source_loader AND target_loader_train at the same time
    for it, (data_source, class_label_source, _, _) in enumerate(source_loader):
        (data_target, _, data_target_rot, rot_label_target) = next(target_loader_train)

        #move tensors to GPU
        data_source, class_label_source  = data_source.to(device), class_label_source.to(device)
        data_target, data_target_rot, rot_label_target  = data_target.to(device), data_target_rot.to(device), rot_label_target.to(device)

        #set gradients of optimized tensors to 0
        optimizer.zero_grad()
        
        # Extract features
        source_features = feature_extractor(data_source)
        target_features = feature_extractor(data_target)
        target_rotated_features = feature_extractor(data_target_rot)
          
        # Pass features to classifiers
        class_scores = obj_cls(source_features)
        # Here we have to concatenate tensors as suggested by the architecture
        rotation_scores = rot_cls(torch.cat([target_features, target_rotated_features], 1))

        #compute the loss of the class and the rotation classification tasks
        class_loss = criterion(class_scores, class_label_source)
        rot_loss = criterion(rotation_scores, rot_label_target)

        loss = class_loss + args.weight_RotTask_step2*rot_loss

        #backpropagation
        loss.backward()

        #update network's parameters
        optimizer.step()

        # Find which is the index that corresponds to the highest "probability"
        class_prediction = torch.argmax(class_scores, dim=1)
        rotation_prediction = torch.argmax(rotation_scores, dim=1)
        
        # Update counters
        correct_classes += torch.sum(class_prediction == class_label_source).item()
        correct_rotations += torch.sum(rotation_prediction == rot_label_target).item()

    #compute accuracy on class and rotation predictions
    acc_cls = correct_classes / len(source_loader)
    acc_rot = correct_rotations / len(target_loader_train)

    print("Class Loss %.4f, Class Accuracy %.4f,Rot Loss %.4f, Rot Accuracy %.4f" % (class_loss.item(), acc_cls, rot_loss.item(), acc_rot))

    #### Implement the final evaluation step, computing OS*, UNK and HOS
    feature_extractor.eval()
    obj_cls.eval()
    rot_cls.eval()
    
    # Correct prediction counters
    correct_classes_known = 0
    correct_classes_unknown = 0
    total_classes_known = 0
    total_classes_unknown = 0

    #deactivate autograd engine for speedup during evaluation phase
    with torch.no_grad():
        for it, (data, class_label,_,_) in enumerate(target_loader_eval):
            
            #move tensors to GPU
            data, class_label  = data.to(device), class_label.to(device)
            
            #Extract features
            features = feature_extractor(data)
            
            # Pass features to classifiers
            class_scores = obj_cls(features)
        
            # Get predictions
            class_prediction = torch.argmax(class_scores, dim=1)
            
            # Update counters
            if (class_label == args.n_classes_tot + 1): #Class label for unknown category is ? assuming args.n_classes_tot + 1 (i.e. 66) for now
                total_classes_unknown += 1
                correct_classes_unknown += torch.sum(class_prediction == class_label).item()
            else:
                total_classes_known += 1
                correct_classes_known += torch.sum(class_prediction == class_label).item()

        #compute accuracies
        if (total_classes_unknown == 0):
            print("Error: No unknown images identitied. Probably the class label was not set correctly.")
        acc_known = correct_classes_known / total_classes_known # OS*
        acc_unknown = correct_classes_unknown / total_classes_unknown # UNK
        hos = 2 * acc_known * acc_unknown / (acc_known + acc_unknown) # Harmonic mean, HOS
        print("\nEvaluation: OS* %.4f, UNK %.4f, HOS %.4f" % (acc_known, acc_unknown, hos))

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