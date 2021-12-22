
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import random


#### Implement the evaluation on the target for the known/unknown separation
"""
    Method is to be completed. It's supposed to do evaluation of the target domain: 
    if I can predict the orientation of the target image => image is from known category
    results are returned in a file under the "new_txt_list" folder

    Parameters
    ---------
        args : data-type -> {source : str, target : str}
            source and target fields contain the name of a class of images (such as "Clipart" or "RealWorld") (? maybe)
        feature_extractor
            only used for calling it's eval() method
        rot_cls
            stands for rotation classifier. Only used for calling it's eval() method
        target_loader_eval : list<data-type> -> {data, class_l, data_rot, rot_l}
            for each image contains: image, loss for the image, rotated image, loss for the rotated image (?)
        device
            CPU or GPU ?
    Returns
    -------
        rand : number
            identifier of the file in which results of the evaluation will be stored
"""
def evaluation(args,feature_extractor,rot_cls,target_loader_eval,device):

    feature_extractor.eval()
    rot_cls.eval()


    with torch.no_grad():
        for it, (data,class_l,data_rot,rot_l) in enumerate(target_loader_eval):
            data, class_l, data_rot, rot_l = data.to(device), class_l.to(device), data_rot.to(device), rot_l.to(device)

    #TODO: compute normality score for each image (i.e. a vector of numbers)
    
    #compute the AUROC score from the vector of normality scores. AUROC MUST be >0.5
    auroc = roc_auc_score(ground_truth,normality_score)
    print('AUROC %.4f' % auroc)

    #TODO: select a threshold, samples with normality score > threshold are known, the others are unknown


    # create new txt files
    rand = random.randint(0,100000)
    print('Generated random number is :', rand)

    # This txt files will have the names of the source images and the names of the target images selected as unknown
    target_unknown = open('new_txt_list/' + args.source + '_known_' + str(rand) + '.txt','w')

    # This txt files will have the names of the target images selected as known
    target_known = open('new_txt_list/' + args.target + '_known_' + str(rand) + '.txt','w')

    #TODO: store known and unknown samples in files target_unknown and target_known
    #TODO: create and update variables number_of_known_samples and number_of_unknown_samples

    print('The number of target samples selected as known is: ',number_of_known_samples)
    print('The number of target samples selected as unknown is: ', number_of_unknown_samples)

    return rand






