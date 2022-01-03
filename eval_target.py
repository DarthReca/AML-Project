from resnet import Classifier, ResNet
import torch
from sklearn.metrics import roc_auc_score
import random
import argparse

from torch.utils.data.dataloader import DataLoader

#### Implement the evaluation on the target for the known/unknown separation
def evaluation(
    args: argparse.Namespace,
    feature_extractor: ResNet,
    rot_cls: Classifier,
    target_loader_eval: DataLoader,
    device: torch.device,
):
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
            Is an iterable wrapping the dataset containing images of target domain and their labels. for each image contains: image, label for the image, rotated image, label for the rotated image
        device
            CPU or GPU ?
    Returns
    -------
        rand : number
            identifier of the file in which results of the evaluation will be stored
    """
    feature_extractor.eval()
    rot_cls.eval()

    normality_score = []
    ground_truth = []

    with torch.no_grad():
        # iterate over the target images to compute normality scores (i.e. how sure we are of the predicted rotation)
        for (data, class_l, data_rot, rot_l) in target_loader_eval:
            data, class_l, data_rot, rot_l = (
                data.to(device),
                class_l.to(device),
                data_rot.to(device),
                rot_l.to(device),
            )

            original_features = feature_extractor(data)
            rotated_features = feature_extractor(data_rot)

            # Here we have to concatenate tensors as suggested by the architecture
            rotation_scores = rot_cls(
                torch.cat([original_features, rotated_features], 1)
            )
            # normality score is maximum prediction of a class. e.g. if image is rotated 90° left with 70% probability, normality score = 0.7
            normality_score.append(max(rotation_scores))
            # ground truth is label of the actual rotation of the image
            ground_truth.append(rot_l)

    # compute the AUROC score from the vector of target labels and the vector of normality scores. AUROC MUST be >0.5
    auroc = roc_auc_score(ground_truth, normality_score)  # type: float
    print("AUROC %.4f" % auroc)

    # create new txt files
    rand = random.randint(0, 100000)
    print("Generated random number is :", rand)

    # This txt files will have ??? -> the names of the source images and <- ??? the names of the target images selected as unknown
    target_unknown = open(
        "new_txt_list/" + args.source + "_known_" + str(rand) + ".txt", "w"
    )

    # This txt files will have the names of the target images selected as known
    target_known = open(
        "new_txt_list/" + args.target + "_known_" + str(rand) + ".txt", "w"
    )

    number_of_known_samples = 0
    number_of_unknown_samples = 0
    with torch.no_grad():
        for img_id, (data, class_l, data_rot, rot_l) in enumerate(target_loader_eval):
            if normality_score[img_id] >= args.threshold:
                # we consider the domain of the image as known
                target_known.write(target_loader_eval.dataset.names[img_id] + "\n")
                number_of_known_samples += 1
            else:
                # we consider the domain of the image as UNknown
                target_unknown.write(target_loader_eval.dataset.names[img_id] + "\n")
                number_of_unknown_samples += 1

    target_known.close()
    target_unknown.close()

    print(
        "The number of target samples selected as known is: ", number_of_known_samples
    )
    print(
        "The number of target samples selected as unknown is: ",
        number_of_unknown_samples,
    )

    return rand
