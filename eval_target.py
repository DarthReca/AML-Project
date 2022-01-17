from math import log
from resnet import Classifier, ResNet
import torch
from sklearn.metrics import roc_auc_score
import random
import argparse
import os
import shutil

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

    normality_score = torch.empty(
        size=[len(target_loader_eval.dataset)], dtype=torch.float32
    )
    ground_truth = torch.empty(
        size=[len(target_loader_eval.dataset)], dtype=torch.int32
    )

    # DEBUG VERSION
    # normality_score = torch.empty(size=[512], dtype=torch.float32)
    # ground_truth = torch.empty(size=[512], dtype=torch.int32)

    with torch.no_grad():
        # iterate over the target images to compute normality scores (i.e. how sure we are of the predicted rotation)
        for index, (data, class_l, _, _) in enumerate(target_loader_eval):
            # DEBUG VERSION
            # if index == 512:
            #    break

            data, class_l = (
                data.to(device),
                class_l.to(device),
            )
            original_features = feature_extractor(data)

            """
            entropy_losses = torch.zeros([4])
            rotation_scores = torch.zeros([4, 4])
            for i in range(1, 5):
            
                rotated_features = feature_extractor(
                    torch.rot90(data, k=i, dims=[2, 3])
                )
            """
            entropy_losses = torch.zeros([2])
            rotation_scores = torch.zeros([2, 2])
            for i in range(2):
                flipped = torch.fliplr(data) if i==1 else data
                rotated_features = feature_extractor(flipped)
                rotation_probabilities = torch.nn.Softmax(dim=0)(
                    torch.flatten(
                        rot_cls(torch.cat([original_features, rotated_features], 1))
                    )
                )
                entropy_losses[i] = (
                    rotation_probabilities.dot(torch.log10(rotation_probabilities))
                    / log(args.n_classes_known, 10)
                ).item()
                rotation_scores[i] = rotation_probabilities
            # normality score is maximum prediction of a class. e.g. if image is rotated 90° left with 70% probability, normality score = 0.7
            normality_score[index] = torch.max(torch.mean(rotation_scores, dim=0))
            """If you want to use entropy
            normality_score[index] = torch.max(
                normality_score[index], 1 - torch.mean(entropy_losses)
            )
            """
            # ground truth is label indicating known/unknown
            ground_truth[index] = 0 if class_l >= args.n_classes_known else 1

    # compute the AUROC score from the vector of target labels and the vector of normality scores. AUROC MUST be >0.5
    auroc = roc_auc_score(ground_truth, normality_score)  # type: float
    print("AUROC %.4f" % auroc)

    # create new txt files
    rand = random.randint(0, 100000)
    print("Generated random number is :", rand)

    if not os.path.isdir("new_txt_list"):
        os.mkdir("new_txt_list")

    # This txt files will have the names of the source images and the names of the target images selected as unknown
    source_unknown_path = shutil.copyfile(
        f"txt_list/{args.source}_known.txt",
        f"new_txt_list/{args.source}_known_{rand}.txt",
    )
    target_unknown = open(source_unknown_path, "a+")
    #new line at the end of source images, otherwise first target unknown image is on the same line as last source image
    target_unknown.write(f"\n")

    # This txt files will have the names of the target images selected as known
    target_known = open(f"new_txt_list/{args.target}_known_{rand}.txt", "w+")

    number_of_known_samples = 0
    number_of_unknown_samples = 0
    with torch.no_grad():
        for img_id, (_, class_l, _, _) in enumerate(target_loader_eval):
            # DEBUG VERSION
            # if img_id == 512:
            #    break
            if normality_score[img_id] >= args.threshold:
                # we consider the domain of the image as known
                target_known.write(
                    f"{target_loader_eval.dataset.names[img_id]} {class_l.item()}\n"
                )
                number_of_known_samples += 1
            else:
                # we consider the domain of the image as UNknown
                target_unknown.write(f"{target_loader_eval.dataset.names[img_id]} {args.n_classes_known}\n")
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
