import argparse

import torch

import data_helper
from resnet import resnet18_feat_extractor, Classifier

from step1_KnownUnknownSep import step1
from eval_target import evaluation
from step2_SourceTargetAdapt import step2


def get_args():
    """
    Arguments for the program

    Parameters
    ----------
    Nil

    Returns
    -------
    The method for parsing command line arguments. Use '-h' as the command line argument to display the list of options and their usage.

    """
    parser = argparse.ArgumentParser(
        description="Script to launch training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--source", default="Art", help="Source name")
    parser.add_argument("--target", default="Clipart", help="Target name")
    parser.add_argument(
        "--n_classes_known", type=int, default=45, help="Number of known classes"
    )
    parser.add_argument(
        "--n_classes_tot", type=int, default=65, help="Number of unknown classes"
    )

    # dataset path
    parser.add_argument(
        "--path_dataset",
        default="./AML-Project/data",
        help="Path where the Office-Home dataset is located",
    )

    # data augmentation
    parser.add_argument(
        "--min_scale", default=0.8, type=float, help="Minimum scale percent"
    )
    parser.add_argument(
        "--max_scale", default=1.0, type=float, help="Maximum scale percent"
    )
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument(
        "--random_grayscale",
        default=0.1,
        type=float,
        help="Randomly greyscale the image",
    )

    # training parameters
    parser.add_argument("--image_size", type=int, default=222, help="Image size")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )

    parser.add_argument(
        "--epochs_step1",
        type=int,
        default=10,
        help="Number of epochs of step1 for known/unknown separation",
    )
    parser.add_argument(
        "--epochs_step2",
        type=int,
        default=10,
        help="Number of epochs of step2 for source-target adaptation",
    )

    parser.add_argument(
        "--train_all",
        type=bool,
        default=True,
        help="If true, all network weights will be trained",
    )

    parser.add_argument(
        "--weight_RotTask_step1",
        type=float,
        default=0.5,
        help="Weight for the rotation loss in step1",
    )
    parser.add_argument(
        "--weight_RotTask_step2",
        type=float,
        default=0.5,
        help="Weight for the rotation loss in step2",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for the known/unkown separation",
    )

    # tensorboard logger
    parser.add_argument(
        "--tf_logger",
        type=bool,
        default=True,
        help="If true will save tensorboard compatible logs",
    )
    parser.add_argument(
        "--folder_name", default=None, help="Used by the logger to save logs"
    )

    return parser.parse_args()


class Trainer:
    """
    Class for the Trainer.

    """

    def __init__(self, args):
        """
        Initialisation of the trainer object

        Parameters
        ----------
        args : argument parser output, the command line arguments

        """
        self.args = args

        # Set device for allocation of the tensor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ResNet model
        self.feature_extractor = resnet18_feat_extractor()
        # Initiate object classifier with input_size=512, and numbers of classes is known classes + 1 (the unknown class, trained only in step2)
        self.obj_classifier = Classifier(512, self.args.n_classes_known + 1)
        # Initiate rotation classifier with input_size=512*2 and 4 classes: [0, 90, 180, 270]
        self.rot_classifier = Classifier(512 * 2, 4)

        # Load model into the device
        self.feature_extractor = self.feature_extractor.to(self.device)
        self.obj_cls = self.obj_classifier.to(self.device)
        self.rot_cls = self.rot_classifier.to(self.device)

        # Path to file with the list of known class images, source is the domain of the image (from command line argument, default is Art)
        source_path_file = "txt_list/" + args.source + "_known.txt"
        # Create the source data loader, providing an iterable over the dataset - using map-style dataset.
        self.source_loader = data_helper.get_train_dataloader(args, source_path_file)

        # Path to the file with the list of target images (from command line argument, default is Clipart)
        target_path_file = "txt_list/" + args.target + ".txt"
        # Create two data loaders for the target dataset, one for training and one for evaluation.
        # (Probably this is a typo and the first one should be .get_train_dataloader as in step 2 below)
        self.target_loader_train = data_helper.get_val_dataloader(
            args, target_path_file
        )
        self.target_loader_eval = data_helper.get_val_dataloader(args, target_path_file)

        print("Source: ", self.args.source, " Target: ", self.args.target)
        print(
            "Dataset size: source %d, target %d"
            % (len(self.source_loader.dataset), len(self.target_loader_train.dataset))
        )

    def do_training(self):
        """
        Runs the training

        """

        # Run step 1: train the object and rotation classifier
        print("Step 1 --------------------------------------------")
        step1(
            self.args,
            self.feature_extractor,
            self.rot_cls,
            self.obj_cls,
            self.source_loader,
            self.device,
        )
        # Evaluate the roation classifier on the target domain to split target into known and unknown classes.
        print("Target - Evaluation -- for known/unknown separation")
        # Returns a random integer (don't really understand why - just to add to the generated file names?)
        rand = evaluation(
            self.args,
            self.feature_extractor,
            self.rot_cls,
            self.target_loader_eval,
            self.device,
        )

        # Step 2 set up
        # Create a new data loader for the known source images
        source_path_file = (
            "new_txt_list/" + self.args.source + "_known_" + str(rand) + ".txt"
        )
        self.source_loader = data_helper.get_train_dataloader(
            self.args, source_path_file
        )

        # Create train and evaluation data loaders for the known target images
        target_path_file = (
            "new_txt_list/" + self.args.target + "_known_" + str(rand) + ".txt"
        )
        self.target_loader_train = data_helper.get_train_dataloader(
            self.args, target_path_file
        )
        self.target_loader_eval = data_helper.get_val_dataloader(
            self.args, target_path_file
        )

        # Run step 2: on known part do source-target domain adaptation, on unknown part of target train the unknown class
        print("Step 2 --------------------------------------------")
        step2(
            self.args,
            self.feature_extractor,
            self.rot_cls,
            self.obj_cls,
            self.source_loader,
            self.target_loader_train,
            self.target_loader_eval,
            self.device,
        )


def main():
    # main function of the program

    # Get arguments from the command line
    args = get_args()

    # Create Trainer object
    trainer = Trainer(args)

    # Initiate training
    trainer.do_training()


if __name__ == "__main__":

    # Enable cudnn to benchmark multiple convolution algorithms and select the fastest.
    torch.backends.cudnn.benchmark = True

    # Launch main program
    main()
