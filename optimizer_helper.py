from torch import optim


def get_optim_and_scheduler(feature_extractor,rot_cls,obj_cls, epochs, lr, train_all):
    """
    Set up the optimizer and learning rate scheduler.

    Parameters
    ----------
    feature_extractor : nn.Module
        Feature extractor for images.
    rot_cls : nn.Module
        Rotation classifier.
    obj_cls : nn.Module
        Object classifier.
    epochs: int
        Number of epochs
    lr: float
        Learning rate
    train_all: boolean
        If true train all features

    Returns
    -------
    optimizer: Optimizer
        Optimizer algorithm
    scheduler: torch.optim.lr_scheduler
        Learning rate scheduler
    """

    if train_all:
        params = list(feature_extractor.parameters()) + list(rot_cls.parameters()) + list(obj_cls.parameters())
    else:
        params = list(rot_cls.parameters()) + list(obj_cls.parameters())

    # Using stochastic gradient descent with momentum
    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, lr=lr)
    
    # Decays the learning rate of each parameter group by gamma (default 0.1) every step_size epochs
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)


    print("Step size: %d" % step_size)
    return optimizer, scheduler