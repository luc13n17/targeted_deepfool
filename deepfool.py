# This file contains the code written by the authors of the original DeepFool paper.

import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
from torchmetrics.functional import structural_similarity_index_measure


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    if is_cuda:
        print("Using GPU")
        image = image.cuda()
        net = net.cuda()
    else:
        print("Using CPU")


    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    # Calculates confidence and prints it
    confidence_pert = torch.nn.functional.softmax(fs, dim=1)[0, k_i].item()
    confidence_orig = torch.nn.functional.softmax(fs, dim=1)[0, label].item()
    print("Source Confidence: {:.2f}%".format(confidence_orig * 100))
    print("Perturbed Confidence: {:.2f}%".format(confidence_pert * 100))
    print("Number of perturbations: {}".format(loop_i))

    # compute percentage of change
    orig_image = image.cpu().numpy().flatten()
    perturbed_image = pert_image.cpu().numpy().flatten()
    l2_dist = np.linalg.norm(perturbed_image - orig_image)
    max_l2_dist = np.sqrt(np.prod(input_shape))
    change = (l2_dist / max_l2_dist)
    print("Change in image: {:.2f}%".format(change * 100))

    # calculate SSIM between original and perturbed image
    ssim = structural_similarity_index_measure(image.to('cuda').unsqueeze(0), pert_image)

    # Convert tensor to float and round to 4 decimal places
    ssim_value = round(ssim.item(), 4)

    # Print the similarity measure
    print("SSIM score:", ssim_value)
    
    return r_tot, loop_i, label, k_i, pert_image