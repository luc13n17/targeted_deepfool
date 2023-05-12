# This file contains the code for Targete DeepFool. 
# Plese check the comments above the functions and the return statement to see which ones are for experiments.

import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure

# Use this to run code on test_deepfool_targeted
def deepfool_targeted(image, net, target_class, overshoot=0.02, min_confidence = 95):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param target_class: target class that the image should be misclassified as
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param min_confidence: used to set minimum amount of confidence.
       :return: minimal perturbation that fools the classifier, number of iterations that it required, target label and perturbed image
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
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    k_i = label
    confidences = []
    confidence = 0
    while k_i != target_class or confidence < min_confidence:

        fs[0, label].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        zero_gradients(x)

        fs[0, target_class].backward(retain_graph=True)
        cur_grad = x.grad.data.cpu().numpy().copy()

        # set new w_k and new f_k
        w_k = cur_grad - grad_orig
        f_k = (fs[0, target_class] - fs[0, label]).data.cpu().numpy()

        pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

        # determine perturbation direction
        w = (pert_k + 1e-4) * w_k / np.linalg.norm(w_k)

        # compute r_i and r_tot
        r_i = (1 + overshoot) * w
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1 + overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1 + overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())
        
        confidence = torch.nn.functional.softmax(fs, dim=1)[0, target_class].item() * 100
        confidences.append(confidence)

        loop_i += 1
    
    r_tot = (1+overshoot)*r_tot

    # Calculate confidence and print it
    confidence_target = torch.nn.functional.softmax(fs, dim=1)[0, target_class].item()
    confidence_orig = torch.nn.functional.softmax(fs, dim=1)[0, label].item()
    print("Source Confidence: {:.2f}%".format(confidence_orig * 100))
    print("Target Confidence: {:.2f}%".format(confidence_target * 100))
    print("Number of perturbations: {}".format(loop_i))

    # Compute percentage of change in image
    orig_image = image.cpu().numpy().flatten()
    perturbed_image = pert_image.cpu().numpy().flatten()
    l2_dist = np.linalg.norm(perturbed_image - orig_image)
    max_l2_dist = np.sqrt(np.prod(input_shape))
    change = (l2_dist / max_l2_dist)
    print("Change in image: {:.2f}%".format(change * 100))

    # Calculate SSIM between original and perturbed image
    ssim = structural_similarity_index_measure(image.to('cuda').unsqueeze(0), pert_image)

    # Convert tensor to float and round to 4 decimal places
    ssim_value = round(ssim.item(), 4)

    # Print the similarity measure
    print("SSIM score:", ssim_value)

    return r_tot, loop_i, label, k_i, pert_image, confidences

# Use this to run the code for "experiment_imagenet_val.ipynb"

# def deepfool_targeted(image, net, target_class, overshoot=0.02, min_confidence = 95, max_iter = 100):

#     """
#        :param image: Image of size HxWx3
#        :param net: network (input: images, output: values of activation **BEFORE** softmax).
#        :param target_class: target class that the image should be misclassified as
#        :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
#        :param min_confidence: used to set minimum amount of confidence.
#        :param max_iter: maximum number of iterations for deepfool (default = 50)
#        :return: minimal perturbation that fools the classifier, number of iterations that it required, target label and perturbed image
#     """

#     is_cuda = torch.cuda.is_available()

#     if is_cuda:
#         # print("Using GPU")
#         image = image.cuda()
#         net = net.cuda()
#     # else:
#     #     print("Using CPU")

#     f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
#     I = (np.array(f_image)).flatten().argsort()[::-1]
#     label = I[0]

#     input_shape = image.cpu().numpy().shape
#     pert_image = copy.deepcopy(image)
#     w = np.zeros(input_shape)
#     r_tot = np.zeros(input_shape)

#     loop_i = 0

#     x = Variable(pert_image[None, :], requires_grad=True)
#     fs = net.forward(x)
#     k_i = label
#     confidences = []
#     confidence = 0
#     successful = False
#     while k_i != target_class or confidence < min_confidence:
#         if loop_i < max_iter: 
#             fs[0, label].backward(retain_graph=True)
#             grad_orig = x.grad.data.cpu().numpy().copy()

#             zero_gradients(x)

#             fs[0, target_class].backward(retain_graph=True)
#             cur_grad = x.grad.data.cpu().numpy().copy()

#             # set new w_k and new f_k
#             w_k = cur_grad - grad_orig
#             f_k = (fs[0, target_class] - fs[0, label]).data.cpu().numpy()

#             pert_k = abs(f_k) / np.linalg.norm(w_k.flatten())

#             # determine perturbation direction
#             w = (pert_k + 1e-4) * w_k / np.linalg.norm(w_k)

#             # compute r_i and r_tot
#             r_i = (1 + overshoot) * w
#             r_tot = np.float32(r_tot + r_i)

#             if is_cuda:
#                 pert_image = image + (1 + overshoot)*torch.from_numpy(r_tot).cuda()
#             else:
#                 pert_image = image + (1 + overshoot)*torch.from_numpy(r_tot)

#             x = Variable(pert_image, requires_grad=True)
#             fs = net.forward(x)
#             k_i = np.argmax(fs.data.cpu().numpy().flatten())

#             confidence = torch.nn.functional.softmax(fs, dim=1)[0, target_class].item() * 100
#             confidences.append(confidence)

#             loop_i += 1
#         else:
#             break
    
#     r_tot = (1+overshoot)*r_tot

#     if k_i == target_class:
#         successful = True
#     # Calculates confidence and prints it
#     confidence_target = torch.nn.functional.softmax(fs, dim=1)[0, target_class].item()

#     # compute percentage of change
#     orig_image = image.cpu().numpy().flatten()
#     perturbed_image = pert_image.cpu().numpy().flatten()
#     l2_dist = np.linalg.norm(perturbed_image - orig_image)
#     max_l2_dist = np.sqrt(np.prod(input_shape))
#     change = (l2_dist / max_l2_dist) * 100

#     # calculate SSIM between original and perturbed image
#     ssim = structural_similarity_index_measure(image.to('cuda').unsqueeze(0), pert_image)

#     # Convert tensor to float and round to 4 decimal places
#     ssim_value = round(ssim.item(), 4)

#     return r_tot, loop_i, label, k_i, pert_image, confidences, confidence_target, change, ssim_value, successful





