import random
import math
import torch
from torch import nn

from torchvision.ops import misc as misc_nn_ops
from .image_list import ImageList
from .roi_heads import paste_masks_in_image


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets=None):
        images = [img for img in images]
        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else targets
            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)
            image, target = self.resize(image, target)
            images[i] = image
            if targets is not None:
                targets[i] = target

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_list = ImageList(images, image_sizes)
        return image_list, targets

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        h, w = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        if self.training:
            size = random.choice(self.min_size)
        else:
            # FIXME assume for now that testing uses the largest scale
            size = self.min_size[-1]
        scale_factor = size / min_size
        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size
        image = torch.nn.functional.interpolate(
            # !! Temp modification for testing purposes !!
            #       !! This will NOT be checked in !!
            #
            # we are adding ONNX support for bilinear resize
            # matching Pytorch's bilinear resize very soon.
            #
            # (+ mismatch bug with scale_factor when not int ?
            # (testing with cases where scale_factor is an int for now))
            image[None], scale_factor=scale_factor, mode='nearest')[0]  # align_corners=False

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        if "masks" in target:
            mask = target["masks"]
            mask = misc_nn_ops.interpolate(mask[None].float(), scale_factor=scale_factor)[0].byte()
            target["masks"] = mask

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints
        return image, target

    def batch_images(self, images, size_divisible=32):
        # concatenate
        if torch._C._get_tracing_state():
            max_size = tuple(torch.max(torch.tensor(s), dim=0, keepdim=True)[0] for s in zip(*[img.shape for img in images]))
            max_size = list(max_size)
            stride = size_divisible
            max_size[1] = (torch.ceil(torch.tensor(max_size[1], dtype=torch.float32) / stride) * stride).to(dtype=torch.int64)
            max_size[2] = (torch.ceil(torch.tensor(max_size[2], dtype=torch.float32) / stride) * stride).to(dtype=torch.int64)
            max_size = tuple(max_size)
        else:
            max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
            max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)
            max_size = tuple(max_size)

        # the line 'pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)'
        # is not supported in onnx (indexing + inplace operation)
        if torch._C._get_tracing_state():
            padded_imgs = ()
            for img in images:
                padding = [int(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
                zeros_0 = torch.zeros(padding[0], img.shape[1], img.shape[2])
                concat_0 = torch.cat((img, zeros_0), 0)
                zeros_1 = torch.zeros(concat_0.shape[0], padding[1], concat_0.shape[2])
                concat_1 = torch.cat((concat_0, zeros_1), 1)
                zeros_2 = torch.zeros(concat_1.shape[0], concat_1.shape[1], padding[2])
                concat_2 = torch.cat((concat_1, zeros_2), 2)
                padded_img = torch.unsqueeze(concat_2, 0)
                padded_imgs = padded_imgs + tuple(padded_img)
            return torch.stack(padded_imgs)
        else:
            batch_shape = (len(images),) + max_size
            batched_imgs = images[0].new(*batch_shape).zero_()
            for img, pad_img in zip(images, batched_imgs):
                pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result


def resize_keypoints(keypoints, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    if torch._C._get_tracing_state():
        resized_data_0 = resized_data[:, :, 0] * ratio_w
        resized_data_1 = resized_data[:, :, 1] * ratio_h
        resized_data = torch.stack((resized_data_0, resized_data_1, resized_data[:, :, 2]), dim=2)
    else:
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
    return resized_data


def resize_boxes(boxes, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_height, ratio_width = ratios

    # unbind and lists not yet supported in ONNX.
    # we are planning on adding support soon.
    if torch._C._get_tracing_state():
        xmin, ymin, xmax, ymax = [torch.squeeze(out, 1) for out in boxes.split(1, dim=1)]
    else:
        xmin, ymin, xmax, ymax = [torch.squeeze(out, 0) for out in boxes.unbind(1)]

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
