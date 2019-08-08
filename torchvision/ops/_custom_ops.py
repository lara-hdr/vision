import os
import torch

torch.ops.load_library(os.path.join(os.path.dirname(__file__),
                                    '..',
                                    'custom_ops.cpython-37m-x86_64-linux-gnu.so'))


def register_custom_op():
    from torch.onnx.symbolic_helper import parse_args, scalar_type_to_onnx
    from torch.onnx.symbolic_opset9 import select, unsqueeze, squeeze, _cast_Long, reshape

    @parse_args('v', 'v', 'f')
    def symbolic_multi_label_nms(g, boxes, scores, iou_threshold):
        boxes = unsqueeze(g, boxes, 0)
        scores = unsqueeze(g, unsqueeze(g, scores, 0), 0)
        max_output_per_class = g.op('Constant', value_t=torch.tensor([2000], dtype=torch.long))
        iou_threshold = g.op('Constant', value_t=torch.tensor([iou_threshold], dtype=torch.float))
        nms_out = g.op('NonMaxSuppression', boxes, scores, max_output_per_class, iou_threshold)
        return squeeze(g, select(g, nms_out, 1, g.op('Constant', value_t=torch.tensor([2], dtype=torch.long))), 1)

    @parse_args('v', 'v', 'f', 'i', 'i', 'i')
    def roi_align(g, input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
        batch_indices = _cast_Long(g, squeeze(g, select(g, rois, 1,
                                   g.op('Constant', value_t=torch.tensor([0], dtype=torch.long))), 1), False)
        rois = select(g, rois, 1, g.op('Constant', value_t=torch.tensor([1, 2, 3, 4], dtype=torch.long)))
        return g.op('RoiAlign',
                    input,
                    rois,
                    batch_indices,
                    spatial_scale_f=spatial_scale,
                    output_height_i=pooled_height,
                    output_width_i=pooled_width,
                    sampling_ratio_i=sampling_ratio)

    @parse_args('v', 'v', 'i', 'i', 'f')
    def roi_pool(g, input, rois, pooled_height, pooled_width, spatial_scale):
        roi_pool = g.op('MaxRoiPool',
                        input,
                        rois,
                        pooled_shape_i=(pooled_height, pooled_width),
                        spatial_scale_f=spatial_scale)
        argmax = g.op('ArgMax', roi_pool, axis_i=0, keepdims_i=False)
        argmax = g.op("Cast", argmax, to_i=scalar_type_to_onnx[3])
        return roi_pool, argmax

    from torch.onnx import register_custom_op_symbolic
    register_custom_op_symbolic('torchvision::nms', symbolic_multi_label_nms, 10)
    register_custom_op_symbolic('torchvision::roi_align_forward', roi_align, 10)
    register_custom_op_symbolic('torchvision::roi_pool_forward', roi_pool, 10)


register_custom_op()
