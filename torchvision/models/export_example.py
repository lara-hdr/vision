import numpy
import torch
import torchvision
import onnxruntime


def ort_validate(model_name, input, output):

    def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    input, _ = torch.jit._flatten(input)
    # output, _ = torch.jit._flatten(output)
    inputs = list(map(to_numpy, input))
    # outputs = list(map(to_numpy, output))

    ort_session = onnxruntime.InferenceSession(model_name)

    # compute onnxruntime output prediction
    ort_inputs = dict((ort_session.get_inputs()[i].name, inpt)
                      for i, inpt in enumerate(inputs))
    ort_outs = ort_session.run(None, ort_inputs)

    for i in range(0, len(output)):
        try:
            numpy.testing.assert_allclose(to_numpy(output[i]), ort_outs[i], rtol=1e-03, atol=1e-05)
        except Exception as e:
            print("******************* failed on ", ort_session.get_outputs()[i].name)
            print(str(e))


def export_and_check_model(model, model_name, input):
    print("Running----------------------")
    out = model(*input)

    print("Exporting----------------------")
    torch.onnx._export(model,
                       (input[0], input[1], input[2]),
                       model_name,
                       do_constant_folding=True,
                       opset_version=10)
    print(model_name + ' exported')
    # ort_validate(model_name, input, out)
    # print(model_name + ' validated')


# ------------------------- roi_align
'''
x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
single_roi = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)

model = torchvision.ops.RoIAlign((5, 5), 1, 2)
export_and_check_model(model, 'roi_align.onnx', (x, single_roi))
'''

# ------------------------- nms
'''
boxes = torch.randn(5,4)
scores = torch.randn(5)
class Module(torch.nn.Module):
    def forward(self, boxes, scores):
        return torchvision.ops.nms(boxes, scores, 0.5)
export_and_check_model(Module(), 'nms.onnx', (boxes, scores))
'''

# ------------------------- box_iou
'''
class Module(torch.nn.Module):
    def forward(self, box1, box2):
        return torchvision.ops.box_iou(box1, box2)
box1 = torch.randn(3,4)
box2 = torch.randn(2,4)
export_and_check_model(Module(), 'box_iou.onnx', (box1, box2))
'''

# ------------------------- multi_scale_roi_align
'''
# from collections import OrderedDict
# m = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat3'], 3, 2)
m = torchvision.ops.MultiScaleRoIAlign([0, 3], 3, 2)
# i = OrderedDict()
# i['feat1'] = torch.rand(1, 5, 64, 64)
# i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
# i['feat3'] = torch.rand(1, 5, 16, 16)
i = [torch.rand(1, 5, 64, 64), torch.rand(1, 5, 32, 32), torch.rand(1, 5, 16, 16)]
boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
# image_sizes = [(512, 512)]
image_sizes = [(torch.tensor(512), torch.tensor(512))]
output = m(i, [boxes], image_sizes)
export_and_check_model(m, 'multi_scale_roi_align.onnx', (i, [boxes], image_sizes))
'''

# ------------------------- feature_pyramid_network
'''
# from collections import OrderedDict
m = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
# x = OrderedDict()
# x['feat0'] = torch.rand(1, 10, 64, 64)
# x['feat2'] = torch.rand(1, 20, 16, 16)
# x['feat3'] = torch.rand(1, 30, 8, 8)
x = [torch.rand(1, 10, 64, 64), torch.rand(1, 20, 16, 16), torch.rand(1, 30, 8, 8)]
export_and_check_model(m, 'feature_pyramid_network.onnx', (x))
'''

# ------------------------- roi_pool
'''
x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
rois = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)

print('->',rois.size(0))
print('-->',x.size(1))
pool_h = 5
pool_w = 5
# input = torch.randn(1,3,5,5)
# boxes = torch.randn(1,5)
model = torchvision.ops.RoIPool((pool_h, pool_w), 2)
model.eval()
out = model(x, rois)

export_and_check_model(model, 'roi_pool.onnx', (x, rois))
'''

# ------------------------- box_area
'''
boxes = torch.randn(2, 4)
min_size = torch.tensor(1.)
class Module(torch.nn.Module):
    def forward(self, boxes, size):
        return torchvision.ops.boxes.remove_small_boxes(boxes, min_size)
export_and_check_model(Module(), 'remove_small_boxes.onnx', (boxes, min_size))
'''

# ------------------------- box_area
'''
boxes = torch.randn(2, 4)
size = (torch.tensor(2), torch.tensor(2))
class Module(torch.nn.Module):
    def forward(self, boxes, size):
        return torchvision.ops.boxes.clip_boxes_to_image(boxes, size)
export_and_check_model(Module(), 'clip_boxes_to_image.onnx', (boxes, size))
'''

# ------------------------- box_area
'''
boxes = torch.randn(2, 4)
class Module(torch.nn.Module):
    def forward(self, boxes):
        return torchvision.ops.boxes.box_area(boxes)
export_and_check_model(Module(), 'box_area.onnx', (boxes))
'''
