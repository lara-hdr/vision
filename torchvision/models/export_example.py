import numpy
import torch
import torchvision
import onnxruntime

def ort_validate(model_name, input, output):

    def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


    if isinstance(output, list) and len(output) >= 1 and isinstance(output[0], dict):
        output = [list(x.values()) for x in output]
    elif isinstance(output, dict):
        output = list(output.values())

    input, _ = torch.jit._flatten(input)
    output, _ = torch.jit._flatten(output)
    inputs = list(map(to_numpy, input))
    # outputs = list(map(to_numpy, output))

    ort_session = onnxruntime.InferenceSession(model_name)

    # compute onnxruntime output prediction
    ort_inputs =  dict((ort_session.get_inputs()[i].name, inpt)
                      for i, inpt in enumerate(inputs))
    ort_outs = ort_session.run(None, ort_inputs)

    #print("~~~~~~~~~~~~~~~")
    #print(ort_outs)

    for i in range(0, len(output)):
        try:
            numpy.testing.assert_allclose(to_numpy(output[i]), ort_outs[i], rtol=1e-02, atol=1e-04)
            print("** input ", i, " validated", output[i].shape)
        except Exception as e:
            print("******************* failed on ", ort_session.get_outputs()[i].name)
            #print("<<")
            #print(output[i])
            #print("==")
            #print(ort_outs[i])
            #print(">>")
            print(str(e))


def export_and_check_model(model, model_name, input):
    model.eval()

    print("Running----------------------")
    out = model(input)
    #print(out)
    #return None
    
    print("Exporting----------------------")
    torch.onnx._export(model,
                       (input),
                       model_name,
                       do_constant_folding=True,
                       opset_version=10,
                       example_outputs = out)
    print(model_name + ' exported')
    ort_validate(model_name, input, out)
    print(model_name + ' validated')


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

pool_h = 5
pool_w = 5
# input = torch.randn(1,3,5,5)
# boxes = torch.randn(1,5)

model = torchvision.ops.RoIPool((pool_h, pool_w), 2)

#out0 = torch.ops.torchvision.roi_pool_forward(x, rois, 2, pool_h, pool_w)
#print(out0)
model.eval()
out = model(x, rois)

#export_and_check_model(model, 'roi_pool.onnx', (x, rois))
torch.onnx._export(model,
                   (x, rois),
                   'roi_pool.onnx',
                   do_constant_folding=True,
                   opset_version=10)
print('roi_pool.onnx' + ' exported')
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print(out)
ort_validate('roi_pool.onnx', (x, rois), out)
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

# -------------------------------------------------- models
# input = [torch.randn(3, 300, 400), torch.randn(3, 500, 400)]

import requests
from PIL import Image
from io import BytesIO


def fetch_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


url = "http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg"
pil_image = fetch_image(url=url)


#image = torch.from_numpy(numpy.array(pil_image)[:, :, [2, 1, 0]])
#image = torch.nn.functional.upsample(image.permute(2, 0, 1).unsqueeze(0).to(torch.float),
#                                     size=(800, 1280)  # size=(960, 1280))
#                                     ).to(torch.uint8).squeeze(0).permute(1, 2, 0).to('cpu')
#image = image.permute(2, 0, 1)
#image = [image.float(), image.float()]
#print(image[0].shape)

pil_image1 = torchvision.transforms.functional.resize(pil_image, (800, 1280))
image_tensor1 = torchvision.transforms.functional.to_tensor(pil_image1)

pil_image2 = torchvision.transforms.functional.resize(pil_image, (1280, 800))
image_tensor2 = torchvision.transforms.functional.to_tensor(pil_image2)

image_tensor = [image_tensor1]#,image_tensor1]#, image_tensor2]#, image_tensor]
print(image_tensor[0].shape, )

#numpy.testing.assert_allclose(input[0], image_tensor[0])

# ------------------------- fasterrcnn_resnet50_fpn

#model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True)
#export_and_check_model(model, 'faster_rcnn.onnx', image_tensor)


# ------------------------- maskrcnn_resnet50_fpn

model = torchvision.models.detection.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True)
export_and_check_model(model, 'mask_rcnn.onnx', image_tensor)

'''
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                output_size=7,
                                                sampling_ratio=2)
keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                         output_size=14,
                                                         sampling_ratio=2)
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
import torchvision.models.detection.rpn as r
anchor_generator = r.AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                      aspect_ratios=((0.5, 1.0, 2.0),))
# put the pieces together inside a FasterRCNN model
import torchvision.models.detection.keypoint_rcnn as k
model = k.KeypointRCNN(backbone,
                     num_classes=2,
                     rpn_anchor_generator=anchor_generator,
                     box_roi_pool=roi_pooler,
                     keypoint_roi_pool=keypoint_roi_pooler)
model.eval()
input = [torch.rand(13, 800, 1280)]#, torch.rand(3, 800, 1280)]
predictions = model(x)
#print(predictions)'''


# ------------------------- keypointrcnn_resnet50_fpn

#input = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
#model = torchvision.models.detection.keypoint_rcnn.keypointrcnn_resnet50_fpn(pretrained=True)
#export_and_check_model(model, 'keypoint_rcnn.onnx', image_tensor)

"""
print("\n\n... TESTING WITH DIFFERENT IMAGE ...\n")

url = "https://d31j74p4lpxrfp.cloudfront.net/sites/default/files/styles/mobile_large_image/public/int_files/1012343_edited-1.jpg?itok=ZF4Mtgbf"
pil_image = fetch_image(url=url)
pil_image = torchvision.transforms.functional.resize(pil_image, (800, 1280)) #)1280, 800)) #
image_tensor = torchvision.transforms.functional.to_tensor(pil_image)
image_tensor = [image_tensor]#, image_tensor]#, image_tensor]
print(image_tensor[0].shape)

model.eval()
out = model(image_tensor)
ort_validate('mask_rcnn.onnx', (image_tensor), out)"""


"""
a = "abc"
x = torch.randn(2, 4)
c = 2
d = {"a" : x}
class Module(torch.nn.Module):
    def forward(self, x):
        return x
torch.jit.trace(Module(), d)
"""