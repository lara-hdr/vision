import io
import torch
from torchvision import ops
from torchvision import models
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from collections import OrderedDict

# onnxruntime requires python 3.5 or above
try:
    import onnxruntime
except ImportError:
    onnxruntime = None

import unittest


@unittest.skipIf(onnxruntime is None, 'ONNX Runtime unavailable')
class ONNXExporterTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(123)

    def run_model(self, model, inputs, other_inputs=None):
        model.eval()

        # run pytorch model
        with torch.no_grad():
            if isinstance(inputs, torch.Tensor):
                inputs = (inputs,)
            outputs = model(*inputs)
            if isinstance(outputs, torch.Tensor):
                outputs = (outputs,)
        #print(outputs)

        #print("RETURNING BEFORE")
        
        onnx_io = io.BytesIO()
        # export to onnx
        torch.onnx.export(model, inputs, "fasterrcnn.onnx", # 
                          do_constant_folding=True,
                          opset_version=10,verbose=False)
        #return
        #dynamic_axes={'input_1': [0,1,2,3],
        #              'input_2': [0,1,2,3],
        #              'input_3': [0,1,2,3]})

        # validate the exported model with onnx runtime
        self.ort_validate(onnx_io, inputs, outputs)

        # verify with the other inputs
        if other_inputs:
            for test_inputs in other_inputs:
                with torch.no_grad():
                    print("\n\n**********NEW TEST : ")
                    if isinstance(test_inputs, torch.Tensor):
                        test_inputs = (test_inputs,)
                    test_ouputs = model(*test_inputs)
                    if isinstance(test_ouputs, torch.Tensor):
                        test_ouputs = (test_ouputs,)
                self.ort_validate(onnx_io, test_inputs, test_ouputs)
    
    def ort_validate(self, onnx_io, inputs, outputs):

        inputs, _ = torch.jit._flatten(inputs)
        outputs, _ = torch.jit._flatten(outputs)

        def to_numpy(tensor):
            if tensor.requires_grad:
                return tensor.detach().cpu().numpy()
            else:
                return tensor.cpu().numpy()

        inputs = list(map(to_numpy, inputs))
        outputs = list(map(to_numpy, outputs))


        ort_session = onnxruntime.InferenceSession("fasterrcnn.onnx")#onnx_io.getvalue())
        print(len(inputs), len(ort_session.get_inputs()) )
        #print(ort_session.get_inputs()[0].name, ' ',
        #ort_session.get_inputs()[1].name, ' ', ort_session.get_inputs()[2].name)
        # compute onnxruntime output prediction
        ort_inputs = dict((ort_session.get_inputs()[i].name, inpt) for i, inpt in enumerate(inputs))
        ort_outs = ort_session.run(None, ort_inputs)
        print("ouput ot " , ort_outs[0].shape)
        #print(outputs[0])
        #print(ort_outs[0])

        for i in range(0, len(outputs)):
            import numpy
            print(i)
            try:
                numpy.testing.assert_allclose(outputs[i], ort_outs[i], rtol=1e-03, atol=1e-03)
            except Exception as e:
                print(i, " failed: ", e)
        print("validated")

    def test_nms(self):
        boxes = torch.rand(5, 4)
        boxes[:, 2:] += torch.rand(5, 2)
        scores = torch.randn(5)

        class Module(torch.nn.Module):
            def forward(self, boxes, scores):
                return ops.nms(boxes, scores, 0.5)

        self.run_model(Module(), (boxes, scores))

    def test_roi_pool(self):
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        model = ops.RoIAlign((5, 5), 1, 2)
        self.run_model(model, (x, single_roi))

    def test_roi_align(self):
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        rois = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        pool_h = 5
        pool_w = 5
        model = ops.RoIPool((pool_h, pool_w), 2)
        model.eval()
        self.run_model(model, (x, rois))

    def test_fpn(self):
        model = ops.FeaturePyramidNetwork([10, 20, 30], 5)

        x = OrderedDict()
        x['feat1'] = torch.rand(1, 10, 64, 64)
        x['feat2'] = torch.rand(1, 20, 16, 16)
        x['feat3'] = torch.rand(1, 30, 8, 8)

        x1 = OrderedDict()
        x1['feat1'] = torch.rand(1, 10, 64, 64)
        x1['feat2'] = torch.rand(1, 20, 16, 16)
        x1['feat3'] = torch.rand(1, 30, 8, 8)
        self.run_model(model, (x,), [(x1,)])

    def test_multi_scale_roi_align(self):
        class TransformModule(torch.nn.Module):
            def __init__(self):
                super(TransformModule, self).__init__()
                self.model = ops.MultiScaleRoIAlign(['feat1', 'feat2'], 3, 2)
                self.image_sizes = [(512, 512)]
            def forward(self, input, boxes):
                return self.model(input, boxes, self.image_sizes)
        
        i = OrderedDict()
        i['feat1'] = torch.rand(1, 5, 64, 64)
        i['feat2'] = torch.rand(1, 5, 16, 16)
        boxes = torch.rand(6, 4) * 256
        boxes[:, 2:] += boxes[:, :2]

        #print("\n\n--==++>",boxes.shape)

        i1= OrderedDict()
        i1['feat1'] = torch.rand(1, 5, 64, 64)
        i1['feat2'] = torch.rand(1, 5, 16, 16)
        boxes1 = torch.rand(6, 4) * 256
        boxes1[:, 2:] += boxes1[:, :2]

        self.run_model(TransformModule(), (i, [boxes],),[(i1, [boxes1],)])

    ### test classification models

    ### test segmentation models

    # skip until the new resize op is implemented in ORT
    def test_fcn(self):
        model = models.segmentation.segmentation.fcn_resnet101(pretrained=True)
        
        input = torch.rand(1, 3, 300, 300) 
        input_test = torch.rand(1, 3, 300, 300)
        self.run_model(model, (input,), [(input_test,)])

    # skip until the new resize op is implemented in ORT
    def test_deeplabv3(self):
        model = models.segmentation.segmentation.deeplabv3_resnet101(pretrained=True)
        
        input = torch.rand(1, 3, 300, 300) 
        input_test = torch.rand(1, 3, 300, 300) 
        self.run_model(model, (input,), [(input_test)])

    ### test detection models

    def _init_test_generalized_rcnn_transform(self):
        min_size = 800
        max_size = 1333
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        return transform

    # skip until the new resize op is implemented in ORT
    #python test/test_onnx.py  ONNXExporterTester.test_transform_images
    def test_transform_images(self):
        class TransformModule(torch.nn.Module):
            def __init__(self_module):
                super(TransformModule, self_module).__init__()
                self_module.transform = self._init_test_generalized_rcnn_transform()
               
            def forward(self_module, images): 
                return self_module.transform(images)[0].tensors

        input = [torch.rand(3, 800, 1280), torch.rand(3, 800, 800)]
        input_test = [torch.rand(3, 800, 1280), torch.rand(3, 800, 800)]
        self.run_model(TransformModule(), (input,), [(input_test,)])

    def _init_test_rpn(self):
        #rpn_anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
        #                                       aspect_ratios=((0.5, 1.0, 2.0),))

        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
                anchor_sizes, aspect_ratios)
        out_channels = 256
        rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_pre_nms_top_n = dict(training=2000, testing=1000)
        rpn_post_nms_top_n = dict(training=2000, testing=1000)
        rpn_nms_thresh = 0.7

        rpn = RegionProposalNetwork(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)
        return rpn


    def test_rpn(self):
        class RPNModule(torch.nn.Module):
            def __init__(self_module):
                super(RPNModule, self_module).__init__()
                self_module.transform = self._init_test_generalized_rcnn_transform()
                self_module.backbone = resnet_fpn_backbone('resnet50', True)
                self_module.rpn = self._init_test_rpn()

            def forward(self_module, images): 
                images, targets = self_module.transform(images)
                features = self_module.backbone(images.tensors)
                return self_module.rpn(images, features, targets)

        image_url = "http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg"
        image1 = self.get_image_from_url3(url=image_url)
        image_url2 = "https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png"
        image2 = self.get_image_from_url3(url=image_url2)
        images = [image1] #, image2]
        test_images = [image2]#, image1]

        '''import requests, numpy
        from PIL import Image
        from io import BytesIO
        def fetch_image(url):
            response = requests.get(url)
            return Image.open(BytesIO(response.content)).convert("RGB")

        pil_image = fetch_image(url="http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
        image = torch.from_numpy(numpy.array(pil_image)[:, :, [2, 1, 0]])
        image = torch.nn.functional.upsample(image.permute(2, 0, 1).unsqueeze(0).to(torch.float),
                                            size=(800, 1280)  # size=(960, 1280))
                                            ).to(torch.uint8).squeeze(0).permute(1, 2, 0).to('cpu')
        image = image.permute(2, 0, 1)
        images = [image.float()]#, image.float()]

        pil_image = fetch_image(url="https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png")
        image = torch.from_numpy(numpy.array(pil_image)[:, :, [2, 1, 0]])
        image = torch.nn.functional.upsample(image.permute(2, 0, 1).unsqueeze(0).to(torch.float),
                                            size=(800, 1280)  # size=(960, 1280))
                                            ).to(torch.uint8).squeeze(0).permute(1, 2, 0).to('cpu')
        image = image.permute(2, 0, 1)
        test_images = [image.float()]#, image.float()]'''


        model = RPNModule()

        '''model.eval()
        import time
        start = time.time()
        model(images)
        end = time.time()
        print("TIME : ", end - start)'''
        #return

        self.run_model(model, (images,))#, [(test_images,)])


    def test_roi(self):
        import copy 
        '''class ROIHeadModule(torch.nn.Module):
            def __init__(self_module, m):
                super(ROIHeadModule, self_module).__init__()
                #self.m = copy.deepcopy(m)
                #self.postprocess = m.transform.postprocess
            def forward(self_module, features_l, proposals):                
                original_image_sizes = [torch.Size([1280, 800])]
                image_sizes = [torch.Size([1280, 800])]

                features = OrderedDict()
                for i in range(len(features_l)):
                    if i == 4:
                        features['pool'] = features_l[i]
                    features[i] = features_l[i] 

                detections, losses = self.roi_heads(features, proposals, image_sizes, None)
                #detections = self.postprocess(detections, image_sizes, original_image_sizes)
               
                return detections'''
        import sys            
        image_url = "http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg"
        image1 = self.get_image_from_url3(url=image_url)
        images = [image1]
        
        model = models.detection.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True)
        model.eval()
        
        detection, features, proposals = model(images)
        features = list(features.values())
        #print("---------------->", features[4])

        print("****************test roi :")
        #roi_model = ROIHeadModule(model)
        #model.eval()
        roi_model = model.roi_heads
        roi_model.eval()
        self.run_model(roi_model, (features, proposals))

        
        

    def test_faster_rcnn(self):
        '''image_url = "http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg"
        image1 = self.get_image_from_url(url=image_url)
        #image_url2 = "https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png"
        #image2 = self.get_image_from_url(url=image_url2)
        images = [image1]#, image2]
        #test_images = [image2, image1]'''

        import requests, numpy
        from PIL import Image
        from io import BytesIO
        def fetch_image(url):
            response = requests.get(url)
            return Image.open(BytesIO(response.content)).convert("RGB")

        #pil_image = fetch_image(url="https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png")
        pil_image = fetch_image(url="http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
        image = torch.from_numpy(numpy.array(pil_image)[:, :, [2, 1, 0]])
        image = torch.nn.functional.upsample(image.permute(2, 0, 1).unsqueeze(0).to(torch.float),
                                            size=(800, 1280)  # size=(960, 1280))
                                            ).to(torch.uint8).squeeze(0).permute(1, 2, 0).to('cpu')
        image = image.permute(2, 0, 1)
        images = [image.float()]#, image.float()]

        #pil_image = fetch_image(url="http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
        pil_image = fetch_image(url="https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png")
        image = torch.from_numpy(numpy.array(pil_image)[:, :, [2, 1, 0]])
        image = torch.nn.functional.upsample(image.permute(2, 0, 1).unsqueeze(0).to(torch.float),
                                            size=(800, 1280)  # size=(960, 1280))
                                            ).to(torch.uint8).squeeze(0).permute(1, 2, 0).to('cpu')
        image = image.permute(2, 0, 1)
        test_images = [image.float()]#, image.float()]

        model = models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True)
        self.run_model(model, (images,), [(test_images,)])   



    def get_image_from_url2(self, url):
        import requests
        from PIL import Image
        from io import BytesIO
        import numpy
        from torchvision.transforms.functional import resize
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")


        '''#convert to BGR
        image = torch.from_numpy(numpy.array(image)[:, :, [2, 1, 0]])
        
        #image = resize(image, (800, 1280))
        image = torch.nn.functional.interpolate(
            image.permute(2, 0, 1)
            .unsqueeze(0)
            .to(torch.float),
            size=(800, 1280) 
            ).to(torch.uint8) \
            .squeeze(0) \
            .permute(1, 2, 0)
        image = image.permute(2, 0, 1)'''

        '''img = Image.fromarray(image.numpy(), 'RGB')
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        imgplot = plt.imshow(img)
        plt.show()'''
        
        image = image.resize((800, 1280), Image.BILINEAR)
        # Convert to BGR
        image = numpy.array(image)[:, :, [2, 1, 0]].astype('float32')
        # HWC -> CHW
        image = numpy.transpose(image, [2, 0, 1])


        return torch.from_numpy(image).float()

    def get_image_from_url3(self, url):
        import requests, numpy
        from PIL import Image
        from io import BytesIO
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = image.resize((800, 1280), Image.BILINEAR)
        # RGB to BGR
        #image = numpy.array(image)[:, :, [2, 1, 0]]
        # HWC to CHW
        #image = numpy.transpose(image, [2, 0, 1])

        '''import transforms as T

        def get_transform(train):
            transforms = []
            transforms.append(T.ToTensor())
            return T.Compose(transforms)'''
        from torchvision import transforms
        t = transforms.ToTensor()
        image = t(image)
        return image #torch.from_numpy(numpy.array(image)).float() #get_transform(image) #
        
    def test_mask_rcnn(self):
        image_url = "http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg"
        #image_url = "https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png"
        #tree
        #image_url2 = "https://images.unsplash.com/photo-1469827160215-9d29e96e72f4?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&w=1000&q=80"
        #black
        #image_url = "https://upload.wikimedia.org/wikipedia/commons/1/1e/A_blank_black_picture.jpg"
        #cars
        #image_url2 = "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/10best-cars-group-cropped-1542126037.jpg?crop=0.8148571428571428xw:1xh;center,top&resize=640:*"
        image1 = self.get_image_from_url3(url=image_url)
        image_url2 = "https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png"
        image2 = self.get_image_from_url3(url=image_url2)
        images = [image1]#, image2]
        test_images = [image2]#, image1]
        
        '''import requests, numpy
        from PIL import Image
        from io import BytesIO
        def fetch_image(url):
            response = requests.get(url)
            return Image.open(BytesIO(response.content)).convert("RGB")

        #pil_image = fetch_image(url="https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png")
        pil_image = fetch_image(url="http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
        image = torch.from_numpy(numpy.array(pil_image)[:, :, [2, 1, 0]])
        print("-+++++-------",image.size())
        image = torch.nn.functional.upsample(image.permute(2, 0, 1).unsqueeze(0).to(torch.float),
                                            size=(800, 1280)  # size=(960, 1280))
                                            ).to(torch.uint8).squeeze(0).permute(1, 2, 0).to('cpu')
        image = image.permute(2, 0, 1)
        images = [image.float()]#, image.float()]

        #pil_image = fetch_image(url="http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
        pil_image = fetch_image(url="https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png")
        
        
        print("<<<<<-------------")
        print(torch.from_numpy(numpy.array(pil_image))[0]  )
        print(torch.from_numpy(numpy.array(pil_image)[:, :, [2, 1, 0]])[0] )
        
        print("------------->>>>>")
        image = torch.from_numpy(numpy.array(pil_image)[:, :, [2, 1, 0]])
        
        
    
        
        image = torch.nn.functional.upsample(image.permute(2, 0, 1).unsqueeze(0).to(torch.float),
                                            size=(800, 1280)  # size=(960, 1280))
                                            ).to(torch.uint8).squeeze(0).permute(1, 2, 0).to('cpu')
        image = image.permute(2, 0, 1)
        test_images = [image.float()]#, image.float()]
        
        print(images[0].size(),image1.size())
        numpy.testing.assert_allclose(images[0], image1)'''


        
        #print("dddsds") 
        #return 
        #model = models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True)
        model = models.detection.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True)
        self.run_model(model, (images,), [(test_images,)])   

if __name__ == '__main__':
    unittest.main()
