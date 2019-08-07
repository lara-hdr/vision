#include <torch/script.h>

#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"

using namespace at;

static auto registry =
  torch::jit::RegisterOperators()
    .op("torchvision::nms", &nms)
    .op("torchvision::roi_align_forward(Tensor input, Tensor rois, float spatial_scale, int pooled_height, int pooled_width, int sampling_ratio) -> Tensor", &ROIAlign_forward)
    .op("torchvision::roi_pool_forward", &ROIPool_forward);
