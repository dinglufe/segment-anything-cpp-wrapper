# Code formatter may move "import segment_anything as SAM" and "import mobile_sam as SAM" to the top
# But this may bring errors after switching models
import torch
import numpy as np
import os

from segment_anything.utils.transforms import ResizeLongestSide

from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

output_names = ['output']

# Generate preprocessing model of Segment-anything in onnx format
# Use original segment-anything, Mobile-SAM or HQ-SAM model
# Each in a separate Python virtual environment

# Uncomment the following lines to generate preprocessing model of Segment-anything
# import segment_anything as SAM
# # Download Segment-anything model "sam_vit_h_4b8939.pth" from https://github.com/facebookresearch/segment-anything#model-checkpoints
# # and change the path below
# checkpoint = 'sam_vit_h_4b8939.pth'
# model_type = 'vit_h'
# output_path = 'models/sam_preprocess.onnx'
# quantize = True

# Mobile-SAM
# # Download Mobile-SAM model "mobile_sam.pt" from https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt
# import mobile_sam as SAM
# checkpoint = 'mobile_sam.pt'
# model_type = 'vit_t'
# output_path = 'models/mobile_sam_preprocess.onnx'
# quantize = False

# HQ-SAM
# # Download Mobile-SAM model "sam_hq_vit_h.pt" from https://github.com/SysCV/sam-hq#model-checkpoints
# # Installation: https://github.com/SysCV/sam-hq#quick-installation-via-pip
import segment_anything as SAM
checkpoint = 'sam_hq_vit_h.pth'
model_type = 'vit_h'
output_path = 'models/sam_hq_preprocess.onnx'
quantize = True
output_names = ['output', 'interm_embeddings']

# Target image size is 1024x720
image_size = (1024, 720)

output_raw_path = output_path
if quantize:
    # The raw directory can be deleted after the quantization is done
    output_name = os.path.basename(output_path).split('.')[0]
    output_raw_path = '{}/{}_raw/{}.onnx'.format(
        os.path.dirname(output_path), output_name, output_name)
os.makedirs(os.path.dirname(output_raw_path), exist_ok=True)

sam = SAM.sam_model_registry[model_type](checkpoint=checkpoint)
sam.to(device='cpu')
transform = ResizeLongestSide(sam.image_encoder.img_size)

image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
input_image = transform.apply_image(image)
input_image_torch = torch.as_tensor(input_image, device='cpu')
input_image_torch = input_image_torch.permute(
    2, 0, 1).contiguous()[None, :, :, :]


class Model(torch.nn.Module):
    def __init__(self, image_size, checkpoint, model_type):
        super().__init__()
        self.sam = SAM.sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device='cpu')
        self.predictor = SAM.SamPredictor(self.sam)
        self.image_size = image_size

    def forward(self, x):
        self.predictor.set_torch_image(x, (self.image_size))
        if 'interm_embeddings' not in output_names:
            return self.predictor.get_image_embedding()
        else:
            return self.predictor.get_image_embedding(), torch.stack(self.predictor.interm_features, dim=0)


model = Model(image_size, checkpoint, model_type)
model_trace = torch.jit.trace(model, input_image_torch)
torch.onnx.export(model_trace, input_image_torch, output_raw_path,
                  input_names=['input'], output_names=output_names)


if quantize:
    quantize_dynamic(
        model_input=output_raw_path,
        model_output=output_path,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
    )
