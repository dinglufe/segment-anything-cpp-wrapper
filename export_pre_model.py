import torch
import numpy as np
import os
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.transforms import ResizeLongestSide
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

# Generate preprocessing model of Segment-anything in onnx format
# Target image size is 1024x720
image_size = (1024, 720)
# Download Segment-anything model "sam_vit_h_4b8939.pth" from https://github.com/facebookresearch/segment-anything#model-checkpoints
# and change the path below
checkpoint = 'sam_vit_h_4b8939.pth'
model_type = 'vit_h'
output_path = 'models/sam_preprocess.onnx'

# The raw directory can be deleted after the quantization is done
output_raw_path = os.path.dirname(output_path) + '/raw/sam_preprocess.onnx'
os.makedirs(os.path.dirname(output_raw_path), exist_ok=True)

sam = sam_model_registry[model_type](checkpoint=checkpoint)
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
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint)
        self.sam.to(device='cpu')
        self.predictor = SamPredictor(self.sam)
        self.image_size = image_size

    def forward(self, x):
        self.predictor.set_torch_image(x, (self.image_size))
        return self.predictor.get_image_embedding()


model = Model(image_size, checkpoint, model_type)
model_trace = torch.jit.trace(model, input_image_torch)
torch.onnx.export(model_trace, input_image_torch, output_raw_path,
                  input_names=['input'], output_names=['output'])

quantize_dynamic(
    model_input=output_raw_path,
    model_output=output_path,
    optimize_model=True,
    per_channel=False,
    reduce_range=False,
    weight_type=QuantType.QUInt8,
)
