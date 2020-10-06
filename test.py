import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import onnxruntime

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

path = 'LR/baboon.png'
base = osp.splitext(osp.basename(path))[0]

# read images
img = cv2.imread(path, cv2.IMREAD_COLOR)
img = img * 1.0 / 255
img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
img_LR = img.unsqueeze(0)
img_LR = img_LR.to(device)

# with torch.no_grad():
#     # output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()        
#     dummy_input = torch.zeros(1, 3, 224, 224)
#     dummy_input = dummy_input.to(device)
#     output = model(dummy_input)
#     input_names = ['input']
#     output_names = ['output']
#     dynamic_axes= {'input':{2: 'w', 3: 'h'}, 
#     'output':{2: 'w', 3: 'h'}}
#     torch.onnx.export(
#         model,
#         dummy_input, 
#         f"{model_path}.onnx", 
#         verbose=True,
#         input_names=input_names,
#         output_names=output_names,
#         dynamic_axes=dynamic_axes,
#         keep_initializers_as_inputs=True
#     )

ort_session = onnxruntime.InferenceSession("models/RRDB_ESRGAN_x4.pth.onnx")
with torch.no_grad():
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_LR)}
    output = ort_session.run(None, ort_inputs)[0]

output = output[0]
output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
output = (output * 255.0).round()
cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
