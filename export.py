import os
import time
import json
import argparse
import numpy as np
import torch
import torch.onnx
import segmentation_models_pytorch as smp
import onnx
import onnxruntime as rt


def export(model, x, onnx_model_file, opset=12):
    torch.onnx.export(
            model,
            x,
            onnx_model_file,
            verbose=False,
            opset_version=opset,
            export_params=True,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['output'],
            dynamic_axes={'image': {0: 'batch_size'},
                          'output': {0: 'batch_size'}})
    print(f'Model exported to ONNX: {onnx_model_file}')

    # check onnx model
    onnx_model = onnx.load(onnx_model_file)
    onnx.checker.check_model(onnx_model)

    # compare onnx and pytorch output
    start = time.time()
    torch_out = model(x)
    np_torch_out = torch_out.detach().cpu().numpy()
    print(f'Pytorch model on {x.device}: {time.time()-start:.5f}s')
    
    ort_device = rt.get_device()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort_device == 'GPU' else ['CPUExecutionProvider']
    session = rt.InferenceSession(onnx_model_file, providers=providers)
    np_x = x.cpu().numpy()
    if ort_device == 'GPU':
        ort_out = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: np_x})  # first run is slow on gpu
    start = time.time()
    ort_out = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: np_x})
    print(f'ONNX model on {ort_device}: {time.time()-start:.5f}s')

    np.testing.assert_allclose(np_torch_out, ort_out[0], rtol=1e-03, atol=1e-05)
    print('Torch and ONNX model outputs are similiar')


def main(args):
    model_file_noext = os.path.splitext(args.model_file)[0]
    onnx_model_file = f'{model_file_noext}.onnx'

    config_file = f'{model_file_noext}.json'
    with open(config_file, 'r') as f:
        config = json.load(f)

    # works only with torch (CUDAExecutionProvider is slower due to copying)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load pytorch model
    model_class = getattr(smp, config['decoder_name'])
    model = model_class(encoder_name=config['encoder_name'], encoder_weights=None, in_channels=len(config['bands']), classes=config['num_classes'])
    model.load_state_dict(torch.load(args.model_file))
    model.to(device)
    model.eval()

    # dummy input
    batch_size = 1
    input_shape = (batch_size, len(config['bands']), config['tile_height'], config['tile_width'])
    x = torch.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).float().to(device)
    
    # export
    export(model, x, onnx_model_file, opset=12)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, required=True)
    args = parser.parse_args()

    if not os.path.isfile(args.model_file):
        raise FileNotFoundError(f'{args.model_file} not found.')

    main(args)

