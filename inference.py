import os
import argparse
import numpy as np
import cv2
import torch  # has to be loaded only when we want to use GPU inference, https://stackoverflow.com/questions/75267445/why-does-onnxruntime-fail-to-create-cudaexecutionprovider-in-linuxubuntu-20/75267493#75267493
import onnxruntime as rt


IMG_FORMATS = 'bmp', 'jpeg', 'jpg', 'png', 'tif', 'tiff', 'webp', 'pfm'


def preprocess_img(img, normalize_mean, normalize_std, max_pixel_value=1.):
    assert img.ndim in [3, 4]
    img = (img - np.array(normalize_mean) * max_pixel_value) / (np.array(normalize_std) * max_pixel_value)  # normalize
    if img.ndim == 3:
        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # CHW to BCHW
    else:
        img = img.transpose(0, 3, 2, 1)  # BHWC to BCHW
    img = img.astype(np.float32)
    return img


def load_model(model_file):
    ort_device = rt.get_device()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort_device == 'GPU' else ['CPUExecutionProvider']
    session = rt.InferenceSession(model_file, providers=providers)
    return session


def predict(session, img):
    ort_out = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img})[0]
    return ort_out


def load_image(img_filepath):
    img_ext = os.path.splitext(img_filepath)[1]
    if img_ext in IMG_FORMATS:
        img = cv2.imread(img_filepath)
    elif img_ext == '.npy':
        img = np.load(img_filepath)
    else:
        img = None
    return img


def calc_tiles(img, tile_h, tile_w, bands):
    tiles = []
    img_h, img_w = img.shape[:2]
    tile_y, tile_x = 0, 0
    while tile_y + tile_h <= img_h:
        tiles.append(img[tile_y:tile_y+tile_h, tile_x:tile_x+tile_w, bands])
        tile_x += tile_w
        if tile_x + tile_w > img_w:
            tile_x = 0
            tile_y += tile_h
    return tiles


def get_tiles(source, tile_height, tile_width, bands, normalize_mean, normalize_std, max_pixel_value):
    if os.path.isdir(source):
        for file_name in os.listdir(source):
            file_path = os.path.join(source, file_name)
            if os.path.isfile(file_path):
                img = load_image(file_path)
                if img is not None:
                    tiles = calc_tiles(img, tile_height, tile_width, bands)
                    for i, tile in enumerate(tiles):
                        tile = preprocess_img(tile, normalize_mean, normalize_std, max_pixel_value)
                        yield tile, file_name, i
    elif os.path.isfile(source):
        img = load_image(source)
        if img is None:
            raise Exception(f'{source} isnt in valid format. Supported formats: {IMG_FORMATS}')
        tiles = calc_tiles(img, tile_height, tile_width, bands)
        for i, tile in enumerate(tiles):
            tile = preprocess_img(tile, normalize_mean, normalize_std, max_pixel_value)
            yield tile, os.path.basename(source), i
    else:
        raise FileNotFoundError(f'{source} doesnt exist.')

def main(args):
    binary_threshold = 0.7 if args.norm_out else 0.0
    
    model = load_model(args.model)
    
    tiles = get_tiles(args.source, args.input_height, args.input_width, args.bands, args.normalize_mean, args.normalize_std, args.max_pixel_value)
    for tile, file_name, tile_id in tiles:
        pred = predict(model, tile)
        
        # postprocess
        if args.multiclass:
            #print("here")
            #pred_img = torch.argmax(pred, dim=2)
            pred_img = np.argmax(pred, axis=1)
            pred_img = (pred_img).squeeze() * int(254/2)
        else:  # binary
            pred_img = (pred>binary_threshold).squeeze() * 255
        #print(pred_img.shape)


        
        if args.save or args.show:
            tile_name = f'{os.path.splitext(file_name)[0]}_{tile_id}'
            tile_img = np.copy(tile)
            tile_img = tile_img.squeeze().transpose(1, 2, 0)  # BCHW to HWC
            tile_img = tile_img[..., args.bands_post]  # channels to show
            tile_img = cv2.normalize(tile_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # normalize to 0-255 values
            pred_img = pred_img.astype(np.uint8)
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
            show_img = np.concatenate((tile_img, pred_img), axis=1)

            #mask_path = "data/datasets/sentinel/masks/" + args.source.split("/")[-1]
            #print(mask_path, tile_id)
            
            if args.mask_comparison:
                mask_path = "data/datasets/sentinel/masks/" + args.source.split("/")[-1]
                dim_x = tile_id % 2
                dim_y = tile_id // 2
                res = 496
                mask_img = np.load(mask_path)[dim_y*res:(dim_y+1)*res, dim_x*res:(dim_x+1)*res, :]
                mask_img = np.argmax(mask_img, axis = 2) * 127
                mask_img = mask_img.astype(np.uint8)
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
                print(mask_path, tile_id)
                #mask_name = f'{os.path.splitext(file_name)[0]}_{tile_id}'
                #mask_img = show_img
                show_img = np.concatenate((show_img, mask_img), axis=1)

        if args.save:
            cv2.imwrite(os.path.join(args.out_folder, tile_name + '.png'), show_img)
        
        if args.show:
            cv2.imshow(tile_name, show_img)
            key = cv2.waitKey(0)
            if key == 27:
                cv2.destroyAllWindows()
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='ONNX model path.')
    parser.add_argument('--source',type=str, default='data/', help='Input source, can be folder with image or numpy files or just a single file.')
    parser.add_argument('--input_width', '--input_w', type=int, default=496)
    parser.add_argument('--input_height', '--input_h', type=int, default=496)
    parser.add_argument('--bands', nargs='+', type=int, default=[3, 2, 1, 7], help='Image bands from source images to model input. Usage: --bands 3 2 1 7')
    parser.add_argument('--bands_post', nargs='+', type=int, default=[2, 1, 0], help='Image bands from model input to show/save (should be in BGR). Usage: --bands 2 1 0')
    parser.add_argument('--max_pixel_value', type=float, default=1., help='Maximum possible pixel value of input images.')
    parser.add_argument('--normalize_mean', nargs='+', type=float, default=[0.485, 0.456, 0.406, 0.388], help='Mean values for each channel for normalization.')
    parser.add_argument('--normalize_std', nargs='+', type=float, default=[0.229, 0.224, 0.225, 0.264], help='Std values for each channel for normalization.')
    parser.add_argument('--multiclass', action='store_true', help='Model multi-class output.')
    parser.add_argument('--norm_out', action='store_true', help='Normalized model output (sigmoid, softmax).')
    parser.add_argument('--save', action='store_true', help='Save predicted mask as image.')
    parser.add_argument('--show', action='store_true', help='Show predicted masks.')
    parser.add_argument('--mask_comparison', action='store_true', help='Show actual masks')
    parser.add_argument('--out_folder', type=str, default='data/preds/', help='Folder for predicted masks.')
    args = parser.parse_args()
   
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f'{args.model_file} not found.')
    
    if args.save:
        os.makedirs(args.out_folder, exist_ok=True)
    
    main(args)

