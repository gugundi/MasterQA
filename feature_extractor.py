import torch
import torchvision
import numpy as np
import cv2

feature_model = 'resnet101'
split = 'train'
max_images = None
model_ = 'resnet101'
model_stage = 3
#batch_size = 32
batch_size = 1
img_h = img_w = 224
image_dir = "../dataset/tmp"
output_dir = "../processed"

def build_model(img_dir, output_h5_file, img_h, img_w, model, model_stage=3,
                batch_size=64):
    if not hasattr(torchvision.models, model):
        raise ValueError('Invalid model "%s"' % model)
    if not 'resnet' in model:
        raise ValueError('Feature extraction only supports ResNets')
    cnn = getattr(torchvision.models, model)(pretrained=True)
    layers = [cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool]
    for i in range(model_stage):
        name = 'layer%d' % (i+1)
        layers.append(getattr(cnn, name))
    model = torch.nn.Sequential(*layers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    model.eval()
    return model

def run_batch(cur_batch, model):
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,3,1,1) #Comes from CLEVR
    std = np.array([0.229, 0.224, 0.224]).reshape(1,3,1,1) #Comes from CLEVR

    image_batch = np.concatenate(cur_batch, 0).astype(np.float32)
    image_batch = (image_batch / 255.0 - mean) / std
    image_batch = torch.FloatTensor(image_batch).cuda()
    with torch.no_grad():
        image_batch = torch.autograd.Variable(image_batch)

    feats = model(image_batch)
    feats = feats.data.cpu().clone().numpy()

    return feats

model = build_model(image_dir, output_dir, img_h, img_w, model_,
                    model_stage=model_stage, batch_size=batch_size)



img_size = (img_h, img_w)
feat_dset = None
i0 = 0
cur_batch = []
paths = []
input_paths = []

for i, (path, idx) in enumerate(input_paths):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)
    img = img.transpose(2,0,1)[None]
    #Old code for scipy.misc.imread
    #img = imread(path, mode='RGB')
    #img = imresize(img, img_size, interp='bicubic')
    #img = img.transpose(2,0,1)[None]
    cur_batch.append(img)
    paths.append(path)
    if len(cur_batch) == batch_size:
        feats = run_batch(cur_batch, model)
        for j in range(feats.shape[0]):
            torch.save(feats[j], output_dir + paths[j].split('/')[-1])
        i1 = i0 + len(cur_batch)
        i0 = i1
        print('Processed %d / %d images' % (i1, len(input_paths)))
        cur_batch = []
        paths = []
if len(cur_batch) > 0:
    feats = run_batch(cur_batch, model)
    for j in range(feats.shape[0]):
        torch.save(feats[j], output_dir + paths[j].split('/')[-1])
        #Files are saved with .png extension, slighty ambigious.
    i1 = i0 + len(cur_batch)
    print('Processed %d / %d images' % (i1, len(input_paths)))
