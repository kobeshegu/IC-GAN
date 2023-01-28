from pytorch_pretrained_biggan import BigGAN, convert_to_images, one_hot_from_names, utils

%cd /content/ic_gan/
import sys
import os
sys.path[0] = '/content/ic_gan/inference'
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import torch 

import numpy as np
import torch
import torchvision
import sys
torch.manual_seed(np.random.randint(sys.maxsize))
import imageio
from IPython.display import HTML, Image, clear_output
from PIL import Image as Image_PIL
from scipy.stats import truncnorm, dirichlet
from torch import nn
from nltk.corpus import wordnet as wn
from base64 import b64encode
from time import time
import cma
from cma.sigma_adaptation import CMAAdaptSigmaCSA, CMAAdaptSigmaTPA
import warnings
warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)
import torchvision.transforms as transforms
import inference.utils as inference_utils
import data_utils.utils as data_utils
from BigGAN_PyTorch.BigGAN import Generator as generator
import sklearn.metrics

def replace_to_inplace_relu(model): #saves memory; from https://github.com/minyoungg/pix2latent/blob/master/pix2latent/model/biggan.py
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.ReLU(inplace=False))
        else:
            replace_to_inplace_relu(child)
    return
    
def save(out,name=None, torch_format=True):
  if torch_format:
    with torch.no_grad():
      out = out.cpu().numpy()
  img = convert_to_images(out)[0]
  if name:
    imageio.imwrite(name, np.asarray(img))
  return img

hist = []
def checkin(i, best_ind, total_losses, losses, regs, out, noise=None, emb=None, probs=None):
  global sample_num, hist
  name = None
  if save_every and i%save_every==0:
    name = '/content/output/frame_%05d.jpg'%sample_num
  pil_image = save(out, name)
  vals0 = [sample_num, i, total_losses[best_ind], losses[best_ind], regs[best_ind], np.mean(total_losses), np.mean(losses), np.mean(regs), np.std(total_losses), np.std(losses), np.std(regs)]
  stats = 'sample=%d iter=%d best: total=%.2f cos=%.2f reg=%.3f avg: total=%.2f cos=%.2f reg=%.3f std: total=%.2f cos=%.2f reg=%.3f'%tuple(vals0)
  vals1 = []
  if noise is not None:
    vals1 = [np.mean(noise), np.std(noise)]
    stats += ' noise: avg=%.2f std=%.3f'%tuple(vals1)
  vals2 = []
  if emb is not None:
    vals2 = [emb.mean(),emb.std()]
    stats += ' emb: avg=%.2f std=%.3f'%tuple(vals2)
  elif probs:
    best = probs[best_ind]
    inds = np.argsort(best)[::-1]
    probs = np.array(probs)
    vals2 = [ind2name[inds[0]], best[inds[0]], ind2name[inds[1]], best[inds[1]], ind2name[inds[2]], best[inds[2]], np.sum(probs >= 0.5)/pop_size,np.sum(probs >= 0.3)/pop_size,np.sum(probs >= 0.1)/pop_size]
    stats += ' 1st=%s(%.2f) 2nd=%s(%.2f) 3rd=%s(%.2f) components: >=0.5:%.0f, >=0.3:%.0f, >=0.1:%.0f'%tuple(vals2)
  hist.append(vals0+vals1+vals2)
  if show_every and i%show_every==0:
    clear_output()
    display(pil_image)  
  print(stats)
  sample_num += 1

def load_icgan(experiment_name, root_ = '/content'):
  root = os.path.join(root_, experiment_name)
  config = torch.load("%s/%s.pth" %
                      (root, "state_dict_best0"))['config']

  config["weights_root"] = root_
  config["model_backbone"] = 'biggan'
  config["experiment_name"] = experiment_name
  G, config = inference_utils.load_model_inference(config)
  G.cuda()
  G.eval()
  return G

def get_output(noise_vector, input_label, input_features):  
  if stochastic_truncation: #https://arxiv.org/abs/1702.04782
    with torch.no_grad():
      trunc_indices = noise_vector.abs() > 2*truncation
      size = torch.count_nonzero(trunc_indices).cpu().numpy()
      trunc = truncnorm.rvs(-2*truncation, 2*truncation, size=(1,size)).astype(np.float32)
      noise_vector.data[trunc_indices] = torch.tensor(trunc, requires_grad=requires_grad, device='cuda')
  else:
    noise_vector = noise_vector.clamp(-2*truncation, 2*truncation)
  if input_label is not None:
    input_label = torch.LongTensor(input_label)
  else:
    input_label = None

  out = model(noise_vector, input_label.cuda() if input_label is not None else None, input_features.cuda() if input_features is not None else None)
  
  if channels==1:
    out = out.mean(dim=1, keepdim=True)
    out = out.repeat(1,3,1,1)
  return out

def normality_loss(vec): #https://arxiv.org/abs/1903.00925
    mu2 = vec.mean().square()
    sigma2 = vec.var()
    return mu2+sigma2-torch.log(sigma2)-1
    

def load_generative_model(gen_model, last_gen_model, experiment_name, model):
  # Load generative model
  if gen_model != last_gen_model:
    model = load_icgan(experiment_name, root_ = '/content')
    last_gen_model = gen_model
  return model, last_gen_model

def load_feature_extractor(gen_model, last_feature_extractor, feature_extractor):
  # Load feature extractor to obtain instance features
  feat_ext_name = 'classification' if gen_model == 'cc_icgan' else 'selfsupervised'
  if last_feature_extractor != feat_ext_name:
    if feat_ext_name == 'classification':
      feat_ext_path = ''
    else:
      !curl -L -o /content/swav_pretrained.pth.tar -C - 'https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar' 
      feat_ext_path = '/content/swav_pretrained.pth.tar'
    last_feature_extractor = feat_ext_name
    feature_extractor = data_utils.load_pretrained_feature_extractor(feat_ext_path, feature_extractor = feat_ext_name)
    feature_extractor.eval()
  return feature_extractor, last_feature_extractor

norm_mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
norm_std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def preprocess_input_image(input_image_path, size): 
  pil_image = Image_PIL.open(input_image_path).convert('RGB')
  transform_list =  transforms.Compose([data_utils.CenterCropLongEdge(), transforms.Resize((size,size)), transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
  tensor_image = transform_list(pil_image)
  tensor_image = torch.nn.functional.interpolate(tensor_image.unsqueeze(0), 224, mode="bicubic", align_corners=True)
  return tensor_image

def preprocess_generated_image(image): 
  transform_list =  transforms.Normalize(norm_mean, norm_std)
  image = transform_list(image*0.5 + 0.5)
  image = torch.nn.functional.interpolate(image, 224, mode="bicubic", align_corners=True)
  return image

last_gen_model = None
last_feature_extractor = None
model = None
feature_extractor = None

#@title Generate images with IC-GAN!
#@markdown 1. Select type of IC-GAN model with **gen_model**: "icgan" is conditioned on an instance; "cc_icgan" is conditioned on both instance and a class index.
#@markdown 1. Select which instance to condition on, following one of the following options:
#@markdown     1. **input_image_instance** is the path to an input image, from either the mounted Google Drive or a manually uploaded image to "Files" (left part of the screen).
#@markdown     1. **input_feature_index** write an integer from 0 to 1000. This will change the instance conditioning and therefore the style and semantics of the generated images. This will select one of the 1000 instance features pre-selected from ImageNet using k-means.
#@markdown 1. For **class_index** (only valid for gen_model="cc_icgan") write an integer from 0 to 1000. This will change the ImageNet class to condition on. Consult [this link](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) for a correspondence between class name and indexes.
#@markdown 1. **num_samples_ranked** (default=16) indicates the number of generated images to output in a mosaic. These generated images are the ones that scored a higher cosine similarity with the conditioning instance, out of **num_samples_total** (default=160) generated samples. Increasing "num_samples_total" will result in higher run times, but more generated images to choose the top "num_samples_ranked" from, and therefore higher chance of better image quality. Reducing "num_samples_total" too much could result in generated images with poorer visual quality. A ratio of 10:1 (num_samples_total:num_samples_ranked) is recommended.
#@markdown 1. Vary **truncation** (default=0.7) from 0 to 1 to apply the [truncation trick](https://arxiv.org/abs/1809.11096). Truncation=1 will provide more diverse but possibly poorer quality images. Trucation values between 0.7 and 0.9 seem to empirically work well.
#@markdown 1. **seed**=0 means no seed.

gen_model = 'icgan' #@param ['icgan', 'cc_icgan']
if gen_model == 'icgan':  
  experiment_name = 'icgan_biggan_imagenet_res256'
else:
  experiment_name = 'cc_icgan_biggan_imagenet_res256'
#last_gen_model = experiment_name
size = '256'
input_image_instance = ""#@param {type:"string"}
input_feature_index =   3#@param {type:'integer'}
class_index =   538#@param {type:'integer'}
num_samples_ranked =   16#@param {type:'integer'}
num_samples_total =    160#@param {type:'integer'}
truncation =  0.7#@param {type:'number'}
stochastic_truncation = False #@param {type:'boolean'}
download_file = True #@param {type:'boolean'}
seed =  50#@param {type:'number'}
if seed == 0:
  seed = None
noise_size = 128
class_size = 1000
channels = 3
batch_size = 4
if gen_model == 'icgan':
  class_index = None
if 'biggan' in gen_model:
  input_feature_index = None
  input_image_instance = None

assert(num_samples_ranked <=num_samples_total)
import numpy as np
state = None if not seed else np.random.RandomState(seed)
np.random.seed(seed)

feature_extractor_name = 'classification' if gen_model == 'cc_icgan' else 'selfsupervised'

# Load feature extractor (outlier filtering and optionally input image feature extraction)
feature_extractor, last_feature_extractor = load_feature_extractor(gen_model, last_feature_extractor, feature_extractor)
# Load features 
if input_image_instance not in ['None', ""]:
  print('Obtaining instance features from input image!')
  input_feature_index = None
  input_image_tensor = preprocess_input_image(input_image_instance, int(size))
  print('Displaying instance conditioning:')
  display(convert_to_images(((input_image_tensor*norm_std + norm_mean)-0.5) / 0.5)[0])
  with torch.no_grad():
    input_features, _ = feature_extractor(input_image_tensor.cuda())
  input_features/=torch.linalg.norm(input_features,dim=-1, keepdims=True)
elif input_feature_index is not None:
  print('Selecting an instance from pre-extracted vectors!')
  input_features = np.load('/content/stored_instances/imagenet_res'+str(size)+'_rn50_'+feature_extractor_name+'_kmeans_k1000_instance_features.npy', allow_pickle=True).item()["instance_features"][input_feature_index:input_feature_index+1]
else:
  input_features = None

# Load generative model
model, last_gen_model = load_generative_model(gen_model, last_gen_model, experiment_name, model)
# Prepare other variables
name_file = '%s_class_index%s_instance_index%s'%(gen_model, str(class_index) if class_index is not None else 'None', str(input_feature_index) if input_feature_index is not None else 'None')

!rm -rf /content/output
!mkdir -p /content/output

replace_to_inplace_relu(model)
ind2name = {index: wn.of2ss('%08dn'%offset).lemma_names()[0] for offset, index in utils.IMAGENET.items()}

from google.colab import files, output

eps = 1e-8

# Create noise, instance and class vector
noise_vector = truncnorm.rvs(-2*truncation, 2*truncation, size=(num_samples_total, noise_size), random_state=state).astype(np.float32) #see https://github.com/tensorflow/hub/issues/214
noise_vector = torch.tensor(noise_vector, requires_grad=False, device='cuda')
if input_features is not None:
  instance_vector = torch.tensor(input_features, requires_grad=False, device='cuda').repeat(num_samples_total, 1)
else: 
  instance_vector = None
if class_index is not None:
  print('Conditioning on class: ', ind2name[class_index])
  input_label = torch.LongTensor([class_index]*num_samples_total)
else:
  input_label = None
if input_feature_index is not None:
  print('Conditioning on instance with index: ', input_feature_index)

size = int(size)
all_outs, all_dists = [], []
for i_bs in range(num_samples_total//batch_size+1):
  start = i_bs*batch_size
  end = min(start+batch_size, num_samples_total)
  if start == end:
    break
  out = get_output(noise_vector[start:end], input_label[start:end] if input_label is not None else None, instance_vector[start:end] if instance_vector is not None else None)

  if instance_vector is not None:
    # Get features from generated images + feature extractor
    out_ = preprocess_generated_image(out)
    with torch.no_grad():
      out_features, _ = feature_extractor(out_.cuda())
    out_features/=torch.linalg.norm(out_features,dim=-1, keepdims=True)
    dists = sklearn.metrics.pairwise_distances(
            out_features.cpu(), instance_vector[start:end].cpu(), metric="euclidean", n_jobs=-1)
    all_dists.append(np.diagonal(dists))
    all_outs.append(out.detach().cpu())
  del (out)
all_outs = torch.cat(all_outs)
all_dists = np.concatenate(all_dists)

# Order samples by distance to conditioning feature vector and select only num_samples_ranked images
selected_idxs =np.argsort(all_dists)[:num_samples_ranked]
#print('All distances re-ordered ', np.sort(all_dists))
# Create figure                
row_i, col_i, i_im = 0, 0, 0
all_images_mosaic = np.zeros((3,size*(int(np.sqrt(num_samples_ranked))), size*(int(np.sqrt(num_samples_ranked)))))
for j in selected_idxs:
  all_images_mosaic[:,row_i*size:row_i*size+size, col_i*size:col_i*size+size] = all_outs[j]
  if row_i == int(np.sqrt(num_samples_ranked))-1:
    row_i = 0
    if col_i == int(np.sqrt(num_samples_ranked))-1:
      col_i = 0
    else:
      col_i +=1
  else:
    row_i+=1
  i_im +=1

name = '/content/%s_seed%i.png'%(name_file,seed if seed is not None else -1)
pil_image = save(all_images_mosaic[np.newaxis,...],name, torch_format=False)  
print('Displaying generated images')
display(pil_image)

if download_file:
  files.download(name)