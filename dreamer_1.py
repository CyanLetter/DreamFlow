# imports and basic notebook setup
from cStringIO import StringIO
import argparse
import os, os.path
import errno
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format

import caffe
import cv2

def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

# Gradient step function

def objective_L2(dst):
    dst.diff[:] = dst.data 

#objective for guided dreaming
def objective_guide(dst,guide_features):
    x = dst.data[0].copy()
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

#from https://github.com/jrosebr1/bat-country/blob/master/batcountry/batcountry.py
def prepare_guide(net, image, end="inception_4c/output", maxW=224, maxH=224):
        # grab dimensions of input image
        (w, h) = image.size

        # GoogLeNet was trained on images with maximum width and heights
        # of 224 pixels -- if either dimension is larger than 224 pixels,
        # then we'll need to do some resizing
        if h > maxH or w > maxW:
            # resize based on width
            if w > h:
                r = maxW / float(w)

            # resize based on height
            else:
                r = maxH / float(h)

            # resize the image
            (nW, nH) = (int(r * w), int(r * h))
            image = np.float32(image.resize((nW, nH), PIL.Image.BILINEAR))

        (src, dst) = (net.blobs["data"], net.blobs[end])
        src.reshape(1, 3, nH, nW)
        src.data[0] = preprocess(net, image)
        net.forward(end=end)
        guide_features = dst.data[0].copy()

        return guide_features


def make_step(net, step_size=1.5, end='inception_4c/output', 
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
            
    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
            
    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

# Ascend through octaves

def deepdream(net, base_img, iter_n=4, octave_n=5, octave_scale=1.4, end='inception_5a/pool_proj', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)
            
            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            showarray(vis)
            print octave, i, end, vis.shape
            clear_output(wait=True)
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

# --------------
# Guided Dreaming
# --------------
def make_step_guided(net, step_size=1.5, end='inception_4c/output',
              jitter=32, clip=True, objective_fn=objective_guide, **objective_params):
    '''Basic gradient ascent step.'''

    #if objective_fn is None:
    #    objective_fn = objective_L2

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    net.forward(end=end)
    objective_fn(dst, **objective_params)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream_guided(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, objective_fn=objective_guide, **step_params):

    #if objective_fn is None:
    #    objective_fn = objective_L2

    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step_guided(net, end=end, clip=clip, objective_fn=objective_fn, **step_params)

            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            showarray(vis)
            print octave, i, end, vis.shape
            clear_output(wait=True)

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

# Custom morph and blend

def morphPicture(filename1,filename2,blend):
    img1 = PIL.Image.open(filename1)
    img2 = PIL.Image.open(filename2)
    return PIL.Image.blend(img1, img2, blend)

def make_sure_path_exists(path):
    '''
    make sure input and output directory exist, if not create them.
    If another error (permission denied) throw an error.
    '''
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def main(input, output, model_path, model_name, octaves, octave_scale, iterations, jitter, stepsize, blend, layers, guide_image, start_frame, end_frame, verbose):

    make_sure_path_exists(input)
    make_sure_path_exists(output)

     # let max nr of frames
    nrframes =len([name for name in os.listdir(input) if os.path.isfile(os.path.join(input, name))])
    if nrframes == 0:
        print("no frames to process found")
        sys.exit(0)

    if octaves is None: octaves = 5
    if octave_scale is None: octave_scale = 1.4
    if iterations is None: iterations = 4
    if jitter is None: jitter = 32
    if stepsize is None: stepsize = 1.5
    if blend is None: blend = 0.5 #can be nr (constant), random, or loop
    if verbose is None: verbose = 1
    if layers is None: layers = 'inception_5a/pool_proj' #['inception_4c/output']
    if start_frame is None:
    	frame_i = 1
    else:
        frame_i = int(start_frame)
    if not end_frame is None:
    	nrframes = int(end_frame)+1
    else:
        nrframes = nrframes+1

    # If your GPU supports CUDA and Caffe was built with CUDA support,
    # uncomment the following to run Caffe operations on the GPU.
    caffe.set_mode_gpu()
    # caffe.set_device(0) # select GPU device if multiple devices exist

    # Loading DNN Model
    net_fn   = model_path + 'deploy.prototxt'
    param_fn = model_path + model_name

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier('tmp.prototxt', param_fn,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

    if verbose == 3:
        from IPython.display import clear_output, Image, display
        print("display turned on")
    img = np.float32(PIL.Image.open(input + '/%08d.png' % (frame_i) ))
    h,w,c = img.shape

    #Choosing between normal dreaming, and guided dreaming
    if guide_image is None:
        hallu = deepdream(net, img, iter_n = iterations, step_size = stepsize, octave_n = octaves, octave_scale = octave_scale, jitter=jitter, end = layers)
    else:
        guide = np.float32(PIL.Image.open(guide_image))
        print('Setting up Guide with selected image')
        guide_features = prepare_guide(net,PIL.Image.open(guide_image), end=layers)
        hallu = deepdream_guided(net, img, iter_n = iterations, step_size = stepsize, octave_n = octaves, octave_scale = octave_scale, jitter=jitter, end = layers, objective_fn=objective_guide, guide_features=guide_features)

    np.clip(hallu, 0, 255, out=hallu)
    PIL.Image.fromarray(np.uint8(hallu)).save(output + '/%08d.png' % (frame_i) )
    grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blend_forward = True
    blend_at = 0
    blend_step = 0.02
    for i in xrange(frame_i, nrframes):
        previousImg = img
        previousGrayImg = grayImg    
        img = np.float32(PIL.Image.open(input + '/%08d.png' % (i+1) ))
        grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previousGrayImg, grayImg, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        halludiff = hallu - previousImg
        halludiff = cv2.remap(halludiff, flow, None, cv2.INTER_LINEAR)   
        hallu = img + halludiff
        if guide_image is None:
            hallu = deepdream(net, hallu, iter_n = iterations, step_size = stepsize, octave_n = octaves, octave_scale = octave_scale, jitter=jitter, end = layers)
        else:
            guide = np.float32(PIL.Image.open(guide_image))
            print('Setting up Guide with selected image')
            guide_features = prepare_guide(net,PIL.Image.open(guide_image), end=layers)
            hallu = deepdream_guided(net, hallu, iter_n = iterations, step_size = stepsize, octave_n = octaves, octave_scale = octave_scale, jitter=jitter, end = layers, objective_fn=objective_guide, guide_features=guide_features)
        np.clip(hallu, 0, 255, out=hallu)
        PIL.Image.fromarray(np.uint8(hallu)).save(output + '/%08d.png' % (i+1))

        # if blend_at > 1 - blend_step: blend_forward = False
        # elif blend_at <= 0.5: blend_forward = True
        # if blend_forward: blend_at += blend_step
        # else: blend_at -= blend_step
        # blendval = blend_at
        blendval = 0.5
    img = morphPicture(input + '/%08d.png' % (i+1), output + '/%08d.png' % (i), blendval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dreaming in videos.')
    parser.add_argument(
        '-i','--input',
        help='Input directory where extracted frames are stored',
        required=True)
    parser.add_argument(
        '-o','--output',
        help='Output directory where processed frames are to be stored',
        required=True)
    parser.add_argument(
        '-t', '--model_path',
        dest='model_path',
        default='../caffe/models/bvlc_googlenet/',
        help='Model directory to use')
    parser.add_argument(
        '-m', '--model_name',
        dest='model_name',
        default='bvlc_googlenet.caffemodel',
        help='Caffe Model name to use')
    parser.add_argument(
        '-oct','--octaves',
        type=int,
        required=False,
        help='Octaves. Default: 4')
    parser.add_argument(
        '-octs','--octavescale',
        type=float,
        required=False,
        help='Octave Scale. Default: 1.4',)
    parser.add_argument(
        '-itr','--iterations',
        type=int,
        required=False,
        help='Iterations. Default: 10')
    parser.add_argument(
        '-j','--jitter',
        type=int,
        required=False,
        help='Jitter. Default: 32')
    parser.add_argument(
        '-s','--stepsize',
        type=float,
        required=False,
        help='Step Size. Default: 1.5')
    parser.add_argument(
        '-b','--blend',
        type=str,
        required=False,
        help='Blend Amount. Default: "0.5" (constant), or "loop" (0.5-1.0), or "random"')
    parser.add_argument(
        '-l','--layers',
        type=str,
        required=False,
        help='Layer to use ie [inception_4c/output]')
    parser.add_argument(
        '-v', '--verbose',
        type=int,
        required=False,
        help="verbosity [0-3]")
    parser.add_argument(
    	'-gi', '--guide_image',
    	required=False,
    	help="path to guide image")
    parser.add_argument(
    	'-sf', '--start_frame',
        type=int,
    	required=False,
    	help="starting frame nr")
    parser.add_argument(
    	'-ef', '--end_frame',
        type=int,
    	required=False,
    	help="end frame nr")

    args = parser.parse_args()

    if not args.model_path[-1] == '/':
        args.model_path = args.model_path + '/'

    if not os.path.exists(args.model_path):
        print("Model directory not found")
        print("Please set the model_path to a correct caffe model directory")
        sys.exit(0)

    model = os.path.join(args.model_path, args.model_name)

    if not os.path.exists(model):
        print("Model not found")
        print("Please set the model_name to a correct caffe model")
        print("or download one with ./caffe_dir/scripts/download_model_binary.py caffe_dir/models/bvlc_googlenet")
        sys.exit(0)
        
    else:
        main(args.input, args.output, args.model_path, args.model_name, args.octaves, args.octavescale, args.iterations, args.jitter, args.stepsize, args.blend, args.layers, args.guide_image, args.start_frame, args.end_frame, args.verbose)