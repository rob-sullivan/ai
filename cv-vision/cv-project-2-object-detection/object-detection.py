#ref: https://github.com/amdegroot/ssd.pytorch
# Note: pytorch had updated removing the need for autograd 'Variable', several changes needed to be made
# to this python file, the ssd.py, l2norm.py and box_utils.py


#import libraries
import torch #pytorch to build nn for cv because it has dynamic graphs, so we can efficiently compute gradient composition functions in backward propagation, (i.e update weights through stocastic gradient decent in deep nn hidden layers)
##update the code for new PyTorch
#from torch.autograd import Variable #autograd is a package that is responsible for gradient decent. Variable class converts tensors into pytorch variable into tensor and gradient.
import cv2 #drawing rectangles
from data import BaseTransform, VOC_CLASSES as labelmap #base transform converts images so they will be compatible with the NN VOC_CLASSES is just a dictionary that converts text fields to numbers
from ssd import build_ssd #this is the constructor that builds the single shot classifier. it's a pretrained model trained on 30 / 40 objects such as dogs, trains, horses, etc.
import imageio #converts video to images and then ouput rectangles on the video
#detector not based on open cv but on deep NN

# define detection function
    #frame is the image, net is the ssd nn, transform converts images into the correct format, we don't need to work in grey scale
def detect(frame, net, transform):
    #get height and width 
    height, width = frame.shape[:2] #:2=0,1
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2,0,1) #rbg, ssd was traind on grb.
    #unsqueeze function is just before filtting nn
    #we use unsqueeze to create a fake dimention of the batch
    #it should always be the the index of dimension of the batch (always first dimension)
    #torch tensor of inputs to a torch variable (tensor + gradient)

    #update the code for new PyTorch 
    #x = Variable(x.unsqueeze(0))
    #y = net(x) #output ssd and extract important data into detections

    x = x.unsqueeze(0)
    with torch.no_grad():
        y = net(x)

    """
    ref: https://stackoverflow.com/questions/72504734/what-is-the-purpose-of-with-torch-no-grad
    The requires_grad argument tells PyTorch that we want to be able to calculate the gradients for those values. 
    However, the with torch.no_grad() tells PyTorch to not calculate the gradients, and the program explicitly uses it with most neural networks
    in order to not update the gradients when it is updating the weights as that would affect the back propagation.
    """

    detections = y.data #torch tensor
    #upper left corner width, height, lower right corner width, height
    scale = torch.Tensor([width, height, width, height]) #needed to normalise images between zero and one

    #detections = [batch, # of classes, car-plane-car, # of occurance how many of a class, tuple (score>0.5-found, x0,y0, x1, y1)]
    #need for loop to loop through detection

    for i in range(detections.size(1)):
        j = 0 #class
        while detections[0, i, j, 0]>0.6:
            pt = (detections[0, i, j, 1:] * scale).numpy() #now collects coordinates
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2) #image, xy coordinates, colour(red), thickness
            cv2.putText(frame, 'Biscuit' if(labelmap[i-1]=='dog') else 'Rob' if(labelmap[i-1]=='person') else labelmap[i-1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA) # image, text string, x-y coordinates, font, font scale, colour, font thickness, line type
            j += 1
    return frame

# create SSD neural network
net = build_ssd('test')
#faster than rcsn and yolo
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location = lambda storage, loc: storage))

#create the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

#object detection on video
reader = imageio.get_reader('biscuit.mp4')#funny_dog.mp4
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps=fps)

for i, frame in enumerate(reader):
    frame = detect(frame, net.eval(), transform)
    writer.append_data(frame)
    print(i)
writer.close()


