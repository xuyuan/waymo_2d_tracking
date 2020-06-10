import torch
from nn import create

net = create('pointdet', ['background', 'pos'], basenet='resnet50')
print(net)

image_size = 800
x = torch.empty((1, 3, image_size, image_size))
net.eval()
#y = net(x)

#pred_hm, pred_wh = y
#print(pred_hm.shape, pred_wh.shape)

output = net.predict(x)
print(output)
