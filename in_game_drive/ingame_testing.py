from torchvision import models
import torch
import torch.nn as nn
from PIL import ImageGrab
import cv2
import torch.nn.functional as F
import albumentations as Al
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
from input_keys import PressKey, ReleaseKey
import time
from Lane_use import process_img
from numpy.linalg import lstsq
from numpy import ones, vstack
from statistics import mean
labels = {0: 'a', 1: 'w', 2: 'd'}#, 3: 's'}

#
# def brightness():
#     x = 0
#     y = 0
#     z = 0
#     screen = np.array(ImageGrab.grab(bbox=(0, 40, 1280, 720)))
#     bright = screen.tolist()
#     # print(bright[0])
#     # print(bright[0][0])
#     # exit()
#     for h in bright:
#         for j in h:
#             x = x + j[0]
#             y = y + j[1]
#             z = z + j[2]
#     # print(x/870400,y/870400,z/870400)
#     x1 = x / 870400
#     y1 = y / 870400
#     z1 = z / 870400
#     outbrightness = (x1 + y1 + z1) / 3
#     return outbrightness


#def ingame_predic():
test_transform = Al.Compose(
    [
        # A.SmallestMaxSize(max_size=160),
        Al.Resize(width = 480, height = 270),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = models.mobilenet_v3_large(pretrained=True)    # mobilenet-v3
# net.classifier[3] = nn.Linear(in_features = 1280, out_features=4)


net = models.efficientnet_b4(pretrained=True)   # efficientnet
net.classifier[1] = nn.Linear(in_features=1792,out_features=3)
net.load_state_dict(torch.load('./models/EFFI_output3.pt', map_location=device))
#
#
# net = models.resnet18(pretrained=True)
# #net.conv1 = nn.Linear(net.conv1.in_features, 64*3*7*7)
# net.fc = nn.Linear(in_features=512,out_features=4)#450개로 분류하잖음
# net.load_state_dict(torch.load('./models/resnet18.pt', map_location=device))

# net = models.densenet121(pretrained=True)
# net.head = nn.Linear(in_features=1024, out_features=3)
# net.load_state_dict(torch.load('./models/densnet.pt', map_location=device))
# net.to(device)


net.to(device)
net.eval()
def ingame_predic():
    while(True):
        with torch.no_grad():



            #i = i + 1
            screen = np.array(ImageGrab.grab(bbox=(0, 40, 1280, 740)))

            #minimap = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)


            # brightness()

            # x = 0
            # y = 0
            # z = 0
            # #screen = np.array(ImageGrab.grab(bbox=(0, 40, 1280, 720)))
            # bright = screen.tolist()
            # # print(bright[0])
            # # print(bright[0][0])
            # # exit()
            # for h in bright:
            #     for j in h:
            #         x = x + j[0]
            #         y = y + j[1]
            #         z = z + j[2]
            # # print(x/870400,y/870400,z/870400)
            # x1 = x / 870400
            # y1 = y / 870400
            # z1 = z / 870400
            # outbrightness = (x1 + y1 + z1) / 3








            #screen = np.array(ImageGrab.grab(bbox=(0, 40, 1280, 720))) # 1024, 768 화면을 받아서 Numpy Array로 전환
            # screen = cv2.imread('./test_image2.jpg') # test image
            # input_image = Image.fromarray(screen)
            input_image = test_transform(image=screen)['image'].float().unsqueeze(0).to(device)
            # print(screen.float())
            # print(type(screen))
            # exit()
            # def test(model, test_loader, device):
            #     model.eval()
            #     correct = 0
            #     total = 0
            #     with torch.no_grad():
            #         for i, (image, labels) in enumerate(test_loader):
            #             image, labels = image.to(device), labels.to(device)
            #             output = model(image)
            #             _, argmax = torch.max(output, 1)
            #             total += image.size(0)
            #             correct += (labels == argmax).sum().item()
            #         acc = correct / total * 100
            #         print("acc for {} image: {:.2f}%".format(total, acc))
            #     model.train()
            #test_aug = test_transform(mode_flag="test")
            #test_dataset = screen, transform=test_aug
            #image, labels = image.to(device), labels.to(device)
            output = net(input_image)
            softmax_result = F.softmax(output)

            new_screen, original_image, m1, m2, minimap,minim = process_img(screen)

            #cv2.imshow('pygta5-7', new_screen)
            #cv2.imshow('pygta5-7-2', cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
            cv2.imshow('mini',cv2.cvtColor(minimap, cv2.COLOR_BGR2RGB))

            if m1 < 0 and m2 < 0:
                print('line talk to me, go right')
                softmax_result = softmax_result + torch.tensor([[-0.0, -0.0, 0.25]]).to(device) #right
            elif m1 > 0 and m2 > 0:
                print('line talk to me, go left')
                softmax_result = softmax_result + torch.tensor([[0.25, -0.0, -0.0]]).to(device) #left
            else:
                print('line talk to me, go foward')
                softmax_result = softmax_result + torch.tensor([[-0.0, 0.35, -0.0]]).to(device)
            print(minim)
            # if minim<0:
            #     print('map talk to me, go right')
            #     softmax_result = softmax_result + torch.tensor([[-0.1, -0.1, 0.2]]).to(device) #right
            # elif minim > 0:
            #     print('map talk to me, go left')
            #     softmax_result = softmax_result + torch.tensor([[0.2, -0.1, -0.1]]).to(device) #left
            # else:
            #     print('map talk to me, go foward')
            #     softmax_result = softmax_result + torch.tensor([[-0.1, 0.2, -0.1]]).to(device)

            top_prob, top_label = torch.topk(softmax_result, 1)
            prob = round(top_prob.item() * 100, 2)
            label = labels.get(int(top_label))

            # print(f'prob: {prob}, label: {label}')



            W = 0x11
            A = 0x1E
            S = 0x1F
            D = 0x20
            T = 0x14

            if (70 < prob) and (label == 'a'):

                PressKey(W)
                PressKey(A)
                ReleaseKey(S)
                ReleaseKey(D)
                # time.sleep(0.7)
                # ReleaseKey(W)
                # ReleaseKey(A)


            elif (70 < prob) and (label == 'w'):
                PressKey(W)
                ReleaseKey(A)
                ReleaseKey(S)
                ReleaseKey(D)
                # time.sleep(0.7)
                # ReleaseKey(W)
            # elif (65 < prob < 80) and (label == 'a'):
            #
            #     PressKey(A)
            #     ReleaseKey(S)
            #     ReleaseKey(D)
            #     ReleaseKey(W)
            #     # #ReleaseKey(W)
            #     # time.sleep(0.3)
            #     # ReleaseKey(A)


            # elif (65 < prob < 80) and (label == 'd'):
            #     PressKey(D)
            #     ReleaseKey(A)
            #     ReleaseKey(S)
            #     ReleaseKey(W)
            #     # time.sleep(0.7)
            #     # ReleaseKey(D)
            elif (70 < prob) and (label == 'd'):
                PressKey(W)
                PressKey(D)
                ReleaseKey(A)
                ReleaseKey(S)
                #ReleaseKey(W)



            # elif (50 < prob) and (label == 's'):
            #     PressKey(S)


            else:
                PressKey(S)
                ReleaseKey(A)
                ReleaseKey(D)
                ReleaseKey(W)


        print(prob, label)
    #return prob, label


if __name__ == '__main__':
    predic_prob, predic_label = ingame_predic()