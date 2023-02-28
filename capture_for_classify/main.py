import numpy as np
import cv2
import time
import os
from grabscreen import grab_screen
from grabkeys import key_check
from PIL import ImageGrab
from PIL import Image
from input_keys import PressKey, ReleaseKey

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
eight = 0x48
four = 0x4B
five = 0x4C
six = 0x4D
def keys_to_output(keys):


    # [A,W,D]
    output = [0,0,0]

    if '4' in keys:
        output[0] = 1
    elif '8' in keys:
        output[1] = 1
    elif '6' in keys:
        output[2] = 1
    # elif '5' in keys:
    #     output[3] = 1


    return output


file_name = 'training_data.npy'

# if os.path.isfile(file_name):
print('File exist, loading previous data!')
training_data = list(np.load(file_name))
time.sleep(1)

# else:
#     print('File does not exist, starting fresh')
#     training_data = []

i=0
def main():

    global training_data
    global i
    # while (True):
    #     x = 0
    #     y = 0
    #     z = 0
    #     i = i+1
    screen = np.array(ImageGrab.grab(bbox=(0, 40, 1280, 720)))
    #     bright = screen.tolist()
    #     # print(bright[0])
    #     # print(bright[0][0])
    #     # exit()
    #     for h in bright:
    #        for j in h:
    #            x = x + j[0]
    #            y = y + j[1]
    #            z = z + j[2]
    #     #print(x/870400,y/870400,z/870400)
    #     x1 = x/870400
    #     y1 = y/870400
    #     z1 = z/870400
    #     brightness = (x1 + y1 + z1)/3

        # screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    screen = cv2.resize(screen, (480, 270))


    keys = key_check()

    output = keys_to_output(keys)

    # print(type(training_data))
    # training_data = list(training_data)
    # training_data.append([screen, output])



    img_2 = Image.fromarray(screen)  # NumPy array to PIL image
    os.makedirs('./w', exist_ok=True)
    os.makedirs('./a', exist_ok=True)
    os.makedirs('./d', exist_ok=True)
    #os.makedirs('./s', exist_ok=True)
    #os.makedirs('./n', exist_ok=True)
    if output == [1, 0, 0]:
        img_2.save(f'./a/aOGA2_{i}.jpg','png')  # save PIL image
        #time.sleep(0.2)
    elif output == [0, 1, 0]:
        img_2.save(f'./w/wOGA2_{i}.jpg','png')
        #time.sleep(0.2)
    elif output == [0, 0, 1]:
        img_2.save(f'./d/dOGA2_{i}.jpg','png')
        #time.sleep(0.2)
    # elif output == [0, 0, 0]:
    #     img_2.save(f'./s/sOGA1_{i}.jpg','png')
    #     #time.sleep(0.2)

    if output == [1, 0, 0]:
        PressKey(A)
    elif output == [0, 1, 0]:
        PressKey(W)
    elif output == [0, 0, 1]:
        PressKey(D)
    # elif output == [0, 0, 0,1]:
    #     PressKey(S)
    else:
        ReleaseKey(W)
        ReleaseKey(A)
        ReleaseKey(D)
        ReleaseKey(S)
    # else:
    #     img_2.save(f'./n/n100_{i}.jpg','png')
    #     time.sleep(0.2)

    #
    #
    # if len(training_data) % 2 == 0:
    #     print(training_data)
    #     training_data = np.array(training_data)#, dtype=object)
    #     print(len(training_data))
    #     np.save(file_name, training_data)
    #

main()