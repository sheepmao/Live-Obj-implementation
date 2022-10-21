from __future__ import division
import time
import traceback
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random
import pickle as pkl
import argparse
import csv
import threading
import _thread
from multiprocessing import Process
import math
from load_data import*



ShrinkV=3
bufferLen=4
UserView=200
lenU = 120          #video length used (seconds)

def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
    parser.add_argument("--Tracking",dest='Tracking',help="Apply tracking method",default = True)
    parser.add_argument("--RL",dest='RL',help="Apply Reinfore learning method",default = True)
    parser.add_argument("--video", dest='video', help=
    "Video to run detection upon",
                        default="video.avi", type=str)
    parser.add_argument("--dataset", dest="dataset", help="Dataset on which the network has been trained",
                        default="pascal")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    return parser.parse_args()



def RL_Qtable(tmp_TileStatus,Pre_TileStatus,Qtable,status_action,TileStatus):
    # # Find max value in Q[idx][status][action] to choose action and update
    #print("table in function",Qtable)\

    for idx in range(100):
        action = 0
        # Previous Not selected
        if Pre_TileStatus[idx] == 0 :
            state = {0:0,1:1,2:2,3:3}    
        else :
            # Previous Selected
            state = {0:4,1:5,2:6,3:7}
        # Look up Q_table to decide if predicted or not predicted

        i = torch.argmin(Q_table[idx],dim=1)  
        # if Qtable[idx, tmp_TileStatus[idx], 0] < Qtable[idx, tmp_TileStatus[idx],1] :
        # # predicted have high value
        #     action = 1
        action = torch.argmax(Q_table[idx,state[tmp_TileStatus[idx]]])
        #updateTiles
        # print("Pretile: {}  Tile:{} action{} state{} status{}".format(Pre_TileStatus[idx],idx,action,state[tmp_TileStatus[idx]],action*ShrinkV))
        # print("idx {} in function min{}:  ".format(idx,i))
        # print("action0: {} , action1: {} \n".format(Qtable[idx][tmp_TileStatus[idx]][0], Qtable[idx][tmp_TileStatus[idx]][1]))
        TileStatus[idx] += action * ShrinkV
        if TileStatus[idx] >3:
            TileStatus[idx] =3
        status_action.append((state[tmp_TileStatus[idx]],action))

def CheckIFUserViewCovered (TileStatus,w,h,tileNo,X,Y,XL,YL,state_action,Q_table):
    Flag = True
    alpha = 0.5
    tmp = [0]*tileNo*tileNo
    wd = w / tileNo
    hd = h / tileNo
    IL = math.floor(X / wd)
    IH = math.ceil(XL / wd)
    JL = math.floor(Y / hd)
    JH = math.ceil(YL / hd)
    if JH>10:
        JH=10
    if IH>10:
        IH=10
    for i in range(IL, IH ):
        for j in range(JL, JH ):
            idx = i + j * tileNo
            tmp [idx] = 1
            state,action = state_action[idx]
            if TileStatus[idx] == 0:
                if action != 1: # predict fail, this tile should be predicted
                    # Update Q_table value
                    Q_table[idx][state][action] = Q_table[idx][state][action] * alpha + -2
                    #print("update fail in usr viewport   Tile NO: {} state: {} action:{}".format(idx,state,action))
                    if Q_table[idx][state][action] <-3:
                        Q_table[idx][state][action] = -3
                Flag = False
            else:
                 # Update Q_table value
                Q_table[idx][state][action] =  Q_table[idx][state][action] * alpha + 3
                #print("update success in usr viewport Tile NO: {} state: {} action:{}".format(idx,state,action))
                if  Q_table[idx][state][action] > 5:
                    Q_table[idx][state][action] = 5
    # outside the viewport tiles   
    for idx in range (tileNo * tileNo):
        if tmp[idx] != 1 : # outside the viewport tiles
            state,action = state_action[idx]
            if action == 1: # predict fail, this tile should not be predicted
            # Update Q_table value
                Q_table[idx][state][action] = Q_table[idx][state][action] * alpha + -2
                #Q_table[idx][state][action] = -3
                #print("update fail outside usr viewport  Tile NO: {} state: {} action:{}".format(idx,state,action))
                if Q_table[idx][state][action] <-3:
                    Q_table[idx][state][action] = -3
            elif action == 0:
                #print("update success outside usr viewport  Tile NO: {} state: {} action:{}".format(idx,state,action))
                 # Update Q_table value
                Q_table[idx][state][action] =  Q_table[idx][state][action] * alpha + 3
                if  Q_table[idx][state][action] > 5:
                    Q_table[idx][state][action] = 5
                
    #print(Q_table[1:5])
    return Flag

def RL_UpdateQtable(Q_table,value,current,previous,action,TileNO):
    ''' Q-table(state,action)
    state:                                                   action: 
    0: no objects and viewport,not previous selected            1:select     0:not select
    1: viewport only,previous not selected                      1:select     0:not select
    2: objects only,previous not selected                       1:select     0:not select
    3: objects and viewport,not previous selected               1:select     0:not select
    4: no objects or viewport, previous selected                1:select     0:not select
    5: viewport only,previous previous selected                 1:select     0:not select
    6: objects only,previous previous selected                  1:select     0:not select
    7: objects and viewport,not previous selected               1:select     0:not select
    '''


    
def FromObjectToUpdatTile(Namelist, CorUPleft, CorDownRig, TileStatus,height, width,tileNo):
    L = len(Namelist)
    if L>10:
        K=50
    else:
        K=100
    for i in range(0, L):
        X=CorUPleft[i][0]
        Y=CorUPleft[i][1]
        XL=CorDownRig[i][0]
        YL=CorDownRig[i][1]
        if XL-X<150 or YL-Y<150:
            X = X - K
            XL = XL + K
            YL=YL+150
            Y=Y-150
            if X<0:
                X=0
            if Y<=0:
                Y=0
            if XL>width:
                XL=width
            if YL>height:
                YL=height
        UpdateTileStatuesBasedOnObject(TileStatus, width, height, tileNo, X, Y, XL, YL)
        # status 2 means only obeject not in userviewport
        UpdateTmpTileStatuesBasedOnObject(tmp_TileStatus,width,height, TileNO, X, Y, XL, YL,2)

def FromObjectToUpdatTileOnTracking(Namelist, CorUPleft, CorDownRig, TileStatus,height, width,tileNo, UX, UY, UXL, UYL):
    ''' Update tile status if object in the user viewport'''
    L = len(Namelist)
    if L>10:
        K=50
    else:
        K=100
    for i in range(0, L):
        X=CorUPleft[i][0]
        Y=CorUPleft[i][1]
        XL=CorDownRig[i][0]
        YL=CorDownRig[i][1]
        if ((X<=UXL and X>=UX )or (XL<=UXL and XL>=UX )) and ((Y<=UYL and Y>=UY )or (YL<=UYL and YL>=UY )):
            if XL-X<200 or YL-Y<200:
                X=X-K
                XL=XL+K
                YL=YL+150
                Y=Y-150
                if X<0:
                    X=0
                if Y<=0:
                    Y=0
                if XL>width:
                    XL=width
                if YL>height:
                    YL=height
            UpdateTileStatuesBasedOnObject(TileStatus, width, height, tileNo, X, Y, XL, YL)
            # status 3 means obeject in userviewport
            UpdateTmpTileStatuesBasedOnObject(tmp_TileStatus,width,height, TileNO, X, Y, XL, YL,3)
        else : 
            # status 2 means only obeject ,not in userviewport
            UpdateTmpTileStatuesBasedOnObject(tmp_TileStatus,width,height, TileNO, X, Y, XL, YL,2)


def UpdateTrack(Namelist, CorUPleft, CorDownRig, FNamelist, FCorUPleft, FCorDownRig, FConfi):
    Count = 0
    #print("tital list")
    #print(len(FNamelist))
    #print("new list")
    #print(len(Namelist))
    ''' Checking object appear in two consecutive frame, then caculate their pair distance '''
    ''' FNamelist is previous object list detected by previous user view, Namelist is current object list'''
    NewList = []
    for j in range(0, len(Namelist)):
        NewList.append(0)

    for i in range(0, len(FNamelist)):
        D = []
        for j in range(0, len(Namelist)):
            if Namelist[j] == FNamelist[i] and NewList[j] == 0:
                # ''' If the object appear in two consecutive frame then calculate this pair distance '''
                UL = (CorUPleft[j][0] - FCorUPleft[i][0]) ** 2 + (CorUPleft[j][1] - FCorUPleft[i][1]) ** 2
                # RD=(CorDownRig[0] - FCorDownRig[i][0]) ** 2 + (CorDownRig[1] - FCorDownRig[i][1]) ** 2
                D.append(UL.item())
                # print(UL.item())
            else:
                # ''' If not then give a huge distance '''
                D.append(10000)
        ## To find min distance in pair(i,m)
        K = D.copy()
        K.sort()
        if K[0] < 400: # for checking if there is one more same object then choose the closer enough object
            m = D.index(K[0])
            FConfi[i] = 1
            FCorUPleft[i] = CorUPleft[m]
            # FCorUPleft[i][1] = CorUPleft[m][1]
            FCorDownRig[i] = CorDownRig[m]
            # FCorDownRig[i][1] = CorDownRig[m][1]
            # match count
            Count = Count + 1
            # 1 means the object matched in two frame
            NewList[m] = 1


        else:
            # match failed object
            FConfi[i] = 0
    ''' Below is New object detected, push it into old object list append FConfi(1),indicate this object will be shown '''
    Cad = 0
    for j in range(0, len(Namelist)):
        if NewList[j] == 0:
            FNamelist.append(Namelist[j])
            FCorUPleft.append(CorUPleft[j])
            # FCorUPleft[i][1] = CorUPleft[I][1]
            FCorDownRig.append(CorDownRig[j])
            FConfi.append(1)
            Cad = Cad + 1
    ''' Check if object in the new list appear in the Estimated user view, then update it tile status  '''
    #print("Total mathed")
    #print(Count)
    #print("Total added")
    #print(Cad, NewList.count(0))


def CheckIFUserViewCovered_origin (TileStatus,w,h,tileNo,X,Y,XL,YL):
    wd = w / tileNo # EX:2560 / 10
    hd = h / tileNo # EX:1440 / 10 
    IL = math.floor(X / wd)
    IH = math.ceil(XL / wd)
    JL = math.floor(Y / hd)
    JH = math.ceil(YL / hd)
    if JH>10:
        JH=10
    if IH>10:
        IH=10
    for i in range(IL, IH ):
        for j in range(JL, JH ):
            #print("Width tile:{} High tile:{} status:{} \n".format(i,j,TileStatus[i + j * tileNo]))
            if TileStatus[i + j * tileNo] ==0:
                return False
    return True

def UpdateTileStatues(TileStatus):
    i=len(TileStatus)
    for k in range(0,i):
        if TileStatus[k]>0:
            TileStatus[k]=TileStatus[k]-1

def DrawTiles(TileStatus,w,h,tileNo,img):
    wd=w/tileNo
    hd=h/tileNo
    color=(0,0,0)
    color2 = (0, 255, 0)
    for i in range(0,tileNo*tileNo):
        Y=i//tileNo
        X=i%tileNo
        a=int(X*wd)
        b=int(Y*hd)
        C1=[a,b]
        a1=int(X * wd+wd)
        b1=int(Y * hd+hd)
        C2 =[a,b]
        if TileStatus[i] == 0:
            cv2.rectangle(img, (a,b), (a1,b1), color, 1) # Black means unpredicted tile
        else:
            cv2.rectangle(img, (a,b), (a1,b1), color2, 1) # Green means predicted tiles
# the object size should be larger than the tile
#X,Y is the coordinate of the upperleft XL,YL is the down right
def UpdateTileStatuesBasedOnObject(TileStatus,w,h,tileNo,X,Y,XL,YL):
    wd = w / tileNo
    hd = h / tileNo
    IL=math.floor(X/wd)
    IH=math.ceil(XL/wd)
    JL = math.floor(Y / hd)
    JH = math.ceil(YL / hd)
    if JH >= 10:
        JH = 9
    if IH >= 10:
        IH = 9
    for i in range(IL, IH + 1):
        for j in range(JL, JH + 1):

            TileStatus[i+j*tileNo]=ShrinkV

def UpdateTmpTileStatuesBasedOnObject(tmpTileStatus,w,h,tileNo,X,Y,XL,YL,status):
    "status : 0-3 "
    wd = w / tileNo
    hd = h / tileNo
    IL=math.floor(X/wd)
    IH=math.ceil(XL/wd)
    JL = math.floor(Y / hd)
    JH = math.ceil(YL / hd)
    if JH >= 10:
        JH = 9
    if IH >= 10:
        IH = 9
    for i in range(IL, IH + 1):
        for j in range(JL, JH + 1):
            tmpTileStatus[i+j*tileNo]=status
def ReadAllUserData(FileHead,NO):
    AllUserData=[]
    for i in range(1,NO+1):
        FileName=FileHead+str(i)+'.csv'
        if i==1:
            flag=1
        else:
            flag=0
        OneUserData,TimeStamp = ReadOneuserData(FileName,flag)
        AllUserData.append(OneUserData)
    return AllUserData,TimeStamp

def ReadOneuserData(FileName,flagTime):
    TimeStamp = []
    #flagTime = 1
    Userdata = []
    with open(FileName) as csvfile:
        csv_reader = csv.reader(csvfile)  
        birth_header = next(csv_reader) 
        for row in csv_reader:  
            Userdata.append(row[1:])
            if flagTime == 1:
                TimeStamp.append(row[0])
    Userdata = [[float(x) for x in row] for row in Userdata]  
    Userdata = np.array(Userdata)  
    return Userdata,TimeStamp



def multiThread_feature(img,i):
    color = (255, 0, 0)
    cv2.rectangle(img, (400, 300), (1000, 600), color, 1)
    print ("good")
    print(i)

def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim


def write(x, img, Namelist, CorUPleft, CorDownRig):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    if len(classes) >= cls:
     label = "{0}".format(classes[cls])
     Namelist.append(label) # append object into namelist
    else:
        label = '99'
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    CorUPleft.append(c1)
    CorDownRig.append(c2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    # cv2.rectangle(img, c1, c2,color, -1)
    # cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img



def DrwaObject(FNamelist, FCorUPleft, FCorDownRig, FConfi, orig_im):
    for i in range(0, len(FNamelist)):
        if FConfi[i] == 1:
            color = (i * 5, 255 - i * 5, i * 7)
            cv2.rectangle(orig_im, FCorUPleft[i], FCorDownRig[i], color, 1)
            t_size = cv2.getTextSize(FNamelist[i], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            c2 = FCorUPleft[i][0] + t_size[0] + 3, FCorUPleft[i][1] + t_size[1] + 4
            # cv2.rectangle(img, FCorUPleft[i], c2, color, -1)
            cv2.putText(orig_im, str(i), (FCorUPleft[i][0], FCorUPleft[i][1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN,1, [225, 255, 255], 1)
def caculate_fwd_vec(qx,qy,qz,qw):
    x = 2*qx*qz + 2*qy*qw
    y = 2*qy*qz - 2*qx*qw
    return x,y



if __name__ == '__main__':

    #for UserIndex in range(1,5):
    args = arg_parse()

    if args.RL == True:
        method = "RL"
    elif args.Tracking == True:
        method = "Tracking"
    else :
        method = "Over_covered"

    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80

    CUDA = torch.cuda.is_available()

    bbox_attrs = 5 + num_classes

    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    if CUDA:
        model.cuda()

    model(get_test_input(inp_dim, CUDA), CUDA)
    model.eval()

        #"1-3-Help": ("Experiment_1","video_2.csv"),
        #"1-7-Cooking Battle" : ("Experiment_1","video_6.csv"),
    videos_track_dict={
        "1-3-Help": ("Experiment_1","video_2.csv"),
        "1-7-Cooking Battle" : ("Experiment_1","video_6.csv"),
        #"2-2-VoiceToy":("Experiment_2","video_1.csv"),
        "2-3-RioVR":("Experiment_2","video_2.csv"),
        "2-4-FemaleBasketball":("Experiment_2","video_3.csv"),
        "2-6-Anitta" :("Experiment_2","video_5.csv"),
        #"2-8-Reloaded":("Experiment_2","video_7.csv")
    }

    for UserIndex in range(1,2):
        videofile = args.video
    # Read file operation
        #=====================================================================

        for video,track_path in videos_track_dict.items():
            VideoName = video          #1-2-FrontB  1-1-Conan Gore FlyB  1-9-RhinosB
            #VideoName ="1-2-Front"
            videofile = VideoName + ".mp4"  # 2-4-FemaleBasketballB 2-6-AnittaB 1-6-FallujaB 2-3-RioVRB  2-5-FightingB  2-8-reloadedB
            print("\n\n    '''    Video{}     '''".format(video))
            cap = cv2.VideoCapture(videofile)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            assert cap.isOpened(), 'Cannot capture source'


            CSVUserFileHeader=VideoName #"video_5_D1_"    #videoname: A-B-Name.mp4 if A=1 add D1; B is the number for video, for user csv it is B-1
                                            #video_0_ or video_0_D1_
            fileName = "video_5_D1_"+str(UserIndex)+".csv"
            NOUsr=UserIndex                         #1-48
            CSVFinalUserFile=CSVUserFileHeader+str(NOUsr)+".csv"
            if args.RL == True:
                CSVFinalUserFileToSave="AAResult_" +CSVUserFileHeader+str(NOUsr)+"_"+str(bufferLen)+"s_RL.csv"
                SfileName = "Asave_tracking_RL_"+str(NOUsr)+'_' + VideoName + ".csv"
                SvideoName = "Asave_tracking_RL_" +str(NOUsr)+'_'+ VideoName + ".avi"
            elif args.Tracking == True:
                CSVFinalUserFileToSave="AAResult_"+CSVUserFileHeader+str(NOUsr)+"_"+str(bufferLen)+"s_tracking.csv"
                SfileName = "Asave_Tracking_"+str(NOUsr)+'_' + VideoName + ".csv"
                SvideoName = "Asave_Trackig_" +str(NOUsr)+'_'+ VideoName + ".avi"
            else:
                CSVFinalUserFileToSave="AAResult_" + CSVUserFileHeader + str(NOUsr) + "_" + str(bufferLen) + "s_Basic.csv"
                SfileName = "Asave_overcover_"+str(NOUsr)+'_' + VideoName + ".csv"
                SvideoName = "Asave_overcover_" +str(NOUsr)+'_'+ VideoName + ".avi"

            outAccuBanDRes = open(CSVFinalUserFileToSave, 'w', newline='')
            outF = open(SfileName, 'w', newline='')        
            csv_writeABF = csv.writer(outAccuBanDRes, dialect='excel')
            ACCbadRestul=[VideoName+str(NOUsr),"UserFeedback","Frame fail predicted","Total Tiles Used","Total tiles"]
            csv_writeABF.writerow(ACCbadRestul)
            csv_writeF = csv.writer(outF, dialect='excel')

            stu3 = ["Object Id","Upleft X","Upleft Y","DownRight X","Downright Y"]
            csv_writeF.writerow(stu3)


            '''initialization '''
            ret, frame = cap.read()
            # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # img1 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # orb = cv2.ORB_create()

            # find the keypoints and descriptors with ORB
            #kp1, des1 = orb.detectAndCompute(img1, None)


            height, width = frame.shape[:2]
            Framerate = int(cap.get(5))
            out = cv2.VideoWriter(SvideoName, fourcc, 30.0, (width, height))
            print("H:{},W:{}\n".format(height,width))


            # Some parameters here:
            Framerate = int(cap.get(5))
            TileNO=10           # the final number should be TileNO*TileNO
            TileStatus=[3]*TileNO*TileNO
            ''' Load user data '''
            #obtain the user trace from reference[27]
            filepath = './Formated_Data/'+ track_path[0]+ '/' + str(UserIndex)+ '/' + track_path[1]
            UserDataPF = load_user_data(filepath,Framerate,height,width)       
            UserLenData=len(UserDataPF)

            '''create lists for the objects'''
            FNamelist = [] # oject label list
            FCorUPleft = [] #UpLeft coordinate
            FCorDownRig = [] # DownRight coordinate
            FConfi = [] # predicted status 0(not) or 1(yes)

            IsFirst = True
            '''
            Test multi thread
            '''
            # KKK=0
            # _thread.start_new_thread(multiThread_feature,(IsFirst,KKK))
            # KKK2=0
            # _thread.start_new_thread(multiThread_feature,(IsFirst,KKK2))



            frames = 0
            start = time.time()
            CountRunTime=0
            Predicted_fail=0
            CountInOnebuffer=0
            Total_tile_used = 0
            wrong_frame = []
            ## For RL Q_table
            Q_table = torch.zeros((TileNO*TileNO, 8 , 2))
            # Initiallize all tile status to select 
            Q_table[:,:,1] = 5
            while cap.isOpened():
                Predicted_fail = 0
                ret, frame = cap.read()
                #cv2.imshow('frame', frame)
                #print("cap frame status{}:\n".format(ret))
                if ret:

                    img, orig_im, dim = prep_image(frame, inp_dim)

                    im_dim = torch.FloatTensor(dim).repeat(1, 2)

                    if CUDA:
                        im_dim = im_dim.cuda()
                        img = img.cuda()

                    with torch.no_grad():
                        output = model(Variable(img), CUDA)
                    output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

                    if type(output) == int:
                        print("====")
                        frames += 1
                        print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
                        cv2.imshow("frame", orig_im)
                        key = cv2.waitKey(1)
                        if key & 0xFF == ord('q'):
                            break
                        continue

                    im_dim = im_dim.repeat(output.size(0), 1)
                    scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

                    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
                    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

                    output[:, 1:5] /= scaling_factor

                    for i in range(output.shape[0]):
                        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

                    classes = load_classes('data/coco.names')
                    colors = pkl.load(open("pallete", "rb"))

                    Namelist = []
                    CorUPleft = []
                    CorDownRig = []
                    Confi = []

                    # check all the objects that are detected
                    list(map(lambda x: write(x, orig_im, Namelist, CorUPleft, CorDownRig), output))


                    #  **************** Yolo-v3 part finished     物件偵測結束 *******************

                    # to record current tile status : 0: non 1: in user viewport 2: object 3: object in user viewport 
                    tmp_TileStatus = [0] * TileNO * TileNO 
                    Pre_TileStatus = TileStatus.copy()
                    status_action = []
                    #get user feedback and check the update：
                    UsrCurX=0
                    UsrCurY=0
                    UserFeedBackResult=10
                    # 當有過去5s的user data 執行velocity method 預測viewport並更新tile status
                    if CountRunTime>bufferLen*Framerate+Framerate: 
                        UserFeedBackResult=5
                        UsrCurX=UserDataPF[CountRunTime-bufferLen*Framerate][0]
                        UsrCurY = UserDataPF[CountRunTime - bufferLen * Framerate][1]
                        Vx=0
                        Vy=0
                        # velocity-based method: caculate 25 frames Velocity
                        for ind in range(CountRunTime-(bufferLen*Framerate+Framerate),CountRunTime-bufferLen*Framerate):
                            Vx=UserDataPF[ind+1][0]-UserDataPF[ind][0]+Vx
                            Vy = UserDataPF[ind + 1][1] - UserDataPF[ind][1] + Vy
                        Vx=Vx/Framerate
                        Vy = Vy / Framerate

                        #estimated user viewport center = origin location center + velocity
                        #previous frame info estimated current  
                        UsrCurX=int(UsrCurX+Vx)
                        UsrCurY=int(UsrCurY+Vy)
                        X = int(UsrCurX - UserView)
                        Y = int(UsrCurY - UserView)
                        XL = int(UsrCurX + UserView)
                        YL = int(UsrCurY + UserView)
                        if X <= 0:
                            X = 0
                        if Y < 0:
                            Y = 0
                        if XL > width:
                            XL = width
                        if YL > height:
                            YL = height
                        # estimated viewport down left
                        UsRX=X
                        UsRL=Y
                        ##  ***********  Update Tile Base on estimated user view  *************
                        UpdateTileStatuesBasedOnObject(TileStatus, width, height, TileNO, X, Y, XL, YL)
                        UpdateTmpTileStatuesBasedOnObject(tmp_TileStatus,width,height, TileNO, X, Y, XL, YL,1)

                    
                    ##  ***********  Below are tracking part  *************
                    '''
                    list(Namelist) current frame detected objects
                    list(CorUPleft) coordinate top left
                    list(CorDownRig) coordinate down righ
                    list(FNamelist) Final object list after tracking 
                    list(FCorUPleft)
                    list(FCorDownRig)
                    list(FConfi) 1: matched in two frame object, 0:not matched 
                    '''


                    #Update Tile based on the objects
                    #print("Current runtime frame:{} Total Frame used:{} Total Usertrace Frame:{}\n".format(CountRunTime,lenU*Framerate,UserLenData))
                    # ************        Advanced Tracking method        ************   If have user data then tracking like Sec 4.  object in viewport
                
                    if args.Tracking ==True and CountRunTime>bufferLen*Framerate+Framerate:
                    # Maintain two list of object, one for current frame another for previous frame 
                        if IsFirst == False:
                            UpdateTrack(Namelist, CorUPleft, CorDownRig, FNamelist, FCorUPleft, FCorDownRig, FConfi)
                            DrwaObject(FNamelist, FCorUPleft, FCorDownRig, FConfi, orig_im)
                        else:
                            
                            FNamelist = Namelist.copy()
                            FCorUPleft = CorUPleft.copy()
                            FCorDownRig = CorDownRig.copy()
                            for i in range(0, len(FNamelist)):
                                FConfi.append(1)
                            DrwaObject(FNamelist, FCorUPleft, FCorDownRig, FConfi, orig_im)

                        FromObjectToUpdatTileOnTracking(FNamelist, FCorUPleft, FCorDownRig, TileStatus,height, width,TileNO, X, Y, XL, YL) 
                        #FromObjectToUpdatTile(Namelist, CorUPleft, CorDownRig, TileStatus,height, width,TileNO)       

                        IsFirst = False
                        # print("2")
                        stu3 = []
                        for i in range(0, len(FNamelist)):
                            if FConfi[i] == 1:
                                stu3.append(str(i))
                                stu3.append(FCorUPleft[i][0].item())
                                stu3.append(FCorUPleft[i][1].item())
                                stu3.append(FCorDownRig[i][0].item())
                                stu3.append(FCorDownRig[i][1].item())
                        csv_writeF.writerow(stu3)

                    # ************        Over-Cover method        ************    Just consider object detection result
                
                    else: 
                        FromObjectToUpdatTile(Namelist, CorUPleft, CorDownRig, TileStatus,height, width,TileNO)



                    # **************        Missing about Reinforce learning Code   *************
                    #Update based on the Q-table
                    ''' 
                    tracking results(tile status) and the estimated user view(X,Y,XL,YL) are fed into the reinforcement learning-based
                    modeling step, which updates the status of each tile
                    '''
                    #print("before function",torch.min(Q_table[:,:,1]))
                    if args.RL == True:
                        RL_Qtable(tmp_TileStatus,Pre_TileStatus,Q_table,status_action,TileStatus)
                    #print("after function",torch.min(Q_table[:,:,1]))
                    # **************                                                *************

                    #Check accuracy
                    ## Current frame groundturth viewport
                    X=UserDataPF[CountRunTime][0]-UserView/2
                    Y=UserDataPF[CountRunTime][1]-UserView/2
                    XL = UserDataPF[CountRunTime][0] + UserView / 2
                    YL = UserDataPF[CountRunTime][1] + UserView / 2
                    if X<=0:
                        X=0
                    if Y<0:
                        Y=0
                    if XL>width:
                        XL=width
                    if YL>height:
                        YL=height
                    C1=(int(X),int(Y))
                    C2=(int(XL),int(YL))


                    #Check the performance
                    if args.RL == True:
                        #print("RL is nice")
                        Res=CheckIFUserViewCovered(TileStatus, width, height, TileNO, X, Y, XL, YL,status_action,Q_table)
                    #print(Q_table[1:5])
                    #print("\n\n") 

                    else:
                        Res=CheckIFUserViewCovered_origin(TileStatus, width, height, TileNO, X, Y, XL, YL)
                    if Res==False:
                        Predicted_fail = 1
                        wrong_frame.append(1)
                        #print("Wrong: tile uncovered by prediction")  
                    else:
                        wrong_frame.append(0)                    
                    CountRunTime=CountRunTime+1

                    # ''' After certain period decrease the Tile status '''
                    if CountRunTime%10 ==0:
                        UpdateTileStatues(TileStatus)

                    #Velocity method performance check based on user feedback
                    if UserFeedBackResult==5:
                        if (UsRX<=X and UsRX+20>=X) and (UsRL<=Y and UsRL+20>Y):
                            UserFeedBackResult=1 # Velocity method estimated good
                        else:
                            UserFeedBackResult=0 # Velocity method estimated bad

                    #show result in video:
                    DrawTiles(TileStatus, width, height, TileNO, orig_im)
                    color=(0,0,255) # Red means user viewport
                    cv2.rectangle(orig_im, C1, C2, color, 1)
                    ACCbadRestul = [time.time() - start, UserFeedBackResult, Predicted_fail, TileNO*TileNO-TileStatus.count(0),
                                    TileNO*TileNO]
                    Total_tile_used = Total_tile_used + (TileNO*TileNO-TileStatus.count(0))
                    csv_writeABF.writerow(ACCbadRestul)
                    # print(Namelist)
                
        
                    
                    cv2.imshow("frame", orig_im)
                    out.write(orig_im)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        break
                    frames += 1
                    if CountRunTime>lenU*Framerate-1:
                        break


                else:
                    break
            Total_tile = lenU * TileNO * TileNO * Framerate
            cap.release()
            out.release()
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
            print("Total wrong frame:{}\n".format(np.array(wrong_frame).sum()))
            print("Accuracy:",(3000-np.array(wrong_frame).sum()) / 3000)
            print("Total tile used:{} / {}, bandwith:{}% \n\n".format(Total_tile_used,Total_tile,Total_tile_used/Total_tile))
            #print("Wrong frame No.",wrong_frame)

