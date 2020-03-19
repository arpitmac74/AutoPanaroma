#!/usr/bin/evn python

# Code starts here:

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math 
import numpy.linalg
import glob
from copy import deepcopy 
import skimage.feature 
import argparse
#starting with ANMS
def ANMS(gray, num_corners):
    features = cv2.goodFeaturesToTrack(gray, 1000, 0.01,10)
    p,_,q = features.shape
    r = 1e+8 *np.ones([p,3])
    #print(r)
    ED = 0

    for i in range(p):
        for j in range(p):
            xi = int(features[i,:,0])
            yi = int(features[i,:,1])
            xj = int(features[j,:,0])
            yj = int(features[j,:,1])
            if gray[yi,xi]>gray[yj,xj]:
               # print(gray[yi,xi])
                ED = (xj-xi)**2 +(yj-yi)**2
            if ED < r[i,0]:
                r[i,0] = ED
                r[i,1] = xi
                r[i,2] = yi            
    feat = r[np.argsort(-r[:, 0])]
    best_corners = feat[:num_corners,:]
    return best_corners
#Feature Descriptor
#image should be grayscale
def featuredes(img, pad_width,anms_out,patch_size):
    feats=[]
    if (patch_size%2)!= 0:
        print('Patch size is not acceptable')
        return -1
    l,w = anms_out.shape
    image_pad =np.pad(img,patch_size,'constant',constant_values=0)
    desc = np.array(np.zeros((int((patch_size/5)**2),1)))
    for i in range(0,l):
    #making patches for descriptors
     patch = image_pad[int(anms_out[i][2]+(patch_size/2)):int(anms_out[i][2]+(3*patch_size/2)),int(anms_out[i][1]+(patch_size/2)):int(anms_out[i][1]+(3*patch_size/2))]
     blur_patch = cv2.GaussianBlur(patch,(5,5),0)
    
     sub_sample = blur_patch[0::5,0::5]
     #cv2.imwrite('./patches/patch'+str(i)+'.png',sub_sample)
     feats = sub_sample.reshape(int((patch_size/5)**2),1)
     #mean 0 
     feats = feats - np.mean(feats)
     #make the variacne =1
     feats = feats/np.std(feats)
     desc = np.dstack((desc,feats))
    return desc[:,:,1:]
# Match Pairs in the lists of two feature descriptors
def match_pairs(features1,features2,best_corners1,best_corners2):
    p,x,q = features1.shape
    m,y,n = features2.shape
    q = int(min(q,n))
    n = int(max(q,n))
    matchPairs = []
    for i in range(q):
        match = {}
        for j in range(n):
            ssdist = np.linalg.norm((features1[:,:,i]-features2[:,:,j]))**2
            match[ssdist] = [best_corners1[i,:],best_corners2[j,:]]
        S =sorted(match)
        first = S[0]
        sec = S[1]
        if first/sec < 0.7:
            pairs = match[first]
            matchPairs.append(pairs)
    return matchPairs   
#draw mactches for featrue matching and ransac
def showfeatures(img1,img2,matchPairs,new_img_name):
    if len(img1.shape)==3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape)==2:
        new_shape = (max(img1.shape[0], img2.shape[0]),img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape,type(img1.flat[0])) 
    #place the two images onto the new image
    new_img[0:img1.shape[0],0:img1.shape[1]]=img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
###############################################################################    
    
    for i in range(len(matchPairs)):
        x1 = int(matchPairs[i][0][1])
        y1 = int(matchPairs[i][0][2])
        x2 = int(matchPairs[i][1][1])+int(img1.shape[1])
        y2 = int(matchPairs[i][1][2])
        
        cv2.line(new_img,(x1,y1),(x2,y2),(200,10,200),2)
        cv2.circle(new_img,(x1,y1),3,255,-1)
        cv2.circle(new_img,(x2,y2),3,255,-1)
    
    ########################cv2.imwrite(new_img_name,new_img)
        
    
    return new_img

# now we have to ransac to reject the outliers 
def ransac(pairs,N,t,thresh):
    M = pairs
    
    H_new =np.zeros((3,3))
    max_inliers = 0
    
    for j in range(N):
        
        index = []
        pts = [np.random.randint(0,len(M)) for i in range(4)]
        p1 = np.array([[M[pts[0]][0][1:3]],[M[pts[1]][0][1:3]],[M[pts[2]][0][1:3]],[M[pts[3]][0][1:3]]],np.float32)
        p2 = np.array([[M[pts[0]][1][1:3]],[M[pts[1]][1][1:3]],[M[pts[2]][1][1:3]],[M[pts[3]][1][1:3]]],np.float32)
        H = cv2.getPerspectiveTransform( p1, p2 )
        inLiers = 0
        for ind in range(len(M)):
            source = M[ind][0][1:3]
            target = M[ind][1][1:3]
            predict =np.matmul(H,(np.array([source[0],source[1],1]))) 
            if(predict[2] == 0):
                predict[2]=0.00001
            predict_x =predict[0]/predict[2]
            predict_y = predict[1]/predict[2]
            predict = np.array([predict_x,predict_y])
            #predict = np.float32([point for point in predict])
            if (np.linalg.norm(target-predict)) < thresh:
                inLiers +=1
                
                index.append(ind)
        
        pts1 = []
        pts2 = []
        if max_inliers < inLiers:
            max_inliers = inLiers
            [pts1.append([M[i][0][1:3]]) for i in index]  
            [pts2.append([M[i][1][1:3]]) for i in index]
            H_new,status = cv2.findHomography(np.float32(pts1),np.float32(pts2))
            if inLiers > t*len(M):
                print('Success')
                
                break
    pairs = [M[i] for i in index]
    if len(pairs)<=4:
        print('pairs after RANSAC too low')
    return H_new,pairs

def image_stitch(image, homography,image2_shape):
    
    h,w,z = image.shape
    
    #finf min and max x,y of new image
    p=np.array([[0, w, w, 0], [0, 0, h, h], [1, 1, 1, 1]])
    p_prime = np.dot(homography, p)
    
    yrow = p_prime[1] / p_prime[2]
    xrow = p_prime[0] / p_prime[2]
    ymin = min(yrow)
    xmin = min(xrow)
    ymax = max(yrow)
    xmax = max(xrow)
    
    # make new matrix that removes offset and multiply by homography
    new_mat = np.array([[1, 0, -1* xmin], [0, 1, -1*ymin], [0, 0, 1]])
    homography  = np.dot(new_mat, homography)
    
    #height and width of new image frame
    height = int(round(ymax - ymin)) + image2_shape[0]
    width = int(round(xmax-xmin)) + image2_shape[1]
    size = (height,width)
    
    #warp 
    warped = cv2.warpPerspective(src=image, M=homography, dsize=size)
    
    return warped, int(xmin), int(ymin)
def Estimated_Homography(img1,img2):
    #gatting gray images
    flag = True
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray1 = np.float32(gray1)
    
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray2 = np.float32(gray2)
    
    #corner detecttion
    corners1 = cv2.goodFeaturesToTrack(gray1, 100000, 0.001,10)
    corners1 = np.int0(corners1)
    
    i1 = deepcopy(img1)
    for corner in corners1:
        x,y = corner.ravel()
        cv2.circle(i1,(x,y),3,255,-1)
        ##########################cv2.imwrite('corner1.png,i1)
        
    corners2 = cv2.goodFeaturesToTrack(gray2, 100000, 0.001,10)
    corners2 = np.int0(corners2)

    i2 = deepcopy(img2)
    for corner in corners2:
        x,y = corner.ravel()
        cv2.circle(i2,(x,y),3,255,-1)
    ########################cv2.imwrite('corners2.png',i2)   
    
    
    #ANMS and save output 
    
    best_corners1 = ANMS(gray1, 700)
    i1 = deepcopy(img1)
    #mark in deepcopy and save the image
    for corner1 in best_corners1:
        _,x1,y1 = corner1.ravel()
        cv2.circle(i1,(int(x1),int(y1)),3,255,-1)
    ###############cv2.imwrite('anms1.png',i1)
    best_corners2 = ANMS(gray2, 700)
    # anms = copy.deepcopy(img2)
    i2 = deepcopy(img2)
    for corner2 in best_corners2:
        _,x2,y2 = corner2.ravel()
        cv2.circle(i2,(int(x2),int(y2)),3,255,-1)
    ######################cv2.imwrite('anms2.png',i2)
    
    
    #feature descriptors
    feat1 = featuredes(img=gray1, pad_width=40,anms_out=best_corners1,patch_size=40)
    feat2 = featuredes(img=gray2, pad_width=40,anms_out=best_corners2,patch_size=40)
    
    #feature matching 
    matchPairs = match_pairs(features1 = feat1,features2 = feat2, best_corners1 = best_corners1,best_corners2=best_corners2)
    if len(matchPairs)<45:
        print("error matched pairs too low")
        flag = False
    showfeatures(img1,img2,matchPairs,new_img_name = 'featurematching.png')
    #ransac
    Hmg,pairs = ransac(matchPairs,N=3000,t=0.9 ,thresh=30.0)
    showfeatures(img1,img2,pairs,new_img_name = 'ransac.png')    
    
    return Hmg,flag

def Blend(images):
    img1 = images[0]
    for im in images[1:]:
        H,flag = Estimated_Homography(img1,im)
        if flag == False:
            print('Number of matches is less than required')
            break
        holder, origin_offset_x, origin_offset_y = image_stitch(img1,H,im.shape)
        oX = abs(origin_offset_x)
        oY = abs(origin_offset_y)
        for y in range(oY,im.shape[0]+oY):
            for x in range(oX,im.shape[1]+oX):
                #print(y)
                img2_y = y - oY
                img2_x = x - oX
                holder[y,x,:] = im[img2_y,img2_x,:]
        img1 = holder
    resize = cv2.GaussianBlur(img1,(5,5),1.2)
    cv2.imwrite('panoblur.png',resize)
    return resize
def main():
	# Add any Command Line arguments here
    Parser = argparse.ArgumentParser()
    
    Parser.add_argument('--BasePath', default='../Data/Train/Set2', help='Define path of image')
    
    Args = Parser.parse_args()
    BasePath = Args.BasePath
    
    images = [cv2.imread(file) for file in sorted(glob.glob(str(BasePath)+'/*.jpg'))]
    mypano = Blend(images)
    #print('number of images',len(images))
#    gray1 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
#    gray1 = np.float32(gray1)
#    gray2 =cv2.cvtColor(images[1], cv2.COLOR_BGR2GRAY)
#    gray2 = np.float32(gray2)
#
#    #best=ANMS(gray1,700)
#    #print(best.shape)
#    #a=featuredes(img=gray1, pad_width=40,anms_out=best,patch_size=40)
#    
#    best1 = ANMS(gray1, 700)
#    best2 = ANMS(gray2, 700)
#    feat1 = featuredes(img=gray1, pad_width=40,anms_out=best1,patch_size=40)
#    feat2 = featuredes(img=gray2, pad_width=40,anms_out=best2,patch_size=40)
#    
#    b = match_pairs(features1=feat1,features2=feat2,best_corners1=best1,best_corners2=best2)
#    a=showfeatures(img1=gray1,img2=gray2,matchPairs=b,new_img_name='featurematching.png')
#    c = ransac(pairs=b,N=3000,t=0.9,thresh=30)
#    print(c[1])
    #plt.imshow(a)
    #plt.show()
 #   for corner in best:
#        _,x2,y2 = corner.ravel()
#    cv2.circle(images[1],(int(x2),int(y2)),3,255,-1)
    #plt.imshow(a)
    #plt.show()
    #cv2.imshow('h',a)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
 
