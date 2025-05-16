import cv2 as cv  #匯入OpenCV模組
import numpy as np #匯入numpy模組
import matplotlib.pyplot as plt #匯入matplotlib模組
im=cv.imread("ArtGallery.jpg")#讀取影像
x,y,z=im.shape#讀取影像大小
fake_im=np.zeros((x,y,z),dtype="uint8")#建立假影像矩陣
fake_im[:,:,:]=im[:,:,:]#複製原影像
Source_Point=np.array([[344,311],[405,639],[831,311],[801,642]])#設定左邊圖畫座標(4_point)
Destination_Point=np.array([[471,1405],[438,1628],[692,1412],[705,1635]])#設定右邊圖畫座標(4_point)  
L1=np.cross([344,311,1],[405,639,1])#計算左邊圖畫邊線L1
print(L1)
L1=L1/((L1[0]**2+L1[1]**2)**0.5)#normalized
print(L1)
print(np.matmul(L1,[344,312,1]))
L2=np.cross([344,311,1],[831,311,1])#計算左邊圖畫邊線L2
L2=L2/((L2[0]**2+L2[1]**2)**0.5)#normalized
L3=np.cross([801,642,1],[405,639,1])#計算左邊圖畫邊線L3
L3=L3/((L3[0]**2+L3[1]**2)**0.5)#normalized
L4=np.cross([801,642,1],[831,311,1])#計算左邊圖畫邊線L4
L4=L4/((L4[0]**2+L4[1]**2)**0.5)#normalized

R1=np.cross([471,1405,1],[438,1628,1])#計算左邊圖畫邊線R1
R1=R1/((R1[0]**2+R1[1]**2)**0.5)#normalized
R2=np.cross([471,1405,1],[692,1412,1])#計算左邊圖畫邊線R2
R2=R2/((R2[0]**2+R2[1]**2)**0.5)#normalized
R3=np.cross([705,1635,1],[438,1628,1])#計算左邊圖畫邊線R3
R3=R3/((R3[0]**2+R3[1]**2)**0.5)#normalized
R4=np.cross([705,1635,1],[692,1412,1])#計算左邊圖畫邊線R4
R4=R4/((R4[0]**2+R4[1]**2)**0.5)#normalized
H=cv.findHomography(Source_Point,Destination_Point)#計算H矩陣
for i in range (344,831):
    for j in range (311,642):
        origin_point_L=np.array([[i],[j],[1]])#紀錄左邊圖畫每一點座標
        if np.matmul(L1,origin_point_L)<=0 and np.matmul(L2,origin_point_L)>=0 and np.matmul(L3,origin_point_L)>=0 and np.matmul(L4,origin_point_L)<=0 :
            #if 判別每一點是否為在圖畫邊線區域內
            swap_point_L=np.matmul(H[0],origin_point_L)#計算得到交換的座標點
            normalized_point_L=np.around(swap_point_L/swap_point_L[2])#將座標normalized並四捨五入
            fake_im[int(normalized_point_L[0]),int(normalized_point_L[1])]=im[i,j]#將交換座標像素數值設為原影像像素數值
for i in range(438,705):
    for j in range(1405,1635):
        origin_point_R=np.array([[i],[j],[1]])#紀錄右邊圖畫每一點座標
        if np.matmul(R1,origin_point_R)<=0 and np.matmul(R2,origin_point_R)>=0 and np.matmul(R3,origin_point_R)>=0 and np.matmul(R4,origin_point_R)<=0 :
            #if 判別每一點是否為在圖畫邊線區域內
            swap_point_R=np.matmul(np.linalg.inv(H[0]),origin_point_R)#計算得到交換的座標點
            normalized_point_R=np.around(swap_point_R/swap_point_R[2])#將座標normalized並四捨五入
            fake_im[int(normalized_point_R[0]),int(normalized_point_R[1])]=im[i,j]#將交換座標像素數值設為原影像像素數值

fake_im=cv.cvtColor(fake_im,cv.COLOR_RGB2BGR)#轉換假影像的色彩通道(BGR>RGB)
print(H[0])
plt.imshow(fake_im)#顯示假影像
plt.axis('off') #關閉X軸以及Y軸
#plt.savefig("M11104201.jpg") #儲存影像
plt.show()#顯示圖片
