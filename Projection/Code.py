import cv2 as cv  #匯入OpenCV模組
import numpy as np #匯入numpy模組
import matplotlib.pyplot as plt #匯入matplotlib模組
im=cv.imread("StadiumSnap.jpg") #讀取影像
Real_Data=np.loadtxt("Trajectory.xyz") #讀取XYZ座標資料
K=np.array([[1.28e+03 ,0.00e+00 ,9.60e+02], #建立K矩陣
 [0.00e+00 ,1.28e+03 ,5.40e+02],
 [0.00e+00 ,0.00e+00 ,1.00e+00]])
RT=np.array([[ 6.4278758e-01 ,-7.6604450e-01  ,1.2365159e-08 ,-1.8081500e+02], #建立RT矩陣
 [-1.9826689e-01 ,-1.6636568e-01 ,-9.6592581e-01  ,3.4364292e-01],
 [ 7.3994207e-01  ,6.2088513e-01 ,-2.5881904e-01  ,3.8508780e+02]])
Image_X=[] #建立X座標清單
Image_Y=[] #建立Y座標清單
for i in Real_Data: 
    X,Y,Z=i[0],i[1],i[2] #將資料XYZ座標,用X,Y,Z變數儲存
    Real_points=np.array([[X],[Y],[Z],[1]]) #建立[X,Y,Z,1] 4x1計算矩陣
    Camera_Position=np.matmul(RT,Real_points) #計算 RT*[X,Y,Z,1] 得Camera_Position
    New_Points=np.matmul(K,Camera_Position) #計算K*Camera_Position 得New_Points
    Image_Point=New_Points/New_Points[2]  #New_Points/New_Points[2] <(Z) 得到影像座標
    Image_X.append((Image_Point[0][0])) #將影像座標X儲存至X座標清單
    Image_Y.append((Image_Point[1][0])) #將影像座標Y儲存至Y座標清單
    

im2=cv.cvtColor(im,cv.COLOR_BGR2RGB) #將讀取影像三通道轉換(BGR>RGB)
plt.imshow(im2) #顯示讀取影像
plt.plot(Image_X,Image_Y,"*",color="blue") #繪製各點X,Y座標
plt.axis('off') #關閉X軸以及Y軸
plt.savefig("M11104201.jpg") #儲存影像
plt.show() #顯示圖片
