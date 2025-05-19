#模組匯入 需要安裝模組 CMD輸入pip install opencv-python
import cv2 as cv 
import numpy as np 
import matplotlib.pyplot as plt
fg=cv.imread("fg.jpg") #輸入前景清晰影像
fg_float=fg.astype(np.float32)/255 #前景清晰影像轉換成float格式且介於0~1之間
Laplacian_filter=np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])#建立Laplacian矩陣
fg_gray=cv.cvtColor(fg_float,cv.COLOR_RGB2GRAY)#前景清晰影像轉換成灰階
fg_Hpass=abs(cv.filter2D(fg_gray,-1,Laplacian_filter))#前景做高通濾波,取絕對值
bg=cv.imread("bg.jpg") #輸入後景清晰影像
bg_float=bg.astype(np.float32)/255#背景清晰影像轉換成float格式且介於0~1之間
bg_gray=cv.cvtColor(bg_float,cv.COLOR_RGB2GRAY)#背景清晰影像轉換成灰階
bg_Hpass=abs(cv.filter2D(bg_gray,-1,Laplacian_filter))#背景做高通濾波,取絕對值

mask_fg = fg_Hpass - bg_Hpass#製作前景遮罩 mask = fg_hipass - bg_hipass
mask_fgmean=cv.blur(mask_fg,[15,15])#前景遮罩做均值濾波
ret,mask_fgbinar = cv.threshold(mask_fgmean,0,1, cv.THRESH_BINARY)#前景遮罩二值化

mask_bg = bg_Hpass - fg_Hpass#製作背景遮罩 mask = bg_hipass - fg_hipass
mask_bgmean=cv.blur(mask_bg,[15,15])#背景遮罩做均值濾波
ret,mask_bgbinar = cv.threshold(mask_bgmean,0,1, cv.THRESH_BINARY)#背景遮罩二值化

RGB_fgmask=cv.cvtColor(mask_fgbinar,cv.COLOR_GRAY2BGR)#前景遮罩轉換成RGB三通道
RGB_bgmask=cv.cvtColor(mask_bgbinar,cv.COLOR_GRAY2BGR)#背景遮罩轉換成RGB三通道
Final=RGB_fgmask*fg_float+bg_float*RGB_bgmask#製作景深擴增影像

ret,fg_Hpass = cv.threshold(fg_Hpass,0.3,1, cv.THRESH_BINARY) #將fg_hipass二值化使影像更清晰
ret,bg_Hpass = cv.threshold(bg_Hpass,0.3,1, cv.THRESH_BINARY) #將bg_hipass二值化使影像更清晰


list0=[]#建立空的list (用於隱藏x,y軸刻度)
plt.subplot(3,2,1) 
fg=cv.cvtColor(fg,cv.COLOR_RGB2BGR)#轉換BGR通道至RGB通道(opencv讀取影像通道順序為BGR)
plt.imshow(fg)#顯示圖片1前景清晰影像
plt.xticks(ticks=list0)#隱藏x軸刻度
plt.yticks(ticks=list0)#隱藏y軸刻度
plt.subplot(3,2,2)
bg=cv.cvtColor(bg,cv.COLOR_RGB2BGR)#轉換BGR通道至RGB通道(opencv讀取影像通道順序為BGR)
plt.imshow(bg)#顯示圖片2背景清晰影像
plt.xticks(ticks=list0)#隱藏x軸刻度
plt.yticks(ticks=list0)#隱藏y軸刻度
plt.subplot(3,2,3)
fg_Hpass=cv.cvtColor(fg_Hpass,cv.COLOR_GRAY2BGR)#轉換BGR通道至RGB通道(opencv讀取影像通道順序為BGR)
plt.imshow(fg_Hpass)#顯示圖片3 fg_hipass影像
plt.xticks(ticks=list0)#隱藏x軸刻度
plt.yticks(ticks=list0)#隱藏y軸刻度
plt.subplot(3,2,4)
bg_Hpass=cv.cvtColor(bg_Hpass,cv.COLOR_GRAY2BGR)#轉換BGR通道至RGB通道(opencv讀取影像通道順序為BGR)
plt.imshow(bg_Hpass)#顯示圖片4 bg_hipass影像
plt.xticks(ticks=list0)#隱藏x軸刻度
plt.yticks(ticks=list0)#隱藏y軸刻度
plt.subplot(3,2,5)
plt.imshow(RGB_fgmask)#顯示圖片5 前景遮罩
plt.xticks(ticks=list0)#隱藏x軸刻度
plt.yticks(ticks=list0)#隱藏y軸刻度
plt.subplot(3,2,6)
Final=cv.cvtColor(Final,cv.COLOR_RGB2BGR)#轉換BGR通道至RGB通道(opencv讀取影像通道順序為BGR)
plt.imshow(Final)#顯示圖片6 景深擴增影像
plt.xticks(ticks=list0)#隱藏x軸刻度
plt.yticks(ticks=list0)#隱藏y軸刻度
plt.tight_layout()#使顯示圖片自動排列整齊
plt.show()
