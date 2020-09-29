from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
import cv2
import numpy as np
#from sklearn.model_selection import GridSearchCV
#import warnings
#warnings.filterwarnings('ignore')

def PDF(m,cov,x):
  var = multivariate_normal(m, cov)
  return var.pdf(x)

#KL Divergence funtion exporting variance
def KL_divergence(Xf,Xb):
  bw=0
  value=-1000
  for i in range(1,1000):
    kl_div=0;s=0
    c=i/100
    Y1=np.reshape(Xf,(10,1))
    Y2=np.reshape(Xb,(10,1))
    kde1 = KernelDensity(kernel='gaussian', bandwidth=c).fit(Y1)
    kde2 = KernelDensity(kernel='gaussian', bandwidth=c).fit(Y2)
    score1 = kde1.score_samples(Y1)
    score2 = kde2.score_samples(Y2)
    for j in range(10):
       pdf=np.exp(score1[j])
       #Normalized KL-Divergence
       kl_div=kl_div+pdf*np.log(np.exp(score1[j])/np.exp(score2[j]))
       s=s+pdf
    if (kl_div/s)>value:
        value=kl_div/s
        bw=c
  sd=np.round(bw*(20**(0.2))/1.06,5)      
  print("Optimized bandwidth from KL Divergence method is ",np.round(sd,5))
  return sd*sd    
   
img = cv2.imread("image.png")
image = cv2.resize(img, (100,80))
imgnew=cv2.resize(img, (100,80))
[r,g,s]=[np.zeros((80,100)),np.zeros((80,100)),np.zeros((80,100))] 
R,G,B = cv2.split(image) 
#r g s convertion
for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        s[i,j] = (np.asscalar(R[i,j])+np.asscalar(G[i,j])+np.asscalar(B[i,j]))/3
        r[i,j] = np.asscalar(R[i,j])/(np.asscalar(R[i,j])+np.asscalar(G[i,j])+np.asscalar(B[i,j]))
        g[i,j] = np.asscalar(G[i,j])/(np.asscalar(R[i,j])+np.asscalar(G[i,j])+np.asscalar(B[i,j]))


xf=np.arange(45,55,1)
yf=np.arange(35,45,1)
xb=np.arange(0,10,1)
yb=np.arange(0,10,1)

Xrf=list(map(lambda xf,yf: r[xf][yf] , xf,yf))
Xgf=list(map(lambda xf,yf: g[xf][yf] , xf,yf))
Xsf=list(map(lambda xf,yf: s[xf][yf] , xf,yf))

Xrb=list(map(lambda xb,yb: r[xb][yb] , xb,yb))
Xgb=list(map(lambda xb,yb: g[xb][yb] , xb,yb))
Xsb=list(map(lambda xb,yb: s[xb][yb] , xb,yb))
#KL Divergence function
print("For r :")
var1=KL_divergence(Xrf,Xrb)
print("For g :")
var2=KL_divergence(Xgf,Xgb)
print("For s :")
var3=KL_divergence(Xsf,Xsb)
fg=[]
bg=[]
cov=np.array([[var1,0,0],[0,var2,0],[0,0,var3]])
#Adding samples
#Equation 2
for i in range(10):
    fsum=0;bsum=0  
    Xfg=[Xrf[i],Xgf[i],Xsf[i]]
    Xbg=[Xrb[i],Xgb[i],Xsb[i]]
    for j in range(10):
        m1=[Xrf[j],Xgf[j],Xsf[j]]
        m2=[Xrb[j],Xgb[j],Xsb[j]]
        fsum=fsum+PDF(m1,cov,Xfg)
        bsum=bsum+PDF(m2,cov,Xbg)
    fg.append(fsum/10)
    bg.append(bsum/10)    
#print(fg)    
#print(bg)
    
#Equation 4 and 5  
for i in range(80):
    for j in range(100):
      sum1=0;sum2=0
      x=[r[i][j],g[i][j],s[i][j]]
      for k in range(10):
         m1=[Xrf[k],Xgf[k],Xsf[k]]
         m2=[Xrb[k],Xgb[k],Xsb[k]]
         sum1=sum1+(PDF(m1,cov,x))*fg[k]
         sum2=sum2+(PDF(m2,cov,x))*bg[k]
      #sum2=wbg,sum1=wfg  
      if sum1<sum2:
          [imgnew[i][j][0],imgnew[i][j][1],imgnew[i][j][2]]=[255,255,255]
          
cv2.imshow("original image",image)          
cv2.imshow("Editted",imgnew)
cv2.waitKey(0)
