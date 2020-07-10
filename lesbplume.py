 #!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv
import pandas as pd
import time
import sys


# In[2]:


#Transpose of 3x3 matrix
def transpose(x): 
    a11 = x[0,0]
    a12 = x[0,1]
    a13 = x[0,2]
    a21 = x[1,0]
    a22 = x[1,1]
    a23 = x[1,2]
    a31 = x[2,0]
    a32 = x[2,1]
    a33 = x[2,2]
    x = np.array([[a11,a21,a31],[a12,a22,a32],[a13,a23,a33]])
    return x

#multiply of 3x3 matrices
def muiltiply(x,y):
    a11 = x[0,0]*y[0,0]+x[0,1]*y[1,0]+x[0,2]*y[2,0]
    a12 = x[0,0]*y[0,1]+x[0,1]*y[1,1]+x[0,2]*y[2,1]
    a13 = x[0,0]*y[0,2]+x[0,1]*y[1,2]+x[0,2]*y[2,2]
    a21 = x[1,0]*y[0,0]+x[1,1]*y[1,0]+x[1,2]*y[2,0]
    a22 = x[1,0]*y[0,1]+x[1,1]*y[1,1]+x[1,2]*y[2,1]
    a23 = x[1,0]*y[0,2]+x[1,1]*y[1,2]+x[1,2]*y[2,2]
    a31 = x[2,0]*y[0,0]+x[2,1]*y[1,0]+x[2,2]*y[2,0]
    a32 = x[2,0]*y[0,1]+x[2,1]*y[1,1]+x[2,2]*y[2,1]
    a33 = x[2,0]*y[0,2]+x[2,1]*y[1,2]+x[2,2]*y[2,2]
    x = np.array([[a11,a21,a31],[a12,a22,a32],[a13,a23,a33]])
    return x

#trace of 3x3 matrix(AT A)
def trace(x):
    tra = (x[0,0]*x[1,1]*x[2,2])
    return tra


# In[3]:


def timemean(dataw):
    wmean = np.mean(dataw,3)
    wmean = np.squeeze(wmean)
    return wmean
def centermean(dataw):
    wmean = timemean(dataw)
    del dataw
    a = np.int(np.ceil(wmean.shape[0]/2))
    b = np.int(np.floor(wmean.shape[0]/2))
    c = np.int(np.ceil(wmean.shape[1]/2))
    d = np.int(np.floor(wmean.shape[1]/2))
    if(a == b and c == d):                   #even even
        wc1 = wmean[a,c,:]
        wc2 = wmean[a,c-1,:]
        wc3 = wmean[a-1,c-1,:]
        wc4 = wmean[a-1,c,:]
        wc = (wc1+wc2+wc3+wc4)/4
    if(a != b and c == d):                   #odd even
        wc1 = wmean[b,c,:]
        wc2 = wmean[b,c-1,:]
        wc = (wc1+wc2)/2
    if(a == b and c != d):                   #even odd
        wc1 = wmean[b,d,:]
        wc2 = wmean[b-1,d,:]
        wc = (wc1+wc2)/2
    if(a!=b and c!=d):
        wc = wmean[b,d,:]
    return wc       


# In[4]:


def sameshape3(datau,datav,dataw):
    if (datau.shape[0] != datav.shape[0] or datau.shape[0] != dataw.shape[0]
        or datau.shape[1] != datav.shape[1] or datau.shape[1] != dataw.shape[1]
        or datau.shape[2] != datav.shape[2] or datau.shape[2] != dataw.shape[2]
        or datau.shape[3] != datav.shape[3] or datau.shape[3] != dataw.shape[3]):
        print ("Please input uvw files as same shape")
        return False
    else:
        return True


# In[5]:


def sameshape2(datau,datav):
    if (datau.shape[0] != datav.shape[0] or datau.shape[1] != datav.shape[1]
        or datau.shape[2] != datav.shape[2]):
        print ("Please input check if your data files are same shape")
        return False
    else:
        return True


# In[6]:


def shape(dt):
    return dt.shape[0],dt.shape[1],dt.shape[2],dt.shape[3]


# In[7]:


def uvw_wcdless(datau,datav,dataw):
    
    if (sameshape3(datau,datav,dataw) is False):
        return
    nx,ny,nz,nt = shape(datau)
    wc = centermean(dataw)
    U = np.zeros((nx,ny,nz,nt),dtype = np.float32)
    V = np.zeros((nx,ny,nz,nt),dtype = np.float32)
    W = np.zeros((nx,ny,nz,nt),dtype = np.float32)
    for ti in range(0,nt):
        for zi in range(0,nz):
            U[:,:,zi,ti] = (datau[:,:,zi,ti]/wc[zi])                #1
            V[:,:,zi,ti] = (datav[:,:,zi,ti]/wc[zi])
            W[:,:,zi,ti] = (dataw[:,:,zi,ti]/wc[zi])
    return U,V,W


# In[8]:


def dless(data):
    nx,ny,nz,nt = shape(data)
    wc = centermean(data)
    D = np.zeros((nx,ny,nz,nt),dtype = np.float32)
    for ti in range(0,nt):
        for zi in range(0,nz):
            D[:,:,zi,ti] = (data[:,:,zi,ti]/wc[zi])                #1
    return D


# In[9]:


def omega_single(datau,datav,dataw,dx,dy,dz,t):
    percent = np.str(np.round(t/datau.shape[3]*100))+"%"
    print('{}\r'.format(percent), end="")

    gu = np.gradient(datau[:,:,:,t],dx,dy,dz)
    gv = np.gradient(datav[:,:,:,t],dx,dy,dz)
    gw = np.gradient(dataw[:,:,:,t],dx,dy,dz)
    
  
    GV = np.array([[gu[1],gv[1],gw[1]],[gu[0],gv[0],gw[0]],[gu[2],gv[2],gw[2]]])
    GVT = transpose(GV)

    A = 0.5*(GV+GVT)
    B = 0.5*(GV-GVT)
    AT = transpose(A)
    BT = transpose(B)
    ATA = muiltiply(AT,A)
    BTB = muiltiply(BT,B)
    a = trace(ATA)
    b = trace(BTB)
    ef=0.001*np.max(b-a)
    result = (b/(a+b+ef))
    return result


# In[10]:


def omega(datau,datav,dataw,dx,dy,dz):
#     print('hi')
    if (sameshape3(datau,datav,dataw) is False):
        return
    np.seterr(divide='ignore', invalid='ignore')
#     datau,datav,dataw = uvw_wcdless(datau,datav,dataw)
    
    ome=np.zeros((datau.shape[0],datau.shape[1],datau.shape[2],datau.shape[3]),dtype=np.float32)
    for t in range(0,datau.shape[3]):
        ome[:,:,:,t] = omega_single(datau,datav,dataw,dx,dy,dz,t)
    ome = np.nan_to_num(ome)
    print('Omega Vortex Identification Completed')
    return ome


# In[11]:


# Memories problems
def vorticity(datau,datav,dataw,dx,dy,dz,dt):
    if (sameshape3(datau,datav,dataw) is False):
        return
    print("To calculate vorticity need at least 32gbs of RAM.")
    gu = np.gradient(datau,dx,dy,dz,dt)
    gu1 = np.array(gu[1],dtype =np.float32)
    gu2 = np.array(gu[2],dtype =np.float32)
    del gu
    
    gv = np.gradient(datav,dx,dy,dz,dt)
    gv0 = np.array(gv[0],dtype =np.float32)
    gv2 = np.array(gv[2],dtype =np.float32)
    del gv
    
    gw = np.gradient(dataw,dx,dy,dz,dt)
    gw0 = np.array(gw[0],dtype =np.float32)
    gw1 = np.array(gw[1],dtype =np.float32)
    del gw
    
    omgx = 0.5*(gw1-gv2)
    del gw1, gv2
    omgy = 0.5*(gu2-gw0)
    del gu2, gw0
    omgz = 0.5*(gv0-gu1)
    del gv0,gu1
    
    vor = np.array([2*omgx,2*omgy,2*omgz])
    return vor


# In[12]:


def plumehalfgraph(dataw,datat,threshold,dx,dy,dz,D,name):
    import matplotlib.pyplot as plt
    if (sameshape2(dataw,datat) is False):
        return
    nx,ny,nz,nt = shape(datat)
    wmean = timemean(dataw)
    tmean = timemean(datat)
    wc = centermean(dataw)
    tc = centermean(datat)
    
    lt = np.zeros((nz,1),dtype=np.float32)
    zt = np.zeros((nz,1),dtype=np.float32)
    
    x = np.arange(-((nx/2)-1)*dx-(dx/2),(nx/2)*dx,dx)
    y = x
    z = np.arange(0,nz*dz,dz)
    [X,Y,Z] = np.meshgrid(x,y,z);
    
    a = np.int(np.ceil(ny/2))
    b = np.int(np.floor(ny/2))
    if(a==b): #even
        cy=a-1
    else:     #odd
        cy=b
    
    for zi in range(0,nz):
        if (tmean[nx-1,cy,zi]) >= (threshold*tc[zi]):
            lt[zi] = x[nx-1]
        else:
            for xi in range(cy,ny):
                if (tmean[xi,cy,zi]) >= (threshold*tc[zi]):
                    c = (tmean[xi,cy,zi] - (threshold*tc[zi]))/(tmean[xi,cy,zi]-tmean[(xi+1),cy,zi])
                    lt[zi] = x[xi] +c*dx;         

    tmeandls = np.zeros(nx*ny*nz,dtype = np.float32)
    tmeandls = np.reshape(tmeandls, (nx,ny,nz))
    xdlst = np.zeros(nx*nz,dtype = np.float32)
    xdlst = np.reshape(xdlst, (ny,nz))
    ydlst = xdlst
    for zi in range(0,nz):
        tmeandls[:,:,zi] = (tmean[:,:,zi]/tc[zi])
        if (lt[zi]!=0):
            xdlst[:,zi] = x[:]/lt[zi]
            ydlst[:,zi] = y[:]/lt[zi]

    graphxt = []
    graphyt = []
    for i in range(0,len(lt)):
        if (lt[i] != 0):
            graphxt = np.append(graphxt, ((i+1)*dz/D))
            graphyt = np.append(graphyt,lt[i]/D)

    graphxt = np.reshape(graphxt, (len(graphxt), 1))
    graphyt = np.reshape(graphyt, (len(graphyt), 1))

    lw = np.zeros((nz,1),dtype=np.float32)
    zw = np.zeros((nz,1),dtype=np.float32)
    for zi in range(0,nz):
        if (wmean[nx-1,cy,zi]) >= (threshold*wc[zi]):
            lw[zi] = x[nx-1]
        else:
            for xi in range(cy,nx):
                if (wmean[xi,cy,zi]) >= (threshold*wc[zi]):
                    c = (wmean[xi,cy,zi] - (threshold*wc[zi]))/(wmean[xi,cy,zi]-wmean[(xi+1),cy,zi])
                    lw[zi] = x[xi] +c*dx;         
    wmeandls = np.zeros(nx*ny*nz,dtype = np.float32)
    wmeandls = np.reshape(wmeandls, (nx,ny,nz))
    xdlsw = np.zeros(nx*nz,dtype = np.float32)
    xdlsw = np.reshape(xdlsw, (ny,nz))
    ydlsw = xdlsw
    for zi in range(0,nz):
        wmeandls[:,:,zi] = (wmean[:,:,zi]/wc[zi])
        if (lw[zi]!=0):
            xdlsw[:,zi] = x[:]/lw[zi]
            ydlsw[:,zi] = y[:]/lw[zi]

    graphxw = []
    graphyw = []
    for i in range(0,len(lw)):
        if (lw[i] != 0):
            graphxw = np.append(graphxw, ((i+1)*dz/D))
            graphyw = np.append(graphyw,lw[i]/D)

    graphxw = np.reshape(graphxw, (len(graphxw), 1))
    graphyw = np.reshape(graphyw, (len(graphyw), 1))
    plt.figure(figsize=(10,10))
    plt.ylabel('Z/D')
    plt.xlabel('halfwidth')
    plt.title(name +' plume-half graph')# giving a title to my graph
    plt.plot(graphyt, graphxt,label='Tmean')
    plt.plot(graphyw, graphxw,label='Wmean')
    plt.legend()
    plt.savefig(name+'.jpg')
    plt.show()
    return


# In[13]:


def topviewmap(data,datatype,filename,dx=45,dy=45,dz=10,D=400,frame=20):
    import matplotlib.pyplot as plt
    import seaborn as sns; sns.set()
    from matplotlib import animation
    
    nx,ny,nz,nt = shape(data)
    data = np.mean(data,3)
    datadless = np.zeros((nx,ny,nz),dtype = np.float32) #created a empty dimensionless data
    
    #solving for WC
    a = np.int(np.ceil(data.shape[0]/2))
    b = np.int(np.floor(data.shape[0]/2))
    c = np.int(np.ceil(data.shape[1]/2))
    d = np.int(np.floor(data.shape[1]/2))
    if(a == b and c == d):                   #even even
        wc1 = data[a,c,:]
        wc2 = data[a,c-1,:]
        wc3 = data[a-1,c-1,:]
        wc4 = data[a-1,c,:]
        wc = (wc1+wc2+wc3+wc4)/4
    if(a != b and c == d):                   #odd even
        wc1 = data[b,c,:]
        wc2 = data[b,c-1,:]
        wc = (wc1+wc2)/2
    if(a == b and c != d):                   #even odd
        wc1 = data[b,d,:]
        wc2 = data[b-1,d,:]
        wc = (wc1+wc2)/2
    if(a!=b and c!=d):
        wc = data[b,d,:]
        
    #Dimensionless Data divied by WC
    for zi in range(0,nz):
        datadless[:,:,zi] = (data[:,:,zi]/wc[zi])
        
    wmax = datadless[:,:,np.int(np.round(nz*0.15)):np.int(np.round(nz*0.85))].max()
    wmin = datadless[:,:,np.int(np.round(nz*0.15)):np.int(np.round(nz*0.85))].min()
    
    fig = plt.figure(figsize=(10,10))
    data = datadless[:,:,0]
    ax = plt.axes()
    sns.heatmap(data,vmin=wmin,vmax=wmax,fmt='.1f',cbar_kws={'label': datatype})

    xaxis = []
    for x in range (0,nx+1,5):
        xaxis.append(x*dx/D)
    yaxis = []
    for y in range (0,ny+1,5):
        yaxis.append(y*dy/D)
        
    def init():
        sns.heatmap(np.zeros((nx, ny)),vmin=wmin,vmax=wmax,fmt='.1f',cbar = False)

    def animate(i):
        i = i +3
        data = datadless[:,:,i*frame]
        sns.heatmap(data,vmin=wmin,vmax=wmax,fmt='.1f',cbar = False)
        ax.set_title('Dimensionless ' + datatype+' '+filename+', z = '+str(round(i*frame*dz/D,2)),fontsize = 20)
        ax.set_ylabel('Y/D',fontsize = 20)
        ax.set_xlabel('X/D',fontsize = 20)
        ax.set_xticks(np.linspace(0, dx, dz))
        ax.set_xticklabels(xaxis)
        ax.set_yticks(np.linspace(0, dx, dz))
        ax.set_yticklabels(yaxis)
        ax.invert_yaxis()
    fr = np.int(np.round(nz/frame*0.9))
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames= fr, interval=400,repeat = False)

    # fig.tight_layout()
    anim.save(filename+'.gif', writer='matplotlib.animation.PillowWritter')


# In[14]:
def lamb2sec(datau,datav,dataw,dx,dy,dz,nt):
    import multifluidlab
    import multiprocessing as mp
    from multiprocessing import Pool
    nx,ny,nz,nt = shape(datau)
    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(multifluidlab.lambda2,[(datau,datav,dataw,t,dx,dy,dz) for t in range(nt)])
    lamb2=np.zeros((nx,ny,nz,nt),dtype=np.float32)
    for i in range(nt):
        lamb2[:,:,:,i] = results[i]     
    return lamb2

def lambda2multicore(datau,datav,dataw,dx=40,dy=40,dz=10):
    import multiprocessing as mp
    if (sameshape3(datau,datav,dataw) is False):
        return
    nx,ny,nz,nt = shape(datau)
    lamb2=np.zeros((nx,ny,nz,nt),dtype=np.float32)
    sectime =np.int(mp.cpu_count()*10)
    nts = 0
    nte = nts+sectime
    while (nts < nt):
        u = datau[:,:,:,nts:nte]
        v = datav[:,:,:,nts:nte]
        w = dataw[:,:,:,nts:nte]
        temp = lamb2sec(u,v,w,dx,dy,dz,nte-nts)
        for i in range (nts,nte):
            j = i-nts
            lamb2[:,:,:,i] = temp[:,:,:,j]
        nts = nte 
        nte = nts + sectime
        if (nte > nt):
            nte = nt   
    return lamb2

def lambda2single(datau,datav,dataw,t,dx=40,dy=40,dz=10,dt=10):
    from numpy import linalg as LA
    nx,ny,nz,nt = shape(datau)         
    percent = np.str(np.round(t/nt*100))+"%"
    print('{}\r'.format(percent), end="")
    print(t)
        
    dudy,dudx,dudz= np.gradient(datau[:,:,:,t],dx,dy,dz)
    dvdy,dvdx,dvdz = np.gradient(datav[:,:,:,t],dx,dy,dz)
    dwdy,dwdx,dwdz = np.gradient(dataw[:,:,:,t],dx,dy,dz)

    J = np.array([[dudx,dudy,dudz],[dvdx,dvdy,dvdz],[dwdx,dwdy,dwdz]])
    JT = transpose(J)
    S = (J+JT)*0.5
    Ome = (J-JT)*0.5
    lamb2 = lambmatrix(S,Ome,nx,ny,nz)
    return lamb2
def lambda2(datau,datav,dataw,dx=40,dy=40,dz=10):
    if (sameshape3(datau,datav,dataw) is False):
        return
    nx,ny,nz,nt = shape(datau)
    for i in range(nt):
        lamb2[:,:,:,i] = lambda2single(datau,datav,dataw,t,dx,dy,dz)
    print('{}\r'.format("Lambda2 Vortex Identification Completed "), end="")
    return lamb2
def cart2pol(datau,datav,dataw,dx,dy,dz):

    #cal nx ny nz nt
    if (sameshape3(datau,datav,dataw) is False):
        return
    nx,ny,nz,nt = shape(datau)
    
    #creating x y z 
    x = np.linspace(start = -((nx-1)/2)*dx, stop = ((nx-1)/2)*dx, num = nx)
    y = np.linspace(start = -((ny-1)/2)*dy, stop = ((ny-1)/2)*dy, num = ny)
    z = np.linspace(start = 0, stop = (nz-1)*dz, num = nz)
    
    r = np.sqrt(x*x+y*y)
    np.seterr(divide='ignore', invalid = 'ignore')
    theta = np.arctan(y/x)
    theta = np.nan_to_num(theta)
    
    theta = np.zeros((nx,ny,nz), dtype = np.float32)
    r = np.zeros((nx,ny,nz), dtype = np.float32)
    zpol = np.zeros((nx,ny,nz), dtype = np.float32)
    np.seterr(divide='ignore', invalid = 'ignore')
    for i in range (nx):
        for j in range (ny):
            for k in range (nz):
                r[i,j,k] = np.sqrt(x[i]*x[i]+y[j]*y[j])
                zpol[i,j,k] = z[k]
                theta[i,j,k] = np.arctan(y[j]/x[i])
    theta = np.nan_to_num(theta)
    Ur = np.zeros((nx,ny,nz,nt), dtype = np.float32)
    Utheta = np.zeros((nx,ny,nz,nt), dtype = np.float32)
    
    for t in range(nt):
        u = datau[:,:,:,t]
        v = datav[:,:,:,t]
        Ur[:,:,:,t] = u*np.cos(theta)+v*np.sin(theta)
        Utheta[:,:,:,t] = -u*np.sin(theta)+v*np.cos(theta)
        
    return Ur, Utheta, dataw 


#saving data into csvfile
def savecsv(filedata,nametypedata):
    import time
    tic = time.perf_counter()
    import csv
    filename = nametypedata+'.csv'
    nx,ny,nz,nt = filedata.shape[0],filedata.shape[1],filedata.shape[2],filedata.shape[3]
    filedata = np.reshape(filedata, (nx*ny*nz*nt,1), order="F")
    # writing to csv file  
    with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile) 
        # writing the data rows   
        csvwriter.writerows(filedata)

    toc1 = time.perf_counter()
    print(f"Saved"+filename+"Time: {((toc1 - tic)/60):0.4f} minutes")

## Video Generating function 
def generate_video(path,videopath = 'None',speed = 5):
    print('Please install cv2 library by run line  (pip install opencv-python) in the command prompt')
    import os 
    import cv2 
    import numpy as np
    image_folder = '.' # make sure to use your folder 
    end = len(path)
    j = 0
    for i in range (end):
        if (path[i] == "\\" ):
            j = i +1
    if (j!=0):
        filename = path[j:end]
    else:
        filename = path
    video_name = filename + '.avi'
    os.chdir(path) 
      
    images = [img for img in os.listdir(image_folder) 
              if img.endswith(".jpg") or
                 img.endswith(".jpeg") or
                 img.endswith("png")] 
     
    # Array images should only consider 
    # the image files ignoring others if any 
#     print(images)  
  
    frame = cv2.imread(os.path.join(image_folder, images[0])) 
  
    # setting the frame width, height width 
    # the width, height of first image 
    height, width, layers = frame.shape   
    speed = np.int(speed)
    if (videopath != 'None'):
        video_name = videopath +'\\'+video_name
#         if not os.path.exists(videopath):
#             os.makedirs(videopath)
    video = cv2.VideoWriter(video_name, 0, speed, (width, height))  
  
    # Appending the images to the video one by one 
    for image in images:  
        video.write(cv2.imread(os.path.join(image_folder, image)))  
      
    # Deallocating memories taken for window creation 
    cv2.destroyAllWindows()  
    video.release()  # releasing the video generated 


    
def isosurface(data,isovalue,name, dpi =30,frame = 5,angle = 60,dx=40,dy=40,dz=10,D=400):
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib import animation
    from skimage import measure
    if not os.path.exists(name):
        os.makedirs(name)
    nx, ny, nz, nt = shape(data)
    t = 0 
    fig = plt.figure(figsize=(30, 30*2.5))
    ax = fig.add_subplot(111, projection='3d',facecolor='w',label='Inline label')
    while(t<nt):
        
        percent = np.str(np.round(t/nt*100))+"%"
        print('{}\r'.format(percent), end="")
        title_name  = name+' Time:'+str(t)
        
        fr = 0
        vol = data[:,:,:,t]
        datamax = vol.max()
        datamin = vol.min()
        while (isovalue>=datamax or isovalue <=datamin):
            t = t+1
            fr = fr+1
            if (fr==frame):
                fr = 0
            vol = data[:,:,:,t]
            datamax = vol.max()
            datamin = vol.min()

        verts, faces, _, _ = measure.marching_cubes_lewiner(vol, isovalue, spacing=(dx, dy, dz))

        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        ax.view_init(15,angle)

        ax.set_xlabel("X/D",fontsize = 50,labelpad=40)
        ax.set_ylabel("Y/D",fontsize = 50,labelpad=40)
        ax.set_zlabel("Z/D",fontsize = 50,labelpad=100)
        # Title:
        namel = len(title_name)
        namexpos = 0.5-0.01*namel   
        ax.text2D(namexpos, 0.85, title_name, transform=ax.transAxes,fontsize = 65)

        ticnum = 11
        ticnumz = 14
        xaxis = []
        for x in range (np.int(-(ticnum-1)/2),np.int((ticnum+1)/2)):
            xaxis.append(((nx-1)*dx/(ticnum-1)*x)/D)
        yaxis = []
        for y in range (np.int(-(ticnum-1)/2),np.int((ticnum-1)/2)):
            yaxis.append(((ny-1)*dy/(ticnum-1)*y)/D)
        zaxis = []
        for z in range (0,np.int((ticnumz+1))):
            zaxis.append(z*(dz*nz/ticnumz)/D)
        ax.set_xticks(np.linspace(0, nx*dx, ticnum))
        ax.set_yticks(np.linspace(0, ny*dy, ticnum))
        ax.set_xticklabels(xaxis)
        ax.set_yticklabels(yaxis)
        ax.invert_yaxis()
        ax.set_zticks(np.linspace(0, nz*dz, ticnumz+1))
        ax.set_zticklabels(zaxis) 
        ax.tick_params(axis='both', which='major', labelsize=30)
        plt.tight_layout()
        

        bbox = fig.bbox_inches.from_bounds(1, 9, 28,58 )
        if (t <10):
            picname = '00'+str(t)
        if (t >=10 and t<100):
            picname = '0'+str(t)
        if (t >= 100):
            picname = str(t)
        filename=name+'/'+picname+'.png'
        plt.savefig(filename, bbox_inches=bbox,dpi = dpi)
        plt.cla()
        if (fr==0):
            fr = frame
        else:
            fr = frame - fr
        t = t + fr 
    print('Done.') 
      
    
def isosurface_timestep(data,timestep,isovalue,name, dpi =30,frame = 9,dx=40,dy=40,dz=10,D=400):
    import os
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib import animation
    from skimage import measure
    nx, ny, nz, nt = shape(data)
    if not os.path.exists(name+'_'+str(timestep)):
        os.makedirs(name+'_'+str(timestep))
    fig = plt.figure(figsize=(30, 30*2.5))
    ax = fig.add_subplot(111, projection='3d',facecolor='w',label='Inline label')
    angle = 0 
    while(angle<=360):
        
        percent = np.str(np.round(angle/360*100))+"%"
        print('{}\r'.format(percent), end="")
        
        nameg  = name+' Time:'+str(timestep)+' Angle:'+str(angle)
        
        vol = data[:,:,:,timestep]
        verts, faces, _, _ = measure.marching_cubes_lewiner(vol, isovalue, spacing=(dx, dy, dz))
        
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        ax.view_init(15,angle)

        ax.set_xlabel("X/D",fontsize = 50,labelpad=40)
        ax.set_ylabel("Y/D",fontsize = 50,labelpad=40)
        ax.set_zlabel("Z/D",fontsize = 50,labelpad=100)
        # Title:
        namel = len(nameg)
        namexpos = 0.5-0.01*namel   
        ax.text2D(namexpos, 0.85, nameg, transform=ax.transAxes,fontsize = 65)

        ticnum = 11
        ticnumz = 14
        xaxis = []
        for x in range (np.int(-(ticnum-1)/2),np.int((ticnum+1)/2)):
            xaxis.append(((nx-1)*dx/(ticnum-1)*x)/D)
        yaxis = []
        for y in range (np.int(-(ticnum-1)/2),np.int((ticnum-1)/2)):
            yaxis.append(((ny-1)*dy/(ticnum-1)*y)/D)
        zaxis = []
        for z in range (0,np.int((ticnumz+1))):
            zaxis.append(z*(dz*nz/ticnumz)/D)
        ax.set_xticks(np.linspace(0, nx*dx, ticnum))
        ax.set_yticks(np.linspace(0, ny*dy, ticnum))
        ax.set_xticklabels(xaxis)
        ax.set_yticklabels(yaxis)
        ax.invert_yaxis()
        ax.set_zticks(np.linspace(0, nz*dz, ticnumz+1))
        ax.set_zticklabels(zaxis) 
        ax.tick_params(axis='both', which='major', labelsize=30)

        plt.tight_layout()
        filename=name+'_T'+str(timestep)+'/'+name+'_t'+str(angle)+'.png'

        bbox = fig.bbox_inches.from_bounds(1, 9, 28,58 )
        plt.savefig(name+'_('+str(timestep)+')/'+name+' '+str(angle)+'.png', bbox_inches=bbox, dpi=dpi)
        plt.cla()
        angle = angle + frame
        
def uvw_import(path,filename,start,end,T=False,nx=45,ny=45,nz=700):
    import pandas as pd #reading data from csv
    import pandas as pd #reading data from csv
    sk = start*nx*ny*nz
    nt = end - start
    n = np.int(nx*ny*nz*nt)
    dfu = pd.read_csv(path+'\\'+filename+'_U.csv',skiprows = sk,nrows = 2, header = None)
    check = str(dfu[0][0])
    if(check.isnumeric()):
        print('')
    else:
        sk = sk+1

    #T data
    if (T==True):
        dft = pd.read_csv(path+'\\'+filename+'_T.csv',skiprows = sk,nrows = n, dtype =np.float32, header = None)
        datat = dft.to_numpy()
        datat = np.reshape(datat, (nx,ny,nz,nt), order="F") #reshaping data to (45,45,700,timestep) in the "F" order
        del dft
    else:
        datat = None

    # U data
    dfu = pd.read_csv(path+'\\'+filename+'_U.csv',skiprows = sk,nrows = n, dtype =np.float32, header = None)
    datau = dfu.to_numpy()
    datau = np.reshape(datau,  (nx,ny,nz,nt), order="F")
    del dfu

    # V data
    dfv = pd.read_csv(path+'\\'+filename+'_V.csv',skiprows = sk,nrows = n, dtype =np.float32, header = None)
    datav = dfv.to_numpy()
    datav = np.reshape(datav,  (nx,ny,nz,nt), order="F")
    del dfv

    # W data
    dfw = pd.read_csv(path+'\\'+filename+'_U.csv',skiprows = sk,nrows = n, dtype =np.float32, header = None)
    dataw = dfw.to_numpy()
    dataw = np.reshape(dataw,  (nx,ny,nz,nt), order="F")
    del dfw
    
    return datau,datav,dataw,datat

def datasingle_import(path,filename,start,end,nx=45,ny=45,nz=700):
    import pandas as pd #reading data from csv
    import pandas as pd #reading data from csv
    sk = start*nx*ny*nz
    nt = end - start
    n = np.int(nx*ny*nz*nt)
    df = pd.read_csv(path+'\\'+filename+'.csv',skiprows = sk,nrows = 2, header = None)
    check = str(int(df[0][0]))
    if(check.isnumeric()):
        print('')
    else:
        sk = sk+1
    df = pd.read_csv(path+'\\'+filename+'.csv',skiprows = sk,nrows = n, dtype =np.float32, header = None)
    data = df.to_numpy()
    data = np.reshape(data,  (nx,ny,nz,nt), order="F")
    del df
    return data

def uvw_import(path,filename,start,end,nx=45,ny=45,nz=700):
    import pandas as pd #reading data from csv
    import pandas as pd #reading data from csv
    sk = start*nx*ny*nz
    nt = end - start
    n = np.int(nx*ny*nz*nt)
    dfu = pd.read_csv(path+'\\'+filename+'_U.csv',skiprows = sk,nrows = 2, header = None)
    check = str(dfu[0][0])
    if(check.isnumeric()):
        print('')
    else:
        sk = sk+1

    # U data
    dfu = pd.read_csv(path+'\\'+filename+'_U.csv',skiprows = sk,nrows = n, dtype =np.float32, header = None)
    datau = dfu.to_numpy()
    datau = np.reshape(datau,  (nx,ny,nz,nt), order="F")
    del dfu

    # V data
    dfv = pd.read_csv(path+'\\'+filename+'_V.csv',skiprows = sk,nrows = n, dtype =np.float32, header = None)
    datav = dfv.to_numpy()
    datav = np.reshape(datav,  (nx,ny,nz,nt), order="F")
    del dfv

    # W data
    dfw = pd.read_csv(path+'\\'+filename+'_U.csv',skiprows = sk,nrows = n, dtype =np.float32, header = None)
    dataw = dfw.to_numpy()
    dataw = np.reshape(dataw,  (nx,ny,nz,nt), order="F")
    del dfw
    
    return datau,datav,dataw

def datasingle_import(path,filename,start,end,nx=45,ny=45,nz=700):
    import pandas as pd #reading data from csv
    import pandas as pd #reading data from csv
    sk = start*nx*ny*nz
    nt = end - start
    n = np.int(nx*ny*nz*nt)
    df = pd.read_csv(path+'\\'+filename+'.csv',skiprows = sk,nrows = 2, header = None)
    check = str(int(df[0][0]))
    if(check.isnumeric()):
        print('')
    else:
        sk = sk+1
    df = pd.read_csv(path+'\\'+filename+'.csv',skiprows = sk,nrows = n, dtype =np.float32, header = None)
    data = df.to_numpy()
    data = np.reshape(data,  (nx,ny,nz,nt), order="F")
    del df
    return data