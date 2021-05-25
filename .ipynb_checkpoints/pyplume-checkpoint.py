 #!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
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
def vorticity(datau,datav,dataw,dx=40,dy=40,dz=10,dt=10):
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

def plumehalfgraph(dataw,datat,name,threshold=0.36788,dx=40,dy=40,dz=10,D=400):
# def plumehalfgraph(dataw,datat,threshold,name,dx,dy,dz,D):
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
    plt.title(name + 'plume-half graph')# giving a title to my graph
    plt.plot(graphyt, graphxt,label='Tmean')
    plt.plot(graphyw, graphxw,label='Wmean')
    plt.legend()
    plt.savefig(name+'.jpg')
    plt.show()
    return


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
#     sectime =np.int(mp.cpu_count()*6)
    sectime =24
    nts = 0
    nte = nts+sectime
    while (nts < nt):
        print(nts)
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
    import multifluidlab
    nx,ny,nz,nt = shape(datau)        
        
    dudy,dudx,dudz= np.gradient(datau[:,:,:,t],dx,dy,dz)
    dvdy,dvdx,dvdz = np.gradient(datav[:,:,:,t],dx,dy,dz)
    dwdy,dwdx,dwdz = np.gradient(dataw[:,:,:,t],dx,dy,dz)

    J = np.array([[dudx,dudy,dudz],[dvdx,dvdy,dvdz],[dwdx,dwdy,dwdz]])
    JT = transpose(J)
    S = (J+JT)*0.5
    Ome = (J-JT)*0.5
    lamb2 = multifluidlab.lambmatrix(S,Ome,nx,ny,nz)
    return lamb2

def lambda2(datau,datav,dataw,dx,dy,dz):
    if (sameshape3(datau,datav,dataw) is False):
        return
    nx,ny,nz,nt = shape(datau)
    lamb2 = np.zeros((nx,ny,nz,nt),dtype=np.float32)
    for i in range(nt):
        percent = np.str(np.round(i/nt*100))+"%"
        print('{}\r'.format(percent), end="")
        lamb2[:,:,:,i] = lambda2single(datau,datav,dataw,i,dx,dy,dz)
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
def savecsv(data,nametypedata):
    import time
    tic = time.perf_counter()
    import csv
    filename = nametypedata+'.csv'
    nx,ny,nz,nt = data.shape[0],data.shape[1],data.shape[2],data.shape[3]
    data = np.reshape(data, (nx*ny*nz*nt,1), order="F")
    # writing to csv file  
    with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile) 
        # writing the data rows   
        csvwriter.writerows(data)

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
     

    frame = cv2.imread(os.path.join(image_folder, images[0])) 

    height, width, layers = frame.shape   
    speed = np.int(speed)
    if (videopath != 'None'):
        video_name = videopath +'\\'+video_name
    video = cv2.VideoWriter(video_name, 0, speed, (width, height))  
    for image in images:  
        video.write(cv2.imread(os.path.join(image_folder, image)))  
    cv2.destroyAllWindows()  
    video.release() 


    
def isosurface(data,isovalue,name, dpi =30,frame = 5,angleH = 15, angleV = 60,dx=40,dy=40,dz=10,D=400):
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
        ax.view_init(angleH,angleV)

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

def loadcsv_partial(filename,start,end,nx=45,ny=45,nz=700):
    import pandas as pd #reading data from csv
    sk = start*nx*ny*nz
    nt = end - start
    n = np.int(nx*ny*nz*nt)
    df = pd.read_csv(filename,skiprows = sk,nrows = 2, header = None)
    check = df[0][0]
    if(type(check) == str):
        sk = sk+1
    df = pd.read_csv(filename,skiprows = sk,nrows = n, dtype =np.float32, header = None)
    data = df.to_numpy()
    data = np.reshape(data,  (nx,ny,nz,nt), order="F")
    del df
    return data

def loadcsv(filename,nx=45,ny=45,nz=700):
    import pandas as pd #reading data from csv
    sk = 0
    df = pd.read_csv(filename,skiprows = sk,nrows = 2, header = None)
    check = df[0][0]
    try:
        check = check+1
    except:
        sk = 1
    df = pd.read_csv(filename,skiprows = sk, dtype =np.float32, header = None)
    data = df.to_numpy()
    datasize = np.shape(data)[0]
    nt = int(datasize /nx/ny/nz)
    data = np.reshape(data,  (nx,ny,nz,nt), order="F")
    del df
    return data

def loadmat(filename):
    from scipy import io
    import numpy as np
    import mat73
    try:
        data = io.loadmat(filename)
        key = sorted(data.keys(),reverse=True)[0]
        data = data[key]
        data = np.array(data)
    except:
        data = mat73.loadmat(filename)
        key = sorted(data.keys(),reverse=True)[0]
        data = data[key]
        data = np.array(data)
    return data

def savemat(data,filename):
    from scipy import io
    try:
        io.savemat(filename+".mat", {"data": data})
    except:
        data = np.array(data,dtype=np.float32)
        io.savemat(filename+".mat", {"data": data})

def csv2mat(filename,nx,ny,nz):
    data = loadcsv(filename,nx,ny,nz)
    savemat(data,filename)

def helicity(datau,datav,dataw,dx=40,dy=40,dz=10):
    if (sameshape3(datau,datav,dataw) is False):
        return
    np.seterr(divide='ignore', invalid='ignore')
    nx, ny, nz, nt = shape(datau)
    H_n = np.zeros((nx,ny,nz,nt), dtype = np.float32)
    H=H_n
    for t in range (nt):
        percent = np.str(np.round(t/datau.shape[3]*100))+"%"
        print('{}\r'.format(percent), end="")
        dudy,dudx,dudz= np.gradient(datau[:,:,:,t],dx,dy,dz)
        dvdy,dvdx,dvdz = np.gradient(datav[:,:,:,t],dx,dy,dz)
        dwdy,dwdx,dwdz = np.gradient(dataw[:,:,:,t],dx,dy,dz)
        u = datau[:,:,:,t]
        v = datav[:,:,:,t]
        w = dataw[:,:,:,t]
        Vorticity_x = dwdy - dvdz
        Vorticity_y= dudz - dwdx
        Vorticity_z= dvdx - dudy
        H_n[:,:,:,t]=(u*Vorticity_x+v*Vorticity_y+w*Vorticity_z)/(np.sqrt(u**2+v**2+w**2)*np.sqrt(Vorticity_x**2+Vorticity_y**2+Vorticity_z**2));
        H[:,:,:,t]=(u*Vorticity_x+v*Vorticity_y+w*Vorticity_z);
    
    print('Helicity Vortex Identification Completed')
    return H_n

def filmax(data):
    for i in range(0,np.size(data)):
        if data[i] == np.max(data):
            data = data[:i]
            return data
        
def plumeheight(data,threshold,dt,dz,D):
    nx,ny,nz,nt = np.shape(data)
    data = data/np.max(data)
    data[data<threshold] = 0
    data = np.sum(data,0)
    data = np.sum(data,0)
#     print(np.shape(data))

    plheight = np.zeros(nt)
    for i in range(nt):
        for j in range (nz):
            if data[j,i] > 1:
                plheight[i]=j*dz/D;
    return plheight

def frontvelocity(data,dt):
    plheight = filmax(data)
    plheight = np.gradient(plheight)/dt
    return plheight

def frontvelocityplot(data,dataname,dt=10,marksize=250):
    plt.figure(figsize=(10,5))
    color = 'k'
    nt = np.shape(data)[0]
    x = list(range(0,(nt)*dt,dt))
    plt.plot(x[0:np.shape(data)[0]],data,color = color)
    plt.locator_params(axis='x', nbins=300)
    plt.grid()
    plt.xticks(np.arange(min(x), max(x)+1, marksize))
    plt.ylabel('Wf/D');plt.xlabel('Time')
    plt.savefig(dataname +' Front Velocity.png')
    
def plumeradius_timeaverage(data,threshold,dx,D):
    nx,ny,nz,nt = np.shape(data)
    data = data/np.max(data)
    data[data<threshold] = 0
    data = np.mean(data,3)
    data = np.sum(data,0)
    a = data
    sizea = int((nx+1)*0.5)
    b = a[0:sizea][::-1]
    c = a[sizea-1:nx]
    d = (b+c)*0.5
    radius = np.zeros(nz)
    for z in range(nz):
        for r in range(sizea):
            if(d[r,z]>0):
                radius[z] = (r+1)*dx/D               
    radius =  radius
    return radius

# def plumeradius(data,threshold,dx,D):
#     nx,ny,nz,nt = np.shape(data)
#     data = data/np.max(data)
#     data = np.mean(data,2)
#     data = np.mean(data,0)
#     a = data
#     sizea = int((nx+1)*0.5)
#     b = a[0:sizea][::-1]
#     c = a[sizea-1:nx]
#     d = (b+c)*0.5
#     radius = np.zeros(nt)
#     for t in range(nt):
#         for r in range(sizea):
#             if(d[r,t]>=threshold):
#                 radius[t] = (r+1)*dx/D               
#     radius =  radius
#     return radius

def gprimeT(data,Ta,g):
    data = data*g/Ta
    return data

def gprimeRho(Q,density_ref,density_ambient,g):
    Q = density_ref*(1-Q)
    Q = ((Q - density_ambient)/density_ambient)*g
    return Q


def wcentermax(data,dz,D):
    nx,ny,nz,nt = np.shape(data)
    data = np.mean(data,3)
    c = int((nx-1)/2)
    maxc = max(data[c,c,:])
    for i in range(nz):
        if (maxc==data[c,c,i]):
            cl = i+1
    cl = cl
    return maxc,cl

def reynoldswc(plumeradius_timeaverage,W,threshold,dx,dz,D,nu):
    wc, cl = wcentermax(W,dz,D)
    r = plumeradius_timeaverage
    l = r[cl]
    re = wc*l*D/nu
    return re

def gT0(data):
    print(np.shape(data))
    nx,ny,nz,nt = np.shape(data)
    data = np.mean(data,3)
    x = int((nx-1)/2)
    gT =data[x,x,1]
    return gT

def wb(data,D):
    nx,ny,nz,nt = np.shape(data)
    data = np.mean(data,3)
    x = int((nx-1)/2)
    wb = np.sqrt(data[x,x,1]*D)
#     print(wb)
    return wb

def froudewc(gT,W,dz,D):
    wc, cl = wcentermax(W,dz,D)
    wbouyance = wb(gT,D)
    fr = wc/wbouyance
#     print("Froude(wc): ",fr)
    return fr

def namelist2dict(path):
    import pandas as pd
    df = pd.read_csv(path, delimiter = "\t")
    namelist = np.squeeze(np.array(df))
    namelist_shape = np.shape(namelist)
    namelist_name = [];namelist_value = []
    for i in range (int(namelist_shape[0])):
        length = len(namelist[i])
        temp = namelist[i]
        for j in range(length):
            if temp[j]=='=':
                a=j+1
            if temp[j]==',':
                b=j
                if temp[j-1]=='.':
                    b = j-1
                break
        name = str(namelist[i][:a-1])
        name = name.replace(' ','')
        namelist_name.append(name)
        try:
            namelist_value.append(float(namelist[i][a:b]))
        except:
            string = str(namelist[i][a:b])
            string = string.replace(' ','')
            string = string.replace('.','')
            namelist_value.append(string)
    dic = dict(zip(namelist_name,namelist_value))
    return dic

def hf_csv2array (data):
    temp = np.array(data)
    nt = np.shape(temp)[0]
    
    a_string = temp[1,0]
    a_list = a_string.split()
    map_object = map(float, a_list)
    list_of_integers = list(map_object)
    testnz = np.array(list_of_integers)
    
    nz = int(np.shape(testnz)[0])
    newarray = np.zeros((nt,nz))
    
    for i in range (nt):
        a_string = temp[i,0]
        a_list = a_string.split()
        map_object = map(float, a_list)
        list_of_integers = list(map_object)
        newarray[i,:] = np.array(list_of_integers)
    return newarray

def hf_loadcsv(filename):
    temp = pd.read_csv(filename,skiprows=1,header=None)
    temp= hf_csv2array(temp)
    temp = temp.T
    return temp

def hf_fulldata(folder,dataname,nx):
    temp= hf_loadcsv(folder+'\\c0001.d01.'+dataname)
    nz,nt = np.shape(temp)
    hfdata = np.zeros((nx,nz,nt))
    hfdata[int((nx-1)/2),:,:] = temp
    for i in range(1,int((nx+1)/2)):
        hfdata[int((nx-1)/2)+i,:,:] = hf_loadcsv(folder+'\\i00'+str(i).zfill(2)+".d01."+dataname)
    for i in range(1,int((nx+1)/2)):
        hfdata[int((nx-1)/2)-i,:,:] = hf_loadcsv(folder+'\\j00'+str(i).zfill(2)+".d01."+dataname)
    return hfdata

def hf_csv2mat(folder,datatype,nx):
    data = hf_fulldata(folder,datatype,nx)
    filename = folder+"\\"+datatype
    savemat(data[0:int(nx/2+1)],filename+'_1')
    savemat(data[int(nx/2+1):nx],filename+'_2')
    
def hf_loadmat(file):
    d1 = loadmat(file+"_1.mat")
    d2 = loadmat(file+"_2.mat")
    data = np.concatenate((d1,d2),axis=0)
    return data

def fluctuation(data):
    nx,nz,nt = np.shape(data)
    dmean = np.mean(data,2)
    dprime = np.zeros(np.shape(data))
    for t in range(nt):
        dprime[:,:,t] = data[:,:,t]-dmean
    return dprime

def hf_contour(data,interface,dx,dz,D,threshold,title):
    import numpy as np
    import matplotlib.pyplot as plt
    data = data.T
    interface = interface.T
    nz,nx = np.shape(data)
    plt.style.use('seaborn-white')
    xi = np.linspace(-nx*.5*dx/D, nx*.5*dx/D, nx)
    zi = np.linspace(0, nz*dz/D, nz)
    plt.figure(figsize=(10, 10))
    contours  = plt.contour(xi,zi,interface, colors='black');
#     plt.clabel(contours, inline=True, fontsize=15)
    plt.ylabel("$z/D$",fontsize=14)
    plt.xlabel("$x/D$", rotation=0, fontsize=14, labelpad=10)
#     plt.title(title,fontsize=18,pad = 20)
    plt.imshow(data, extent=[-nx*.5*dx/D, nx*.5*dx/D,0, nz*dz/D], origin='lower',cmap='jet',alpha=1,aspect='auto')
    plt.colorbar()
    plt.savefig(title+".png")

def plot_plume_spacial_data(data,x1,x2,n,title,axis,dx,dy,dz,D):
    
    data = np.mean(data,3) #Time average the data [45,45,700,541] to [45,45,700]
    color = 'k'
    nx,ny,nz = np.shape(data)
    def d2x(x1,nx,D):
        if (x1<0):
            x1=x1*-1
            x1 = int(nx/2)-int(x1*D/dx)
        if (x1==0):
            x1 = int((nx-1)/2)
        else:
            x1 = int(x1*2*D/dx)-1
        return x1
    
    if n == 0:
        plt.figure(figsize=(10,5))
        xaxis = "x/D"
        yaxis = axis
        x1 = d2x(x1,ny,D)
        if x2 != 0:
            x2 = int(x2*D/dz)-1
        try:
            data = data[:,x1,x2] # Take a slide of data at y=x1 and z=x2
        except:
            print("Out of range")
            print("x/D range: ",float(-(nx)*dx/2/D)," to ",float((nx)*dx/2/D))
            print("z/D range: ",0," to ",float((nz)*dz/D))
        x = np.linspace(float(-(nx)*dx/2/D),float((nx)*dx/2/D),int(nx))
        plt.plot(x[0:np.shape(data)[0]],data,linewidth=1,color=color,
        marker = 'o',ms =5  ,mfc = color,
        label=title,markevery = 1)
        
    if n == 1:
        plt.figure(figsize=(10,5))
        xaxis = "y/D"
        yaxis = axis
        x1 = d2x(x1,nx,D)
        if x2 != 0:
            x2 = int(x2*D/dz)-1
        try:
            data = data[x1,:,x2] # Take a slide of data at x=x1 and z=x2
        except:
            print("Out of range")
            print("y/D range: ",float(-(ny)*dy/2/D)," to ",float((ny)*dy/2/D))
            print("z/D range: ",0," to ",float((nz)*dz/D))
        x = np.linspace(float(-(ny)*dy/2/D),float((ny)*dy/2/D),int(ny))
        plt.plot(x[0:np.shape(data)[0]],data,linewidth=1,color=color,
        marker = 'o',ms =5  ,mfc = color,
        label=title,markevery = 1)
    
    if n == 2:
        yaxis = "z/D"
        xaxis = axis
        plt.figure(figsize=(5,10))
        x1 = d2x(x1,nx,D)
        x2 = d2x(x2,ny,D)
        print(x1,x2)
        data = data[x1,x2,5:] #Take a slide of data at x=x1, y=x2 and z is range from 5 to 700 
        x = np.linspace(0,float((nz)*dz/D),int(nz))
        plt.plot(data,x[0:np.shape(data)[0]],linewidth=1,color=color,
        marker = 'o',ms =1.5,mfc = color,
        label=title,markevery = 1)
    try:
        plt.ylabel("$"+yaxis+"$",fontsize=14)
        plt.xlabel("$"+xaxis+"$", rotation=0, fontsize=14, labelpad=20)
#         plt.title(title,fontsize=18,pad = 20)
    except:
        print("n=0 : xz plane ")
        print("n=1 : yz plane ")
        print("n=2 : xy plane ")
    plt.savefig(title+".png")

def hf_plume_interface(data,threshold):
    data = data/np.max(data)
    data = np.mean(data,2)
    data[data<threshold] = 0
    data[data>threshold] = 1
    return data

        
class plume_metadata:
    def __init__(self,namelist_path):
        namelist_dictionary = namelist2dict(namelist_path)
        self.nx_original = int(namelist_dictionary["nx_original"])
        self.ny_original = int(namelist_dictionary["ny_original"])
        self.nz_original = int(namelist_dictionary["nz_original"])
        self.nt_original = int(namelist_dictionary["nt_original"])
        self.nx_trimmed = int(namelist_dictionary["nx_trimmed"])
        self.ny_trimmed = int(namelist_dictionary["ny_trimmed"])
        self.nz_trimmed = int(namelist_dictionary["nz_trimmed"])
        self.nt_trimmed = int(namelist_dictionary["nt_trimmed"])
        self.dx = namelist_dictionary["dx"]
        self.dy = namelist_dictionary["dy"]
        self.dz = namelist_dictionary["dz"]
        self.dt = namelist_dictionary["dt"]
        self.Ho =  namelist_dictionary["tke_heat_flux_hot"]
        self.gas_constant = namelist_dictionary["plume_gas_constant"]
        self.plume_gas = namelist_dictionary["plume_gas"]
        self.D = namelist_dictionary["source_dia"]
        self.g = namelist_dictionary["gravity"]
        self.threshold = namelist_dictionary["plume_threshold"]
        self.Ta = namelist_dictionary["ambient_temperature"]
        self.nu = namelist_dictionary["dynamic_viscosity"]
        self.density_ref = namelist_dictionary["density_reference"]
        self.density_amb = namelist_dictionary["density_ambient"]

class plume_metrics:
    def __init__(self,T,Q,dataW,dx,dy,dz,dt,D,threshold,Ta,g,density_ref,density_amb):
        self.gT = gprimeT(T,Ta,g)
        self.gRho = gprimeRho(Q,density_ref,density_amb,g)
        W = dataW
        self.frontheight = plumeheight(self.gT,threshold,dt,dz,D);
        self.frontvelocity = frontvelocity(self.frontheight,dt);
        self.width = plumeradius(self.gT,threshold,dx,D);
    
class plume_centerline_metrics:
    def __init__(self,gprimeT,dataW,dx,dy,dz,dt,D,threshold,nu): #v=0.00004765
        gT = gprimeT
        W = dataW
        nx,ny,nz,nt = np.shape(gT)
        self.gprimeT0 = gT0(gT)
        self.wcmax,self.wcmax_location = wcentermax(W ,dz,D)
        self.z_wmax = self.wcmax_location*dz/D
        self.radius_profile = plumeradius_timeaverage(gT,threshold,dx,D)
        self.lwcm = self.rtimeaverage[self.wcmax_location]
        self.Reynolds_max_centerline = reynoldswc(self.rtimeaverage,W,threshold,dx,dz,D,nu)
        self.Froude_max_centerline = froudewc(gT,W,dz,D)
        
class high_frequency_profile:
    def __init__(self,T,U,V,W,dx,dz,D,Ta,g,threshold):
        nx,nz,nt = np.shape(T)
        T = T - Ta
        T = gprimeT(T,Ta,g)
        self.interface = hf_plume_interface(T,threshold)
        shear = (np.gradient(np.mean(W,2),dx,dz))[0]
        dTdz = (np.gradient(np.mean(T,2),dx,dz))[0]
        U = fluctuation(U)
        V = fluctuation(V)
        W = fluctuation(W)
        T = fluctuation(T)
        nx,nz,nt = np.shape(T)
        self.Re_stress_UW = np.mean(U*W,2)
        self.Re_stress_VW = np.mean(V*W,2)
        self.Re_stress_UU = np.mean(U*U,2)
        self.Re_stress_VV = np.mean(V*V,2)
        self.Re_stress_WW = np.mean(W*W,2)
        self.TKE_shear_pro_UW = -(self.Re_stress_UW*shear)
        self.TKE_shear_pro_VW = -(self.Re_stress_VW*shear)
        self.TKE_buoyant_production_I = np.mean(W*T,2)
        self.TKE_buoyant_production_II = self.TKE_buoyant_production_I*dTdz
        self.TKE = self.Re_stress_UU+self.Re_stress_VV+self.Re_stress_WW
        centerline = int((nx-1)/2)
        self.U_rms_centerline = self.Re_stress_UU[centerline,:]
        self.V_rms_centerline = self.Re_stress_VV[centerline,:]
        self.W_rms_centerline = self.Re_stress_WW[centerline,:]
        self.TKE_centerline = self.TKE[centerline,:]
        self.TKE_buoyant_production_I_centerline = self.TKE_buoyant_production_I[centerline,:]
        self.TKE_buoyant_production_II_centerline = self.TKE_buoyant_production_II[centerline,:]
        
        # Plot
        mark = 20
        plt.figure(figsize=(10, 7))
        zi = np.linspace(0, nz*dz/D, nz)
        plotdata = U_rms_centerline/U_rms_centerline.max()
        plt.plot(zi,plotdata,'go--', label='U_rms_centerline', markevery = mark, linewidth=3)
        plotdata = V_rms_centerline/V_rms_centerline.max()
        plt.plot(zi,plotdata,'bs--', label='V_rms_centerline',markevery = mark, linewidth=3)
        plotdata = W_rms_centerline/W_rms_centerline.max()
        plt.plot(zi,plotdata,'rv--', label='W_rms_centerline',markevery = mark, linewidth=3)
        plotdata = TKE_centerline/TKE_centerline.max()
        plt.plot(zi,plotdata,'c8--', label='TKE_centerline',markevery = mark, linewidth=3)
        plotdata = TKE_buoyant_production_I_centerline/TKE_buoyant_production_I_centerline.max()
        plt.plot(zi,plotdata,'yx--', label='TKE_buoyant_production_I_centerline',markevery = mark, linewidth=3)
        plotdata = TKE_buoyant_production_II_centerline/TKE_buoyant_production_II_centerline.max()
        plt.plot(zi,plotdata,'kp--', label='TKE_buoyant_production_II_centerline', markevery = mark,linewidth=3)
        plt.legend()
        plt.xlabel("$z/D$")
        plt.savefig("HighFrequency_Centerline_Profile.png")
        

class plume_polar_coordinates:
    def __init__(self,dataU,dataV,dataW,dx,dy,dz):
        self.Ur, self.Utheta, self.W = cart2pol(dataU,dataV,dataW,dx,dy,dz)
