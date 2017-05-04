import h5py
import numpy as np
import time, os, collections, itertools
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# pngquant *.png
# ffmpeg -pix_fmt yuv422p -i %06d-fs8.png -c:v libx264 -f mp4 output.mp4
# ffmpeg -i %06d-fs8.png output.gif
frames = 40

FLAG_SAVEFIG = True

def slerp(p0, p1, t):
    p0 /= np.linalg.norm(p0)
    p1 /= np.linalg.norm(p1)
    
    omega = np.arccos(np.dot(p0, p1))
    s0 = np.sin(omega)
    q0 = (np.sin((1.0-t)*omega)/s0).reshape(-1,1)
    q1 = (np.sin(t*omega)/s0).reshape(-1,1)

    return p0.reshape(1,-1)*q0 + p1.reshape(1,-1)*q1


def plot_sphere(
        xyz,
        show_charges=True,
        show_connections=True,
        show_labels=False,
        show_sphere=False,
        hull_points=20,
        charge_size=100,
        radius=1.0,
        ax=None,
):

    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')

    xyz /= radius
    N = xyz.shape[0]
    xyz = xyz[np.argsort(xyz[:,2])]

    if show_charges==True:
        X,Y,Z = xyz.T
        C = np.array([np.linspace(.2,.8,N),np.linspace(0,0.0,N),np.linspace(.5,.2,N)]).T
        
        ax.scatter(X,Y,Z,s=charge_size,lw=3,c=C)

    if show_labels==True:
        for i,pt in enumerate(xyz):
            x,y,z=pt
            ax.text(x, y, z, str(i))

    if show_connections==True:
        _is_connected = set()
        hull = ConvexHull(xyz)

        def draw_connection(i,j):
            if (i,j) in _is_connected:
                return None
            T = np.linspace(0,1,hull_points)
            v0, v1 = xyz[i], xyz[j]
            
            X,Y,Z = (slerp(v0,v1,T)/radius).T
            plt.plot(X,Y,Z,'k',alpha=0.25)

            # Used for vertex drawing
            X = np.linspace(v0[0], v1[0], k)
            Y = np.linspace(v0[1], v1[1], k)
            Z = np.linspace(v0[2], v1[2], k)
            #plt.plot(X,Y,Z,'r',alpha=0.5)
            
            _is_connected.add((i,j))

        for i,j,k in hull.simplices:
            draw_connection(i,j)
            draw_connection(j,k)
            draw_connection(i,k)

    if show_sphere==True:
        r = 0.90
        phi, theta = np.mgrid[0.0:np.pi:20j, 0.0:2.0*np.pi:20j]
        x = r*np.sin(phi)*np.cos(theta)
        y = r*np.sin(phi)*np.sin(theta)
        z = r*np.cos(phi)
        
        ax.plot_surface(x, y, z,
                        rstride=1, cstride=1, color='c',
                        alpha=0.6, linewidth=0)
        

    ax.set_aspect('equal')
    ax.set_axis_off()
    plt.tight_layout()
    return ax

os.system('mkdir -p movies')

if __name__ == "__main__":
    from adjust_coords import coordinates, align, orient
    C = coordinates('configurations.h5')

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the first figure
    
    xyz = orient(C[100])
    plot_sphere(xyz,charge_size=300,
                #show_sphere=True,
                show_connections=False,
                radius=1.,ax=ax)

    
    #X = np.linspace(0,np.pi/2,frames)
    #D = np.sin(X+np.pi/2)*5 + 5
    
    X = np.linspace(0,np.pi,frames)[::-1]
    D = np.sin(X+2*np.pi)*5+5
    A = np.linspace(0,360/2,frames)
    
    ax.dist = D[0]
    ax.azim = A[0]
    ax.elev = 0
    plt.show(block=False)
    
    name = 'first'
    os.system('mkdir -p movies/{}'.format(name))
    fignum = itertools.count()
        
    for dist,a in zip(D,A):
        ax.dist=dist
        ax.azim=a
        ax.elev=a*0.75
        plt.draw()
        plt.pause(.00005)
        if FLAG_SAVEFIG:
            plt.savefig('movies/{}/{:06}.png'.format(name,fignum.next()))

    A = np.linspace(360/2.0,360,frames)
    for dist,a in zip(D,A):
        ax.dist=dist
        ax.azim=a
        ax.elev=a*0.75
        plt.draw()
        plt.pause(.00005)
        if FLAG_SAVEFIG:
            plt.savefig('movies/{}/{:06}.png'.format(name,fignum.next()))
    plt.show()

    
    
