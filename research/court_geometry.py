import numpy as np
R = 22 + 1.75 / 12
JOIN = np.sqrt(R * R - 22 * 22)

def offset(x, y):
    x, y = np.asarray(x), np.asarray(y)
    theta = np.clip(np.arctan2(y, x), -np.arcsin(22 / R), np.arcsin(22 / R))
    arc = np.hypot(x - R * np.cos(theta), y - R * np.sin(theta))
    segment = np.hypot(x - np.clip(x, -5.25, JOIN), np.abs(y) - 22)
    inside = np.where(x <= JOIN, np.abs(y) < 22, np.hypot(x, y) < R)
    return np.minimum(arc, segment) * np.where(inside, -1, 1)

def court(ax):
    y = np.linspace(-22, 22, 300)
    ax.plot(y, np.sqrt(R * R - y * y), color='#e55b22', lw=1.5)
    for a in [-22,22]:
        ax.plot([a,a],[-5.25,JOIN],color='#e55b22',lw=1.5)
    ax.scatter([0],[0],s=45,facecolors='none',edgecolors='#e55b22')
    ax.set(xlim=(-25,25),ylim=(-5.25,41.75),xlabel='Lateral position (feet)',ylabel='Distance toward half court (feet)')
    ax.set_aspect('equal')


def zones(x,y):
    x,y=np.asarray(x),np.asarray(y)
    d=np.hypot(x,y);a=offset(x,y)
    return np.select([x>41.75,np.abs(a)<=1,(a>1)&(x<=JOIN),a>1,d<4,d<8,d<16],['Beyond half court','Near the line','Outside corner','Outside arc','Near rim','Short interior','Middle interior'],default='Long interior')
