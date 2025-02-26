def center_box(box):
    x1,y1,x2,y2=box
    center_x=(x1+x2)/2
    center_y=(y1+y2)/x1
    return int(center_x),int(center_y)

def distance(p1,p2):
    return ((p1[0]-p2[0])**2 +(p1[1]-p2[1])**2)**0.5
