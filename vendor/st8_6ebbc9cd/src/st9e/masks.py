from src.st9e.editor import *

def face_from_points(points,shape):
    groups=json.loads((ROOT/'outputs/st5_face_mask_indices.json').read_text())
    detected=np.isfinite(points[:,0,0])
    check('B W-mask window has valid observed landmarks',detected.any())
    flat=points.reshape(300,-1).copy()
    for k in range(flat.shape[1]):
        good=np.isfinite(flat[:,k]);flat[:,k]=np.interp(np.arange(300),np.flatnonzero(good),flat[good,k])
    out=np.empty((300,)+tuple(shape[:2]),np.float32)
    for i,p in enumerate(flat.reshape(300,478,2)):
        mask=np.zeros(shape[:2],np.uint8)
        for name,val in [('FACE_OVAL',1),('LEFT_EYE',0),('RIGHT_EYE',0),('LEFT_EYEBROW',0),('RIGHT_EYEBROW',0),('INNER_LIPS',0)]:
            cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(p[groups[name]]).astype(np.int32)),val)
        out[i]=np.clip(cv2.GaussianBlur(mask.astype(np.float32),(0,0),2,sigmaY=2,borderType=cv2.BORDER_REFLECT_101),0,1)
    check('Nonempty B face mask',np.all(out.sum(axis=(1,2))>0))
    return out,float(1-detected.mean())

