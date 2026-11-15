"""Exact official model constructors and test forward operations; no metric imports."""
import torch


def create(name,cfg):
    device=torch.device('cuda:0');wrapped=name in ['DeepPhys','Tscan','EfficientPhys','PhysFormer','RhythmFormer','FactorizePhys']
    if name in ['PhysNet','Physnet']:
        from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
        model=PhysNet_padding_Encoder_Decoder_MAX(frames=cfg.MODEL.PHYSNET.FRAME_NUM)
    elif name=='RhythmFormer':
        from neural_methods.model.RhythmFormer import RhythmFormer
        model=RhythmFormer()
    elif name=='DeepPhys':
        from neural_methods.model.DeepPhys import DeepPhys
        model=DeepPhys(img_size=cfg.TEST.DATA.PREPROCESS.RESIZE.H)
    elif name=='Tscan':
        from neural_methods.model.TS_CAN import TSCAN
        model=TSCAN(frame_depth=cfg.MODEL.TSCAN.FRAME_DEPTH,img_size=cfg.TEST.DATA.PREPROCESS.RESIZE.H)
    elif name=='EfficientPhys':
        from neural_methods.model.EfficientPhys import EfficientPhys
        model=EfficientPhys(frame_depth=cfg.MODEL.EFFICIENTPHYS.FRAME_DEPTH,img_size=cfg.TEST.DATA.PREPROCESS.RESIZE.H)
    elif name=='iBVPNet':
        from neural_methods.model.iBVPNet import iBVPNet
        model=iBVPNet(frames=cfg.MODEL.iBVPNet.FRAME_NUM,in_channels=cfg.MODEL.iBVPNet.CHANNELS)
    elif name=='PhysFormer':
        from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
        p=cfg.MODEL.PHYSFORMER
        model=ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(cfg.TEST.DATA.PREPROCESS.CHUNK_LENGTH,cfg.TRAIN.DATA.PREPROCESS.RESIZE.H,cfg.TRAIN.DATA.PREPROCESS.RESIZE.W),
            patches=(p.PATCH_SIZE,)*3,dim=p.DIM,ff_dim=p.FF_DIM,num_heads=p.NUM_HEADS,num_layers=p.NUM_LAYERS,dropout_rate=cfg.MODEL.DROP_RATE,theta=p.THETA)
    elif name=='FactorizePhys':
        p=cfg.MODEL.FactorizePhys;md={k:p[k] for k in ['FRAME_NUM','MD_TYPE','MD_FSAM','MD_TRANSFORM','MD_S','MD_R','MD_STEPS','MD_INFERENCE','MD_RESIDUAL']}
        if p.TYPE.lower()=='standard':
            from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys as Constructor
        elif p.TYPE.lower()=='big':
            from neural_methods.model.FactorizePhys.FactorizePhysBig import FactorizePhysBig as Constructor
        else:raise ValueError('Unregistered FactorizePhys architecture')
        model=Constructor(frames=p.FRAME_NUM,md_config=md,in_channels=p.CHANNELS,dropout=cfg.MODEL.DROP_RATE,device=device)
    else:raise ValueError('Unregistered sealed model')
    model=model.to(device)
    if wrapped:model=torch.nn.DataParallel(model,device_ids=[0])
    return model


def predict(model,name,data,cfg):
    batch=len(data);length=int(cfg.TEST.DATA.PREPROCESS.CHUNK_LENGTH)
    if name in ['DeepPhys','Tscan','EfficientPhys']:
        n,d,c,h,w=data.shape;data=data.reshape(n*d,c,h,w)
        if name in ['Tscan','EfficientPhys']:
            depth=cfg.MODEL.TSCAN.FRAME_DEPTH if name=='Tscan' else cfg.MODEL.EFFICIENTPHYS.FRAME_DEPTH
            data=data[:len(data)//depth*depth]
        if name=='EfficientPhys':data=torch.cat((data,data[-1:].clone()),dim=0)
        prediction=model(data).reshape(-1)
        return torch.stack([prediction[i*length:(i+1)*length] for i in range(batch)])
    if name in ['FactorizePhys','iBVPNet']:
        data=torch.cat((data,data[:,:,-1:].clone()),dim=2);value=model(data)
        prediction=value[0] if isinstance(value,tuple) else value
        if name=='FactorizePhys':prediction=(prediction-prediction.mean())/prediction.std()
    elif name=='PhysFormer':prediction=model(data,2.)[0]
    elif name in ['PhysNet','Physnet']:prediction=model(data)[0]
    elif name=='RhythmFormer':
        prediction=model(data);prediction=(prediction-prediction.mean(dim=-1,keepdim=True))/prediction.std(dim=-1,keepdim=True)
    else:raise ValueError('Unregistered sealed forward operation')
    return prediction.reshape(batch,length)
