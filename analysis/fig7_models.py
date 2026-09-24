"""Checkpoint reconstruction and explicit FP32 inference copies for Fig.7 only."""
import types
import numpy as np
import torch
from analysis import fig5_representation_alignment as f5
from src.cl_tsmnet import training as t
from src.cl_tsmnet.spd_pca import migrate_legacy_spddsbn_buffers


def value(record,key,default):
    v=record.get(key)
    if v is None or (isinstance(v,float) and np.isnan(v)) or v=='': return default
    return type(default)(v) if default is not None else v


def build(record,ds,split,checkpoint,device):
    kind=record['model_type']; c,s=ds['x'].shape[1:]; classes=len(np.unique(ds['y'][split['source_ids']]))
    def options(prefix,defaults): return {k:value(record,prefix+k,v) for k,v in defaults.items()}
    if kind in ['tsmnet','ms_tgc_spddsbn']:
        if kind=='ms_tgc_spddsbn' and record.get('mstgc_architecture')!='shared_channel_graph_augmented_spd_v3':
            raise ValueError('Legacy AGMNet architecture: reconstruct with its original code or retrain; no silent v3 substitution')
        model=f5._build_model(ds,split['domains'],np.r_[split['source_ids'],split['val_ids'],split['target_ids']],
                 split['source_ids'],dict(model_type=kind,bnorm='spddsbn'),f5._model_config(record),device)
    elif kind=='eegnet':
        model=t.build_eegnet(c,s,classes,**options('eegnet_',dict(temporal_filters=64,spatial_filters=4,dropout=.5,avgpool_factor=2))).to(device)
    elif kind=='eegconformer':
        model=t.build_eegconformer(c,s,classes,temporal_kernel=value(record,'temp_kernel',25),
            **options('conformer_',dict(emb_size=40,depth=6,num_heads=5,dropout=.5,classifier_hidden=256))).to(device)
    elif kind=='bfgcn':
        model=t.build_bfgcn(c,classes,**options('bfgcn_',dict(kadj=2,num_out=16,att_hidden=16,classifier_hidden=32,avgpool=2,dropout=0.))).to(device)
    elif kind=='mdtn':
        model=t.build_mdtn_gmda(c,classes,**options('mdtn_',dict(hidden_dim=64,num_nodes=0,kernel_length=16,num_heads=4,cheby_order=3,dropout=.5)),
                              max_iter=max(1,value(record,'epochs',30)*1000)).to(device)
    else: raise ValueError(kind)
    state,migrations=migrate_legacy_spddsbn_buffers(f5._load_state(str(checkpoint)),model.state_dict())
    model.load_state_dict(state,strict=True)
    return model.eval(),migrations


def _tsm_fp32(self,x,d):
    # Same inference math as vendored TSMNet.forward, with an explicit FP32 port.
    h=self.cnn(x[:,None,...])
    l=self.spdnet(self.cov_pooling(h))
    if hasattr(self,'spdbnorm'): l=self.spdbnorm(l)
    if hasattr(self,'spddsbnorm'): l=self.spddsbnorm(l,d)
    l=self.logeig(l)
    if hasattr(self,'tsbnorm'): l=self.tsbnorm(l)
    if hasattr(self,'tsdsbnorm'): l=self.tsdsbnorm(l,d)
    return self.classifier(l)


def _moments_fp32(self,maps):
    # GraphSPDManifoldHead._moments, without its hard-coded CPU/FP64 cast.
    b,c,f,s=maps.shape
    observations=maps.permute(0,2,1,3).reshape(b,f,c*s)
    mean=observations.mean(-1,keepdim=True)
    centered=observations-mean
    covariance=torch.bmm(centered,centered.transpose(1,2))/float(c*s-1)
    eye=torch.eye(f,device=maps.device,dtype=maps.dtype)[None]
    scale=covariance.diagonal(dim1=-2,dim2=-1).sum(-1).view(b,1,1)/float(f)
    covariance=(1-self.shrinkage)*covariance+self.shrinkage*scale*eye+self.covariance_epsilon*eye
    return mean,covariance


def convert_fp32(model,kind,device):
    # Applies to a separately loaded evaluation instance, never saved to checkpoint.
    manifold_buffers=[(module,name,tensor.manifold) for module in model.modules()
                      for name,tensor in module._buffers.items() if hasattr(tensor,'manifold')]
    plain_tensors=[(module,name,tensor) for module in model.modules()
                   for name,tensor in vars(module).items() if isinstance(tensor,torch.Tensor)]
    torch.nn.Module.to(model,device=device,dtype=torch.float32)
    if manifold_buffers or any(hasattr(t,'manifold') for _,_,t in plain_tensors):
        from geoopt import ManifoldTensor
        for module,name,manifold in manifold_buffers:
            module._buffers[name]=ManifoldTensor(module._buffers[name],manifold=manifold)
    for module,name,tensor in plain_tensors:
        converted=tensor.to(device=device,dtype=torch.float32 if tensor.is_floating_point() else tensor.dtype)
        if hasattr(tensor,'manifold'): converted=ManifoldTensor(converted,manifold=tensor.manifold)
        setattr(module,name,converted)
    for module in model.modules():
        for attr in ['device_','spd_device_','spd_device','graph_device']:
            if hasattr(module,attr): setattr(module,attr,device)
    if kind=='tsmnet': model.forward=types.MethodType(_tsm_fp32,model)
    if kind=='ms_tgc_spddsbn': model.spd_branch._moments=types.MethodType(_moments_fp32,model.spd_branch)
    bad=[n for n,p in list(model.named_parameters())+list(model.named_buffers()) if p.is_floating_point() and (p.dtype!=torch.float32 or p.device!=device)]
    if bad: raise ValueError('Non-FP32/device state after port: '+str(bad))
    return model.eval()


def prepare_input(kind,windows,domains,fs,device):
    if kind=='bfgcn':
        # BF-GCN forward accepts precomputed bandpower and PLV, NOT raw EEG.
        return (torch.from_numpy(t._bfgcn_bandpower_features(windows,fs)).to(device),
                torch.from_numpy(t._bfgcn_plv(windows,fs)).to(device))
    return torch.from_numpy(windows).to(device),torch.as_tensor(domains,dtype=torch.long,device=device)


def forward(model,kind,inputs):
    if kind=='bfgcn': return model(*inputs,alpha=0.)[0]
    return t._forward_logits(model,*inputs)
