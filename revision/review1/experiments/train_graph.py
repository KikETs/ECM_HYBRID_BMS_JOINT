import numpy as np
import torch

import sop_trim as base

def train_graph(model_cls, tr, te, epochs, lr, seed, dev, lam=1e-2, delta=.02, ablate=None, q=0.):
    if dev!="cuda" or q!=0.:
        return base.train_fold(model_cls,tr,te,epochs,lr,seed,dev,lam,delta,ablate,q)
    base.determinism.enable(seed)
    Xtr, Xte = tr["X"].copy(), te["X"].copy()
    if ablate:
        Xtr[:,ablate] = 0.
        Xte[:,ablate] = 0.
    features = Xtr.reshape(-1, Xtr.shape[-1])
    mu, sd = features.mean(0), features.std(0)+1e-8
    def dev_array(x):
        return torch.from_numpy(x.astype(np.float32, copy=False)).to(dev)
    x = dev_array(((Xtr-mu)/sd).clip(-4,4))
    xte = dev_array(((Xte-mu)/sd).clip(-4,4))
    y,n,i = (dev_array(tr[k]) for k in ("Y","NOM","I"))
    m = model_cls(x.shape[-1]).to(dev)
    # Fixed-shape CUDA Graph replays the same Adam/Huber updates. The
    # permutations are generated with the original seeded randperm.
    opt = torch.optim.Adam(m.parameters(), lr=torch.tensor(lr,device=dev,dtype=torch.float64),
                           weight_decay=1e-4, capturable=True)
    loss_function = torch.nn.HuberLoss(delta=delta)
    perm = torch.arange(len(x),device=dev)
    batch = 256
    def step(b):
        opt.zero_grad(set_to_none=True)
        kf,ks = base.decode(m(x[b]))
        p = base.dv_hat(kf,ks,n[b],i[b])
        loss = loss_function(p,y[b])+lam*((torch.log(kf)**2).mean()+(torch.log(ks)**2).mean())
        loss.backward()
        opt.step()
    params=list(m.parameters())
    before=[p.detach().clone() for p in params]
    warm=torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        step(perm[:batch])
        step(perm[-(len(x)%batch or batch):])
    torch.cuda.current_stream().wait_stream(warm)
    with torch.no_grad():
        for p,b in zip(params,before):
            p.copy_(b)
        for p in params:
            for value in opt.state[p].values():
                value.zero_()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for start in range(0,len(x),batch):
            step(perm[start:start+batch])
    for ep in range(epochs):
        if ep in (int(epochs*.5),int(epochs*.75)):
            opt.param_groups[0]["lr"].mul_(.1)
        perm.copy_(torch.randperm(len(x),device=dev))
        graph.replay()
    m.eval()
    state={"model":m.state_dict(),"mu":mu,"sd":sd,"cls":model_cls.__name__,"n_in":x.shape[-1],
           "ablate":ablate,"KF_SPAN":base.KF_SPAN,"KS_SPAN":base.KS_SPAN}
    with torch.no_grad():
        kf,ks=base.decode(m(xte))
        pred = base.dv_hat(kf,ks,dev_array(te["NOM"]),dev_array(te["I"])).cpu().numpy()
        kf,ks=kf.cpu().numpy(),ks.cpu().numpy()
    return pred,kf,ks,sum(p.numel() for p in params),state
