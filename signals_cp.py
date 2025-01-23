import cupy as cp

def dens(pos, Q):
    """Computes projected density"""
    #print(pos.shape)
    #print(Q.shape)
    return (cp.exp(1.0j*pos.dot(Q.T))).sum(axis = 1)

def _j(pos, vel, Q, Qn):
    """Computes full projected current"""
    return cp.sum(vel[:, :, cp.newaxis] * (cp.exp(1.0j*pos.dot(Q.T))[..., cp.newaxis]), axis = 1)

def _j_L(pos, vel, Q, Qn):
    """Computes longtitugonal projected current"""
    return (cp.sum(vel.dot(Qn.T)*cp.exp(1.0j*pos.dot(Q.T)), axis = 1)[..., cp.newaxis])*Qn

def _j_T(pos, vel, Q, Qn):
    """Computes transversivel projected current"""
    return cp.sum((vel[:, :, cp.newaxis] - vel.dot(Qn.T)[..., cp.newaxis]*Qn) * (cp.exp(1.0j*pos.dot(Q.T))[..., cp.newaxis]), axis = 1)

cur = {'cur': _j, 'cur_L': _j_L, 'cur_T': _j_T}
