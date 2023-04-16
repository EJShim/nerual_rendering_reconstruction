import torch
import numpy as np
import openmesh as om
from sklearn.neighbors import KDTree


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_unique_index(mat, device):

    
    tensor_size = mat.shape[0]    
    results = [torch.zeros(tensor_size, dtype=torch.int32) - 1]
    for idx, val in enumerate(mat):
        add = True
        for rid, res in enumerate( results):
            if not val in res:
                results[rid][idx] = val
                add = False
                break
            
        if add:
            #if it does not break:
            new_tensor = torch.zeros(tensor_size, dtype=torch.int32) - 1
            new_tensor[idx] = val            
            results.append(new_tensor)


    return results

def to_sparse(spmat, device):

    sptensor = spmat.tocoo()
    result = {
        "rows":to_unique_index(sptensor.row,  device),
        "row":torch.LongTensor(sptensor.row).to(device),
        "col":torch.LongTensor(sptensor.col).to(device),
        "data" : torch.FloatTensor(sptensor.data).to(device), 
        "size" : torch.Size(sptensor.shape)
    }

    return result
    
def to_edge_index(mat):
    return torch.LongTensor(np.vstack(mat.nonzero()))



## Spiral
def _next_ring(mesh, last_ring, other):
    res = []

    def is_new_vertex(idx):
        return (idx not in last_ring and idx not in other and idx not in res)

    for vh1 in last_ring:
        vh1 = om.VertexHandle(vh1)
        after_last_ring = False
        for vh2 in mesh.vv(vh1):
            if after_last_ring:
                if is_new_vertex(vh2.idx()):
                    res.append(vh2.idx())
            if vh2.idx() in last_ring:
                after_last_ring = True
        for vh2 in mesh.vv(vh1):
            if vh2.idx() in last_ring:
                break
            if is_new_vertex(vh2.idx()):
                res.append(vh2.idx())
    return res


def extract_spirals(mesh, seq_length, dilation=1):
    # output: spirals.size() = [N, seq_length]
    spirals = []
    for vh0 in mesh.vertices():
        reference_one_ring = []
        for vh1 in mesh.vv(vh0):
            reference_one_ring.append(vh1.idx())
        spiral = [vh0.idx()]
        one_ring = list(reference_one_ring)
        last_ring = one_ring
        next_ring = _next_ring(mesh, last_ring, spiral)
        spiral.extend(last_ring)
        while len(spiral) + len(next_ring) < seq_length * dilation:
            if len(next_ring) == 0:
                break
            last_ring = next_ring
            next_ring = _next_ring(mesh, last_ring, spiral)
            spiral.extend(last_ring)
        if len(next_ring) > 0:
            spiral.extend(next_ring)
        else:
            kdt = KDTree(mesh.points(), metric='euclidean')
            spiral = kdt.query(np.expand_dims(mesh.points()[spiral[0]],axis=0),k=seq_length * dilation,
                               return_distance=False).tolist()
            spiral = [item for subspiral in spiral for item in subspiral]
        spirals.append(spiral[:seq_length * dilation][::dilation])
    return spirals


def preprocess_spiral(face, seq_length, vertices=None, dilation=1):    
    assert face.shape[1] == 3
    if vertices is not None:
        mesh = om.TriMesh(np.array(vertices), np.array(face))
    else:
        n_vertices = face.max() + 1
        mesh = om.TriMesh(np.ones([n_vertices, 3]), np.array(face))
    spirals = torch.tensor(
        extract_spirals(mesh, seq_length=seq_length, dilation=dilation))
    return spirals
