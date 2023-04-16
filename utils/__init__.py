import vtk
import numpy as np
from vtk.util import numpy_support
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

def polydata2mesh(polydata):
    
    verts = numpy_support.vtk_to_numpy( polydata.GetPoints().GetData())
    faces = numpy_support.vtk_to_numpy(polydata.GetPolys().GetData())
    faces = faces.reshape( int(faces.shape[0]/4), 4 )[:,1:]

    verts = torch.tensor(verts)
    faces = torch.tensor(faces)

    # Textures mandatory for neural rendering??
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = TexturesVertex(verts_features=verts_rgb)

    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

    # Normalize MEsh
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)));
    
    return mesh

def mesh2polydata(mesh):
    verts = mesh.verts_packed().detach().cpu().numpy()
    faces = mesh.faces_packed().detach().cpu().numpy()

    polydata = vtk.vtkPolyData()
    polydata.SetPoints( vtk.vtkPoints() )
    polydata.GetPoints().SetData(numpy_support.numpy_to_vtk(verts))
    
    polydata.SetPolys( vtk.vtkCellArray() )
    polydata.GetPolys().SetData(3, numpy_support.numpy_to_vtk(faces.ravel()) )

    return polydata

def make_actor(polydata):
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


# def textured_sphere():
#     mesh = ico_sphere(4)
#     verts_rgb = torch.ones_like(mesh.verts_packed())[None]  # (1, V, 3)
#     textures = TexturesVertex(verts_features=verts_rgb)
#     mesh.textures = textures
    
#     return mesh
