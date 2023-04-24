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
    
    return mesh

def mesh2polydata(mesh):
    
    polydata = vtk.vtkPolyData()
    
    # set vertices
    verts = numpy_support.numpy_to_vtk(mesh.verts_packed().detach().cpu().numpy())
    polydata.SetPoints( vtk.vtkPoints() )
    polydata.GetPoints().SetData(verts)

    faces = numpy_support.numpy_to_vtk(mesh.faces_packed().detach().cpu().numpy().ravel())
    polydata.SetPolys( vtk.vtkCellArray() )
    polydata.GetPolys().SetData(3, faces )

    # set normals if exists
    normals = numpy_support.numpy_to_vtk(mesh.verts_normals_packed().detach().cpu().numpy())    
    normals.SetName("Normals")
    polydata.GetPointData().SetNormals(normals)

    # without this, face information gets corrupted
    results = vtk.vtkPolyData()
    results.DeepCopy(polydata)

    return results

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


def read_polydata(path):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(path)
    reader.Update()

    return reader.GetOutput()
