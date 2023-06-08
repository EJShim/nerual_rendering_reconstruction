import os
import torch
import cv2
import vtk
from pytorch3d.structures import Meshes
# Data structures and functions for rendering
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
)
from renderers import initialize_depthmap_renderer
from vtk.util import numpy_support
# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")



class NeuralDepthRenderer():
    def __init__(self, device="cuda"):
        
        self.elev = 0
        self.azim = 0
        self.dist = 2.5
        self.znear = 1.5
        self.zfar = 3.0

        self.renderer = initialize_depthmap_renderer()
        
        self.input = None
        self.output = None
        self.device = device


    def set_polydata(self, polydata):
        self.input = polydata2mesh(polydata).to(device)

    def set_data(self, V, F):
        self.input = Meshes(verts=[V], faces=[F])

    def set_mesh(self, mesh):
        self.input = mesh

    def render(self):
        if self.input == None:            
            return

        # Normalize Mesh
        verts = self.input.verts_packed()
        # N = verts.shape[0]
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        self.input.offset_verts_(-center)
        self.input.scale_verts_((1.0 / float(scale)))

        # Set Camera
        R, T = look_at_view_transform(dist=self.dist, elev=self.elev, azim=self.azim)
        cameras = FoVOrthographicCameras(device=self.device,
                                        znear=self.znear,
                                        zfar=self.zfar,
                                        R=R,
                                        T=T)
        
        depth_tensor = self.renderer(self.input, cameras=cameras)
        self.output = depth_tensor


        return depth_tensor



def polydata2mesh(polydata):
    
    verts = numpy_support.vtk_to_numpy( polydata.GetPoints().GetData())
    faces = numpy_support.vtk_to_numpy(polydata.GetPolys().GetData())
    faces = faces.reshape( int(faces.shape[0]/4), 4 )[:,1:]

    verts = torch.tensor(verts)
    faces = torch.tensor(faces)

    # Textures mandatory for neural rendering??
    verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
    # textures = TexturesVertex(verts_features=verts_rgb)

    mesh = Meshes(verts=[verts], faces=[faces])
    
    return mesh

def render_depthmap(polydata):

    trg_mesh = polydata2mesh(polydata).to(device)

    # Normalize Mesh
    verts = trg_mesh.verts_packed()
    # N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    trg_mesh.offset_verts_(-center)
    trg_mesh.scale_verts_((1.0 / float(scale)))

    # Initialize Cameras

    elev = 0
    azim = 0
    dist = 2.5
    znear = 1.5
    zfar = 3.0
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device,
                                        znear=znear,
                                        zfar=zfar,
                                        R=R,
                                        T=T
                                    )

    # differentiable renderer
    renderer = initialize_depthmap_renderer()
    silhouette_images = renderer(trg_mesh, cameras=cameras)
    # silhouette_images = silhouette_images[...,0]

    return silhouette_images[0]



if __name__ == "__main__":
    # Set paths
    DATA_DIR = "./data"
    obj_filename = os.path.join(DATA_DIR, "cape_simulation.obj")

    # Load obj file
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_filename)
    reader.Update()

    input_poly = reader.GetOutput()

    #
    renderer = NeuralDepthRenderer()
    renderer.set_polydata(input_poly)
    out = renderer.render()

    image = out[0].detach().cpu().numpy()



    cv2.imshow('output', cv2.WINDOW_NORMAL)
    cv2.imshow("output", image)
    cv2.waitKey(0)