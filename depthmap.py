import os
import torch
import cv2
import vtk
from vtk.util import numpy_support
from utils import polydata2mesh, mesh2polydata, make_actor
from pytorch3d.utils import ico_sphere
import numpy as np
from tqdm import tqdm

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, save_obj

from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)
from plot_image_grid import image_grid


# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")



class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        # images = self.shader(fragments, meshes_world, **kwargs)
        return fragments.zbuf


def initialize_depthmap_renderer():
    # Rasterization settings for silhouette rendering  
    
    raster_settings = RasterizationSettings(
        image_size=128        
    )
    raster_settings.perspective_correct = True

    rasterizer = MeshRasterizer(raster_settings=raster_settings)

    # Silhouette renderer 
    renderer = MeshRendererWithDepth(
        rasterizer= rasterizer,
        shader=SoftPhongShader()
    )

    return renderer


if __name__ == "__main__":
    # Set paths
    DATA_DIR = "./data"
    obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

    # Load obj file
    mesh = load_objs_as_meshes([obj_filename], device=device)

    # Normalize Mesh
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)));



    # the number of different viewpoints from which we want to render the mesh.

    # Initialize Cameras
    num_views = 20
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)
    R, T = look_at_view_transform(dist=1.0, elev=elev, azim=azim)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)

    # differentiable renderer
    depthmap_renderer = initialize_depthmap_renderer()

    # Render Silhouette image, light needed???
    meshes = mesh.extend(num_views)
    images = depthmap_renderer(meshes, cameras=cameras)

    print(torch.min(images), torch.max(images))
    
    # images = images[...,3]


    sample_image = images[0].detach().cpu().numpy()

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.imshow("output", sample_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    