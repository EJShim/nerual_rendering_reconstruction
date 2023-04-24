
import torch
from pytorch3d.renderer.mesh.shader import SoftDepthShader
import numpy as np
from pytorch3d.renderer import (
    blending,
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer, 
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.shader import ShaderBase


    
# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


class DepthShader(ShaderBase):

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)

        # unsqueeze b, 1, 1, 1
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        znear = kwargs.get("znear", getattr(cameras, "znear", 0.1))
        zfar = zfar.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        znear = znear.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        
        cameras = kwargs['cameras']
        # not sure this can be dist, cameras' focal points always origin?
        dist = cameras.T.norm()

        

        background_mask = fragments.pix_to_face[..., 0:1] < 0
        zbuf = fragments.zbuf[..., 0:1].clone()


        # normalize zbuf using zfar and znear
        zbuf = (zfar - zbuf)/(zfar - znear)
        zbuf[background_mask] = 0.0

        return zbuf



class MeshRendererWithDepth(torch.nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        return images

class SilhouetteRenderer(MeshRendererWithDepth):
    
    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        images = super().forward(meshes_world, **kwargs)
        images = images[...,3]

        return images
    

def initialize_depthmap_renderer():
    # Rasterization settings for silhouette rendering  
    
    raster_settings = RasterizationSettings(
        image_size=256,
        blur_radius = 0.0,
        faces_per_pixel = 1,     
    )
    
    blend_params = blending.BlendParams(background_color=[0,0,0])
    rasterizer = MeshRasterizer(raster_settings=raster_settings)

    # Silhouette renderer 
    renderer = MeshRendererWithDepth(
        rasterizer= rasterizer,
        shader=DepthShader(
            device=device,
            blend_params=blend_params
        )
    )

    return renderer



def initialize_silhouette_renderer():
    # Rasterization settings for silhouette rendering  
    sigma = 1e-4
    raster_settings_silhouette = RasterizationSettings(
        image_size=256, 
        blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
        faces_per_pixel=50, 
    )

    # Silhouette renderer 
    renderer_silhouette = SilhouetteRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings_silhouette),
        shader=SoftSilhouetteShader()
    )

    return renderer_silhouette
