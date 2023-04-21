
import torch
from pytorch3d.renderer.mesh.shader import SoftDepthShader, HardDepthShader
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
from pytorch3d.renderer.mesh.shader import SoftDepthShader


    
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
        images = self.shader(fragments, meshes_world, **kwargs)

        results = -images
        results += 3
        results /= 2

        return results


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
        shader=HardDepthShader(
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
    renderer_silhouette = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings_silhouette),
        shader=SoftSilhouetteShader()
    )

    return renderer_silhouette
