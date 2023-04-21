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
        results = fragments.zbuf + 1
        results = results / 3
        return results

def initialize_depthmap_renderer():
    # Rasterization settings for silhouette rendering  
    
    raster_settings = RasterizationSettings(
        image_size=256     
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
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_filename)
    reader.Update()

    trg_polydata = reader.GetOutput()

    trg_mesh = polydata2mesh(trg_polydata).to(device)

    # Normalize Mesh
    verts = trg_mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    trg_mesh.offset_verts_(-center)
    trg_mesh.scale_verts_((1.0 / float(scale)))



    # the number of different viewpoints from which we want to render the mesh.

    # Initialize Cameras
    num_views = 20
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)
    R, T = look_at_view_transform(dist=2.0, elev=0, azim=180)
    cameras = FoVOrthographicCameras(device=device,
                                        znear=0.0,
                                        zfar=2.0,
                                        R=R,
                                        T=T
                                    )

    # differentiable renderer
    renderer = initialize_depthmap_renderer()

    # Render Silhouette image, light needed???
    meshes = trg_mesh.extend(num_views)
    trg_images = renderer(meshes, cameras=cameras)
    # trg_images = trg_images[...,3]


    sample_image = trg_images[0].detach().cpu().numpy()

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)

    cv2.imshow("output", sample_image)
    cv2.waitKey(1)
    
    # We initialize the source shape to be a sphere of radius 1.  
    src_mesh = ico_sphere(4, device)



    ## Initialize Renderwindow
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1000 , 1000)
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    ren.SetBackground(.9, .9, .9)
    renWin.AddRenderer(ren)

    # Render TArget
    # target_polydata = mesh2polydata(trg_mesh)
    target_actor = make_actor(trg_polydata)
    target_actor.GetProperty().SetOpacity(0.2)
    ren.AddActor(target_actor)


    ### Run Trianing
    src_polydata = mesh2polydata(src_mesh)
    src_polydata.GetPointData().RemoveArray("Normals")
    src_actor = make_actor(src_polydata)
    src_actor.GetProperty().SetColor(0.5, 0.2, 0.2)

    ren.AddActor(src_actor)
    ren.ResetCamera()
    
    iren.InvokeEvent(vtk.vtkCommand.StartEvent, None)
    iren.Initialize()
    renWin.Render()
    

    # Number of views to optimize over in each SGD iteration
    num_views_per_iteration = 5
    Niter = 2000

    # Loss weights
    w_silhoutte = 1.0
    w_edge = 1.0
    w_normal = 0.01
    w_laplacian = 1.0
    

    # Direct vertices optimization
    verts_shape = src_mesh.verts_packed().shape
    deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)

    # The optimizer
    optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

    loop = tqdm(range(Niter))

    for i in loop:

        iren.ProcessEvents()
        renWin.Render()

        # Initialize optimizer
        optimizer.zero_grad()
        
        # Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts)
        
        # Losses to smooth /regularize the mesh shape
        loss_edge = mesh_edge_loss(new_src_mesh)    * w_edge   
        loss_normal = mesh_normal_consistency(new_src_mesh)  * w_normal       
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform") * w_laplacian


        # Silhouette Renderer
        loss_silhouette = torch.tensor(0.0, device=device)
        for j in np.random.permutation(num_views).tolist()[:1]:
            j = 0
            # Differentiable Render            
            images_predicted = renderer(new_src_mesh, cameras=cameras[j]) 
            predicted_silhouette = images_predicted # only 4th channels is meaningful
            
            l_s = ((predicted_silhouette - trg_images[j]) ** 2).mean()
            loss_silhouette += l_s / num_views_per_iteration * w_silhoutte
        
        cv2.imshow("output", predicted_silhouette.detach().cpu().numpy()[0])
        cv2.waitKey(1)
    
        
        # Weighted sum of the losses
        sum_loss = loss_edge + loss_normal + loss_laplacian + loss_silhouette
        

        
        # Print the losses
        loop.set_description("total_loss = %.6f" % sum_loss)
        
            
        # Optimization step
        sum_loss.backward()
        optimizer.step()


        # update src poldata
        src_polydata.GetPoints().SetData( numpy_support.numpy_to_vtk(new_src_mesh.verts_packed().detach().cpu().numpy()) )
        src_polydata.GetPoints().Modified() 

    iren.Start()