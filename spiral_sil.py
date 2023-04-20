import os
import torch
import cv2
import vtk
from vtk.util import numpy_support
from utils import polydata2mesh, mesh2polydata, make_actor
from pytorch3d.utils import ico_sphere
import numpy as np
from tqdm import tqdm
from utils.mesh_sampling import generate_transform
from utils import sparse
from network.spiralnetp import Decoder

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

def initialize_model(polydata):    

    downsample = [4,4,4,4]
    seq_length = [9,9,9,9]
    dilation = [1,1,1,1]
    in_channels = 3
    out_channels = [32,32,32,64]
    latent_channels=64


    transform = generate_transform(polydata, downsample)
    spiral_indices_list = [
        sparse.preprocess_spiral(transform['face'][idx], seq_length[idx], 
                                transform['vertices'][idx], dilation[idx]).to(device)
        for idx in range(len(transform['face']) - 1)
    ]
    down_transform_list = [
        sparse.to_sparse(down_transform, device)
        for down_transform in transform['down_transform']
    ]
    up_transform_list = [
        sparse.to_sparse(up_transform, device)
        for up_transform in transform['up_transform']
    ]

    model = Decoder(in_channels, out_channels, latent_channels, spiral_indices_list, up_transform_list).to(device)

    return model


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
    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # differentiable renderer
    silhouette_renderer = initialize_silhouette_renderer()

    # Render Silhouette image, light needed???
    meshes = mesh.extend(num_views)
    silhouette_images = silhouette_renderer(meshes, cameras=cameras)
    silhouette_images = silhouette_images[...,3]


    sample_image = silhouette_images[0].detach().cpu().numpy()

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
    target_polydata = mesh2polydata(mesh)
    target_actor = make_actor(target_polydata)
    target_actor.GetProperty().SetOpacity(0.2)
    ren.AddActor(target_actor)
    
    template_target_actor = make_actor(target_polydata)
    template_target_actor.SetPosition(1, 0, 0)
    ren.AddActor(template_target_actor)



    ### Run Trianing
    src_polydata = mesh2polydata(src_mesh)
    src_polydata.GetPointData().RemoveArray("Normals")
    src_actor = make_actor(src_polydata)
    src_actor.GetProperty().SetColor(0.2, 0.5, 0.2)

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
    # verts_shape = src_mesh.verts_packed().shape
    # deform_verts = torch.full(verts_shape, 0.0, device=device, requires_grad=True)
    sample_input = torch.randn([1,64]).to(device)
    model = initialize_model(src_polydata).to(device)
    model.train()

    # The optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    loop = tqdm(range(Niter))

    for i in loop:

        iren.ProcessEvents()
        renWin.Render()

        # Initialize optimizer
        optimizer.zero_grad()
        
        # Deform the mesh
        pred = model(sample_input) * 0.01
        new_src_mesh = src_mesh.offset_verts(pred[0])
        
        # Losses to smooth /regularize the mesh shape
        loss_edge = mesh_edge_loss(new_src_mesh)    * w_edge   
        loss_normal = mesh_normal_consistency(new_src_mesh)  * w_normal       
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform") * w_laplacian


        # Silhouette Renderer
        loss_silhouette = torch.tensor(0.0, device=device)
        for j in np.random.permutation(num_views).tolist()[:5]:
            # j = 0
            # Differentiable Render            
            images_predicted = silhouette_renderer(new_src_mesh, cameras=cameras[j]) 
            predicted_silhouette = images_predicted[..., 3] # only 4th channels is meaningful
            
            l_s = ((predicted_silhouette - silhouette_images[j]) ** 2).mean()
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