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
from pytorch3d.loss import (
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    FoVOrthographicCameras
)
from renderers import initialize_depthmap_renderer, initialize_silhouette_renderer
import argparse
from pathlib import Path

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")



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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default="data/cow_mesh/cow.obj")
    args = parser.parse_args()


    # Load obj file
    reader = vtk.vtkOBJReader()
    reader.SetFileName(str(args.input))
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
    R, T = look_at_view_transform(dist=2.7, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device,
                                        znear=1.5,
                                        zfar=3.5,
                                        R=R,
                                        T=T
                                    )

    # differentiable renderer
    renderer = initialize_depthmap_renderer()

    # Render Silhouette image, light needed???
    meshes = trg_mesh.extend(num_views)
    silhouette_images = renderer(meshes, cameras=cameras)

    sample_image = silhouette_images[0].detach().cpu().numpy()
    print(sample_image.shape)
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)

    cv2.imshow("output", sample_image)
    cv2.waitKey(0)

    
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
    # target_polydata = mesh2polydata(mesh)
    trg_actor = make_actor(trg_polydata)
    trg_actor.GetProperty().SetOpacity(0.2)
    # ren.AddActor(target_actor)
    
    template_target_actor = make_actor(trg_polydata)
    template_target_actor.SetPosition(1.2, 0, 0)
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
    w_edge = 0.1
    w_normal = 0.1
    w_laplacian = 0.001
    

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
        pred = model(sample_input) * 0.05
        new_src_mesh = src_mesh.offset_verts(pred[0])
        
        # Losses to smooth /regularize the mesh shape
        loss_edge = mesh_edge_loss(new_src_mesh)    * w_edge   
        loss_normal = mesh_normal_consistency(new_src_mesh)  * w_normal       
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform") * w_laplacian


        # Silhouette Renderer
        loss_silhouette = torch.tensor(0.0, device=device)
        for j in np.random.permutation(num_views).tolist()[:1]:
            j=0
            # Differentiable Render            
            images_predicted = renderer(new_src_mesh, cameras=cameras[j]) 
            predicted_silhouette = images_predicted # only 4th channels is meaningful
            gt_image = silhouette_images[j]
            
            l_s = ((predicted_silhouette - gt_image) ** 2).mean()
            loss_silhouette += l_s / num_views_per_iteration * w_silhoutte
        
        sample_output = torch.cat([predicted_silhouette[0], gt_image])
        cv2.imshow("output", sample_output.detach().cpu().numpy())
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