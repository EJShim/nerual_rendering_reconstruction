from pickletools import optimize
from re import template
import vtk
from vtk.util import numpy_support
from network.spiralnetp import Decoder
import torch
from pytorch3d.utils import ico_sphere
from utils import mesh2polydata, polydata2mesh
from utils.mesh_sampling import generate_transform, vertex_quadrics
from utils import sparse
from utils import make_actor
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import tqdm

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
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    renWin = vtk.vtkRenderWindow()
    renWin.SetSize(1000, 1000)
    iren.SetRenderWindow(renWin)
    ren = vtk.vtkRenderer()
    renWin.AddRenderer(ren)

    # Read Target Polydata
    reader = vtk.vtkOBJReader()
    reader.SetFileName("data/cow_mesh/cow.obj")
    reader.Update()
    trg_poly = reader.GetOutput()
    trg_actor = make_actor(trg_poly)
    trg_mesh = polydata2mesh(trg_poly)
    ren.AddActor(trg_actor)

    # topology for model training
    template_mesh = ico_sphere(4)
    template_poly = mesh2polydata(template_mesh)
    actor = make_actor(template_poly)
    ren.AddActor(actor)

    # Initialize Model
    model = initialize_model(template_poly)
    model.train()

    # Random Input
    sample_input = torch.randn([1,64])

    # Random Output

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.50)

    # Weight for the chamfer loss
    w_chamfer = 0.001
    # Weight for mesh edge loss
    w_edge = 1.0 
    # Weight for mesh normal consistency
    w_normal = 0.01 
    # Weight for mesh laplacian smoothing
    w_laplacian = 0.1 

    # std = 0.1
    # mean = template_mesh.verts_packed()

    iren.InvokeEvent(vtk.vtkCommand.StartEvent, None)
    iren.Initialize()
    


    ren.ResetCamera()
    renWin.Render()

    for i in tqdm.tqdm(range(500)):
        pred = model(sample_input)
        pred_mesh = Meshes(verts=pred, faces=template_mesh.faces_packed().unsqueeze(0))
        # pred_mesh = template_mesh.offset_verts(pred[0])
        # Chamfer Distance between target mesh and pred mesh
        # We sample 5k points from the surface of each mesh 
        sample_trg = sample_points_from_meshes(trg_mesh, 5000)
        sample_src = sample_points_from_meshes(pred_mesh, 5000)
        
        # We compare the two sets of pointclouds by computing (a) the chamfer loss
        loss_chamfer, _ = chamfer_distance(sample_src, sample_trg)


        # and (b) the edge length of the predicted mesh
        loss_edge = mesh_edge_loss(pred_mesh)
        
        # mesh normal consistency
        loss_normal = mesh_normal_consistency(pred_mesh)
        
        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(pred_mesh, method="uniform")
        
        # Weighted sum of the losses
        loss = loss_chamfer * w_chamfer

        loss.backward()
        optimizer.step()


        # Set Initial Output Sphere???
        sample_output = pred_mesh.verts_packed()        
        sample_output = sample_output.detach().cpu().numpy()
        template_poly.GetPoints().SetData( numpy_support.numpy_to_vtk( sample_output ))
        renWin.Render()
        iren.ProcessEvents()

        
    renWin.Render()
    iren.Start()
