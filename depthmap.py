import os
import torch
import cv2
import vtk
from vtk.util import numpy_support
import numpy as np
from pytorch3d.renderer import (    
    look_at_view_transform,
    FoVOrthographicCameras,
    PointLights    
)
import argparse
from pathlib import Path
from utils import polydata2mesh
from renderers import initialize_depthmap_renderer

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default="data/cow_mesh/cow.obj")
    args = parser.parse_args()
    

    # Load obj file
    reader = vtk.vtkOBJReader()
    reader.SetFileName(args.input)
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
                                        znear=2.0,
                                        zfar=3.0,
                                        R=R,
                                        T=T
                                    )
    
    # differentiable renderer
    depthmap_renderer = initialize_depthmap_renderer()

    # Render Silhouette image, light needed???
    meshes = trg_mesh.extend(num_views)
    images = depthmap_renderer(meshes, cameras=cameras)

    print(torch.min(images), torch.max(images))
    
    # images = images[...,3]


    sample_image = images[0].detach().cpu().numpy()

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.imshow("output", sample_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    