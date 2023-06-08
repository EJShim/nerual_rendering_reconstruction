from pytorch3d.utils import ico_sphere
import vtk
from utils import mesh2polydata

if __name__ == "__main__":
    mesh = ico_sphere(5)

    polydata = mesh2polydata(mesh)

    writer = vtk.vtkOBJWriter()
    writer.SetInputData(polydata)
    writer.SetFileName("ico5.obj")
    writer.Write()