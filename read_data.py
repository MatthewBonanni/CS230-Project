import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import pickle

# The data file
file_name = "flow.vtu"

# Read the VTU file
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(file_name)
reader.Update()
output = reader.GetOutput()

# Extract pressure and location into Numpy arrays
pressure = vtk_to_numpy(output.GetPointData().GetArray("Pressure"))
x = vtk_to_numpy(output.GetPoints().GetData())
# Remove the z-axis since it's 2D
x = x[:, :2]

# Write to file
with open('data.pkl', 'wb') as outfile:
    pickle.dump((x, pressure), outfile, protocol=pickle.HIGHEST_PROTOCOL)
