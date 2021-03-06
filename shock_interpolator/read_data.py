import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import os
import pickle

def main():
    # CFD results directory
    cfd_results = 'DeepLearningProject'
    n_cases = 67

    # Dictionary to hold processed data
    data = {}
    # Loop over directories
    for directory in os.listdir(cfd_results):
        # Name of case
        case = directory.split('_')[-1]
        data[case] = ([], [], [])
        print(f'Processing the {case} cases')
        print('Working on case: ', end='', flush=True)

        # Loop over possible cases
        for i in range(n_cases):
            print(f'{i}, ', end='', flush=True)
            # Data file
            file_name = f'{cfd_results}/{directory}/flow_{i}.vtu'
            # Check if it exists
            if os.path.exists(file_name):
                # If it does, process it
                x, pressure, grad = process_data(file_name)
                data[case][0].append(x)
                data[case][1].append(pressure)
                data[case][2].append(grad)
            # If it doesn't exist, append None
            else:
                data[case][0].append(None)
                data[case][1].append(None)
                data[case][2].append(None)
        print()

    # Write to file
    with open('data.pkl', 'wb') as outfile:
        pickle.dump(data, outfile, protocol=pickle.HIGHEST_PROTOCOL)

def process_data(file_name):
    '''
    Read pressure data in from a VTU file generated by SU2.
    '''
    # Read the VTU file
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    output = reader.GetOutput()

    # Extract pressure and location into Numpy arrays
    pressure = vtk_to_numpy(output.GetPointData().GetArray("Pressure"))
    x = vtk_to_numpy(output.GetPoints().GetData())
    conn = vtk_to_numpy(output.GetCells().GetData())
    # Remove the unnecessary number of vertices (there is a "4" every 4
    # points...)
    conn = conn[np.mod(np.arange(conn.size),5)!=0]
    # Remove the z-axis since it's 2D
    x = x[:, :2]
    n_pts = pressure.size
    grad = np.empty((n_pts, 2))
    # Loop over points
    for i in range(n_pts):
        # Cells containing this point
        cells = np.argwhere(conn == i)[:, 0] // 4
        # Loop cells
        points_list = []
        for cell in cells:
            # Points in this cell
            cell_points = conn[4*cell:4*cell+4]
            points_list.append(cell_points)
        # Get unique points in single array
        points = np.unique(np.concatenate(points_list))
        # Remove the original point
        points = points[points != i]
        # Compute distances
        dist = x[points] - x[i]
        dist_norm = np.linalg.norm(dist, axis=1)
        dist_unit = dist / dist_norm.reshape(-1, 1)
        # Compute pressure differences
        diff = pressure[points] - pressure[i]
        # Compute gradient in the direction of these points
        grad_norm = diff / dist_norm
        # Multiply with unit vector to get directional gradient
        grad_vec = dist_unit * grad_norm.reshape(-1, 1)
        # Sum all gradient vectors to get the resultant
        grad[i] = np.sum(grad_vec, axis = 0)

    return x, pressure, grad

if __name__ == '__main__':
    main()
