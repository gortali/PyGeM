"""
Utilities for reading and writing parameters files to perform RBF
geometrical morphing.
"""
try:
    import configparser as configparser
except ImportError:
    import ConfigParser as configparser
import os
import numpy as np
import vtk
import matplotlib.pyplot as plt


class RBFParameters(object):
    """
    Class that handles the Radial Basis Functions parameters in terms of RBF
    control points and basis functions.

    :cvar string basis: name of the basis functions to use in the
        transformation. The functions implemented so far are: gaussian spline,
        multi quadratic biharmonic spline, inv multi quadratic biharmonic
        spline, thin plate spline, beckert wendland c2 basis, polyharmonic
        splines. For a comprehensive list with details see the class
        :class:`~pygem.radialbasis.RBF`. The default value is 'gaussian_spline'.
    :cvar float radius: the scaling parameter r that affects the shape of the
        basis functions.  For details see the class
        :class:`~pygem.radialbasis.RBF`. The default value is 0.5.
    :cvar int power: the power parameter that affects the shape of the basis
        functions.  For details see the class :class:`~pygem.radialbasis.RBF`.
        The default value is 2.
    :cvar numpy.ndarray original_control_points: *n_control_points*-by-3 array
        with the coordinates of the original interpolation control points
        before the deformation. The default values are the coordinates of unit
        cube vertices.
    :cvar numpy.ndarray deformed_control_points: it is an
        `n_control_points`-by-3 array with the coordinates of the
        interpolation control points after the deformation. The default values
        are the coordinates of the unit cube vertices.
    """

    def __init__(self):
        self.basis = 'gaussian_spline'
        self.radius = 0.5
        self.power = 2
        self.original_control_points = np.array(
            [[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.],
             [0., 1., 1.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]])
        self.deformed_control_points = np.array(
            [[0., 0., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.],
             [0., 1., 1.], [1., 0., 1.], [1., 1., 0.], [1., 1., 1.]])

    @property
    def n_control_points(self):
        """
        Total number of control points.

        :rtype: int
        """
        return self.original_control_points.shape[0]

    def read_parameters(self, filename='parameters_rbf.prm'):
        """
        Reads in the parameters file and fill the self structure.

        :param string filename: parameters file to be read in. Default value is
            parameters_rbf.prm.
        """
        if not isinstance(filename, str):
            raise TypeError('filename must be a string')

        # Checks if the parameters file exists. If not it writes the default
        # class into filename.  It consists in the vetices of a cube of side one
        # with a vertex in (0, 0, 0) and opposite one in (1, 1, 1).
        if not os.path.isfile(filename):
            self.write_parameters(filename)
            return

        config = configparser.RawConfigParser()
        config.read(filename)

        self.basis = config.get('Radial Basis Functions', 'basis function')
        self.radius = config.getfloat('Radial Basis Functions', 'radius')
        self.power = config.getint('Radial Basis Functions', 'power')

        ctrl_points = config.get('Control points', 'original control points')
        lines = ctrl_points.split('\n')
        self.original_control_points = np.zeros((len(lines), 3))
        for line, i in zip(lines, list(range(0, self.n_control_points))):
            values = line.split()
            self.original_control_points[i] = np.array(
                [float(values[0]),
                 float(values[1]),
                 float(values[2])])

        mod_points = config.get('Control points', 'deformed control points')
        lines = mod_points.split('\n')

        if len(lines) != self.n_control_points:
            raise TypeError("The number of control points must be equal both in"
                            "the 'original control points' and in the 'deformed"
                            "control points' section of the parameters file"
                            "({0!s})".format(filename))

        self.deformed_control_points = np.zeros((self.n_control_points, 3))
        for line, i in zip(lines, list(range(0, self.n_control_points))):
            values = line.split()
            self.deformed_control_points[i] = np.array(
                [float(values[0]),
                 float(values[1]),
                 float(values[2])])

    def write_parameters(self, filename='parameters_rbf.prm'):
        """
        This method writes a parameters file (.prm) called `filename` and fills
        it with all the parameters class members. Default value is
        parameters_rbf.prm.

        :param string filename: parameters file to be written out.
        """
        if not isinstance(filename, str):
            raise TypeError("filename must be a string")

        output_string = ""
        output_string += '\n[Radial Basis Functions]\n'
        output_string += '# This section describes the radial basis functions'
        output_string += ' shape.\n'

        output_string += '\n# basis funtion is the name of the basis functions'
        output_string += ' to use in the transformation. The functions\n'
        output_string += '# implemented so far are: gaussian_spline,'
        output_string += ' multi_quadratic_biharmonic_spline,\n'
        output_string += '# inv_multi_quadratic_biharmonic_spline,'
        output_string += ' thin_plate_spline, beckert_wendland_c2_basis,'
        output_string += ' polyharmonic_spline.\n'
        output_string += '# For a comprehensive list with details see the'
        output_string += ' class RBF.\n'
        output_string += 'basis function: {}\n'.format(str(self.basis))

        output_string += '\n# radius is the scaling parameter r that affects'
        output_string += ' the shape of the basis functions. See the'
        output_string += ' documentation\n'
        output_string += '# of the class RBF for details.\n'
        output_string += 'radius: {}\n'.format(str(self.radius))

        output_string += '\n# The power parameter k for polyharmonic spline'
        output_string += '\n# See the documentation for details\n'
        output_string += 'power: {}\n'.format(self.power)

        output_string += '\n\n[Control points]\n'
        output_string += '# This section describes the RBF control points.\n'

        output_string += '\n# original control points collects the coordinates'
        output_string += ' of the interpolation control points before the'
        output_string += ' deformation.\n'

        output_string += 'original control points:'
        offset = 1
        for i in range(0, self.n_control_points):
            output_string += offset * ' ' + str(
                self.original_control_points[i][0]) + '   ' + str(
                    self.original_control_points[i][1]) + '   ' + str(
                        self.original_control_points[i][2]) + '\n'
            offset = 25

        output_string += '\n# deformed control points collects the coordinates'
        output_string += ' of the interpolation control points after the'
        output_string += ' deformation.\n'

        output_string += 'deformed control points:'
        offset = 1
        for i in range(0, self.n_control_points):
            output_string += offset * ' ' + str(
                self.deformed_control_points[i][0]) + '   ' + str(
                    self.deformed_control_points[i][1]) + '   ' + str(
                        self.deformed_control_points[i][2]) + '\n'
            offset = 25

        with open(filename, 'w') as f:
            f.write(output_string)

    def __str__(self):
        """
        This method prints all the RBF parameters on the screen. Its purpose is
        for debugging.
        """
        string = ''
        string += 'basis function = {}\n'.format(self.basis)
        string += 'radius = {}\n'.format(self.radius)
        string += 'power = {}\n'.format(self.power)
        string += '\noriginal control points =\n'
        string += '{}\n'.format(self.original_control_points)
        string += '\ndeformed control points =\n'
        string += '{}\n'.format(self.deformed_control_points)
        return string

    def save_points(self, filename, write_deformed=True):
        """
        Method that writes a vtk file containing the control points. This method
        allows to visualize where the RBF control points are located before the
        geometrical morphing. If the `write_deformed` flag is set to True the
        method writes out the deformed points, otherwise it writes one the
        original points.

        :param str filename: name of the output file.
        :param bool write_deformed: flag to write the original or modified
            control lattice. The default is set to True.

        :Example:

            >>> from pygem.params import RBFParameters
            >>>
            >>> params = RBFParameters()
            >>> params.read_parameters(
            >>>     filename='tests/test_datasets/parameters_rbf_cube.prm')
            >>> params.save_points('tests/test_datasets/box_cube.vtk')

        .. note::
            In order to visualize the points in Paraview, please select the
            **Point Gaussian** representation.

        """
        box_points = self.deformed_control_points if write_deformed else self.original_control_points
        points = vtk.vtkPoints()

        for box_point in box_points:
            points.InsertNextPoint(box_point[0], box_point[1], box_point[2])

        data = vtk.vtkPolyData()
        data.SetPoints(points)

        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(data)
        writer.Write()

    def plot_points(self, filename=None):
        """
        Method to plot the control points. It is possible to save the resulting
        figure.

        :param str filename: if None the figure is shown, otherwise it is saved
            on the specified `filename`. Default is None.
        """
        fig = plt.figure(1)
        axes = fig.add_subplot(111, projection='3d')
        orig = axes.scatter(
            self.original_control_points[:, 0],
            self.original_control_points[:, 1],
            self.original_control_points[:, 2],
            c='blue',
            marker='o')
        defor = axes.scatter(
            self.deformed_control_points[:, 0],
            self.deformed_control_points[:, 1],
            self.deformed_control_points[:, 2],
            c='red',
            marker='x')

        axes.set_xlabel('X axis')
        axes.set_ylabel('Y axis')
        axes.set_zlabel('Z axis')

        plt.legend(
            (orig, defor), ('Original', 'Deformed'),
            scatterpoints=1,
            loc='lower left',
            ncol=2,
            fontsize=10)

        # Show the plot to the screen
        if filename is None:
            plt.show()
        else:
            fig.savefig(filename)
