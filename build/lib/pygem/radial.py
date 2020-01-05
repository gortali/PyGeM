"""
Module focused on the implementation of the Radial Basis Functions interpolation
technique.  This technique is still based on the use of a set of parameters, the
so-called control points, as for FFD, but RBF is interpolatory. Another
important key point of RBF strategy relies in the way we can locate the control
points: in fact, instead of FFD where control points need to be placed inside a
regular lattice, with RBF we hano no more limitations. So we have the
possibility to perform localized control points refiniments.
The module is analogous to the freeform one.

:Theoretical Insight:

    As reference please consult M.D. Buhmann, Radial Basis Functions, volume 12
    of Cambridge monographs on applied and computational mathematics. Cambridge
    University Press, UK, 2003.  This implementation follows D. Forti and G.
    Rozza, Efficient geometrical parametrization techniques of interfaces for
    reduced order modelling: application to fluid-structure interaction coupling
    problems, International Journal of Computational Fluid Dynamics.

    RBF shape parametrization technique is based on the definition of a map,
    :math:`\\mathcal{M}(\\boldsymbol{x}) : \\mathbb{R}^n \\rightarrow
    \\mathbb{R}^n`, that allows the possibility of transferring data across
    non-matching grids and facing the dynamic mesh handling. The map introduced
    is defines as follows

    .. math::
        \\mathcal{M}(\\boldsymbol{x}) = p(\\boldsymbol{x}) + 
        \\sum_{i=1}^{\\mathcal{N}_C} \\gamma_i
        \\varphi(\\| \\boldsymbol{x} - \\boldsymbol{x_{C_i}} \\|)

    where :math:`p(\\boldsymbol{x})` is a low_degree polynomial term,
    :math:`\\gamma_i` is the weight, corresponding to the a-priori selected
    :math:`\\mathcal{N}_C` control points, associated to the :math:`i`-th basis
    function, and :math:`\\varphi(\\| \\boldsymbol{x} - \\boldsymbol{x_{C_i}}
    \\|)` a radial function based on the Euclidean distance between the control
    points position :math:`\\boldsymbol{x_{C_i}}` and :math:`\\boldsymbol{x}`.
    A radial basis function, generally, is a real-valued function whose value
    depends only on the distance from the origin, so that
    :math:`\\varphi(\\boldsymbol{x}) = \\tilde{\\varphi}(\\| \\boldsymbol{x}
    \\|)`.

    The matrix version of the formula above is:

    .. math::
        \\mathcal{M}(\\boldsymbol{x}) = \\boldsymbol{c} +
        \\boldsymbol{Q}\\boldsymbol{x} +
        \\boldsymbol{W^T}\\boldsymbol{d}(\\boldsymbol{x})

    The idea is that after the computation of the weights and the polynomial
    terms from the coordinates of the control points before and after the
    deformation, we can deform all the points of the mesh accordingly.  Among
    the most common used radial basis functions for modelling 2D and 3D shapes,
    we consider Gaussian splines, Multi-quadratic biharmonic splines, Inverted
    multi-quadratic biharmonic splines, Thin-plate splines, Beckert and
    Wendland :math:`C^2` basis and Polyharmonic splines all defined and
    implemented below.
"""
import numpy as np

from scipy.spatial.distance import cdist


class RBF(object):
    """
    Class that handles the Radial Basis Functions interpolation on the mesh
    points.

    :param RBFParameters rbf_parameters: parameters of the RBF.
    :param numpy.ndarray original_mesh_points: coordinates of the original
        points of the mesh.
    :cvar RBFParameters parameters: parameters of the RBF.
    :cvar numpy.ndarray original_mesh_points: coordinates of the original points
        of the mesh.  The shape is `n_points`-by-3.
    :cvar numpy.ndarray modified_mesh_points: coordinates of the points of the
        deformed mesh.  The shape is `n_points`-by-3.
    :cvar dict bases: a dictionary that associates the names of the basis
        functions implemented to the actual implementation.
    :cvar numpy.matrix weights: the matrix formed by the weights corresponding
        to the a-priori selected N control points, associated to the basis
        functions and c and Q terms that describe the polynomial of order one
        p(x) = c + Qx.  The shape is (n_control_points+1+3)-by-3. It is computed
        internally.

    :Example:

        >>> import pygem.radial as rbf
        >>> import pygem.params as rbfp
        >>> import numpy as np
        >>> rbf_parameters = rbfp.RBFParameters()
        >>> fname = 'tests/test_datasets/parameters_rbf_cube.prm'
        >>> rbf_parameters.read_parameters(fname)
        >>> nx, ny, nz = (20, 20, 20)
        >>> mesh = np.zeros((nx * ny * nz, 3))
        >>> xv = np.linspace(0, 1, nx)
        >>> yv = np.linspace(0, 1, ny)
        >>> zv = np.linspace(0, 1, nz)
        >>> z, y, x = np.meshgrid(zv, yv, xv)
        >>> mesh = np.array([x.ravel(), y.ravel(), z.ravel()])
        >>> original_mesh_points = mesh.T
        >>> radial_trans = rbf.RBF(rbf_parameters, original_mesh_points)
        >>> radial_trans.perform()
        >>> new_mesh_points = radial_trans.modified_mesh_points
    """

    def __init__(self, rbf_parameters, original_mesh_points):
        self.parameters = rbf_parameters
        self.original_mesh_points = original_mesh_points
        self.modified_mesh_points = None

        self.bases = {
            'gaussian_spline':
            self.gaussian_spline,
            'multi_quadratic_biharmonic_spline':
            self.multi_quadratic_biharmonic_spline,
            'inv_multi_quadratic_biharmonic_spline':
            self.inv_multi_quadratic_biharmonic_spline,
            'thin_plate_spline':
            self.thin_plate_spline,
            'beckert_wendland_c2_basis':
            self.beckert_wendland_c2_basis,
            'polyharmonic_spline':
            self.polyharmonic_spline
        }

        # to make the str callable we have to use a dictionary with all the
        # implemented radial basis functions
        if self.parameters.basis in self.bases:
            self.basis = self.bases[self.parameters.basis]
        else:
            raise NameError(
                """The name of the basis function in the parameters file is not
                correct or not implemented. Check the documentation for
                all the available functions.""")

        self.weights = self._get_weights(
            self.parameters.original_control_points,
            self.parameters.deformed_control_points)

    @staticmethod
    def gaussian_spline(X, r):
        """
        It implements the following formula:

        .. math::
            \\varphi(\\boldsymbol{x}) = e^{-\\frac{\\boldsymbol{x}^2}{r^2}}

        :param numpy.ndarray X: the norm x in the formula above.
        :param float r: the parameter r in the formula above.

        :return: result: the result of the formula above.
        :rtype: float
        """
        result = np.exp(-(X * X) / (r * r))
        return result

    @staticmethod
    def multi_quadratic_biharmonic_spline(X, r):
        """
        It implements the following formula:

        .. math::
            \\varphi(\\boldsymbol{x}) = \\sqrt{\\boldsymbol{x}^2 + r^2}

        :param numpy.ndarray X: the norm x in the formula above.
        :param float r: the parameter r in the formula above.

        :return: result: the result of the formula above.
        :rtype: float
        """
        result = np.sqrt((X * X) + (r * r))
        return result

    @staticmethod
    def inv_multi_quadratic_biharmonic_spline(X, r):
        """
        It implements the following formula:

        .. math::
            \\varphi(\\boldsymbol{x}) =
            (\\boldsymbol{x}^2 + r^2 )^{-\\frac{1}{2}}

        :param numpy.ndarray X: the norm x in the formula above.
        :param float r: the parameter r in the formula above.

        :return: result: the result of the formula above.
        :rtype: float
        """
        result = 1.0 / (np.sqrt((X * X) + (r * r)))
        return result

    @staticmethod
    def thin_plate_spline(X, r):
        """
        It implements the following formula:

        .. math::
            \\varphi(\\boldsymbol{x}) =
            \\left(\\frac{\\boldsymbol{x}}{r}\\right)^2
            \\ln\\frac{\\boldsymbol{x}}{r}

        :param numpy.ndarray X: the norm x in the formula above.
        :param float r: the parameter r in the formula above.

        :return: result: the result of the formula above.
        :rtype: float
        """
        arg = X / r
        result = arg * arg
        result = np.where(arg > 0, result * np.log(arg), result)
        return result

    @staticmethod
    def beckert_wendland_c2_basis(X, r):
        """
        It implements the following formula:

        .. math::
            \\varphi(\\boldsymbol{x}) = 
            \\left( 1 - \\frac{\\boldsymbol{x}}{r}\\right)^4 +
            \\left( 4 \\frac{ \\boldsymbol{x} }{r} + 1 \\right)

        :param numpy.ndarray X: the norm x in the formula above.
        :param float r: the parameter r in the formula above.

        :return: result: the result of the formula above.
        :rtype: float
        """
        arg = X / r
        first = np.where((1 - arg) > 0, np.power((1 - arg), 4), 0)
        second = (4 * arg) + 1
        result = first * second
        return result

    def polyharmonic_spline(self, X, r):
        """
        It implements the following formula:

        .. math::
            
            \\varphi(\\boldsymbol{x}) =
                \\begin{cases}
                \\frac{\\boldsymbol{x}}{r}^k
                    \\quad & \\text{if}~k = 1,3,5,...\\\\
                \\frac{\\boldsymbol{x}}{r}^{k-1}
                \\ln(\\frac{\\boldsymbol{x}}{r}^
                {\\frac{\\boldsymbol{x}}{r}})
                    \\quad & \\text{if}~\\frac{\\boldsymbol{x}}{r} < 1,
                    ~k = 2,4,6,...\\\\
                \\frac{\\boldsymbol{x}}{r}^k
                \\ln(\\frac{\\boldsymbol{x}}{r})
                    \\quad & \\text{if}~\\frac{\\boldsymbol{x}}{r} \\ge 1,
                    ~k = 2,4,6,...\\\\
                \\end{cases}

        :param numpy.ndarray X: the norm x in the formula above.
        :param float r: the parameter r in the formula above.

        :return: result: the result of the formula above.
        :rtype: float
        """

        k = self.parameters.power
        r_sc = X / r

        # k odd
        if k & 1:
            return np.power(r_sc, k)

        print(r_sc)
        # k even
        result = np.where(r_sc < 1,
                          np.power(r_sc, k - 1) * np.log(np.power(r_sc, r_sc)),
                          np.power(r_sc, k) * np.log(r_sc))
        return result

    def _get_weights(self, X, Y):
        """
        This private method, given the original control points and the deformed
        ones, returns the matrix with the weights and the polynomial terms, that
        is :math:`W`, :math:`c^T` and :math:`Q^T`. The shape is
        (n_control_points+1+3)-by-3.

        :param numpy.ndarray X: it is an n_control_points-by-3 array with the
            coordinates of the original interpolation control points before the
            deformation.
        :param numpy.ndarray Y: it is an n_control_points-by-3 array with the
            coordinates of the interpolation control points after the
            deformation.

        :return: weights: the matrix with the weights and the polynomial terms.
        :rtype: numpy.matrix
        """
        n_points, dim = X.shape
        H = np.zeros((n_points + 3 + 1, n_points + 3 + 1))
        H[:n_points, :n_points] = self.basis(
            cdist(X, X), self.parameters.radius)
        H[n_points, :n_points] = 1.0
        H[:n_points, n_points] = 1.0
        H[:n_points, -3:] = X
        H[-3:, :n_points] = X.T

        rhs = np.zeros((n_points + 3 + 1, dim))
        rhs[:n_points, :] = Y
        weights = np.linalg.solve(H, rhs)
        return weights

    def perform(self):
        """
        This method performs the deformation of the mesh points. After the
        execution it sets `self.modified_mesh_points`.
        """
        n_mesh_points = self.original_mesh_points.shape[0]
        n_control_points = self.parameters.original_control_points.shape[0]
        H = np.zeros((n_mesh_points, n_control_points+3+1))
        H[:, :n_control_points] = self.basis(
            cdist(self.original_mesh_points,
                  self.parameters.original_control_points),
            self.parameters.radius)
        H[:, n_control_points] = 1.0
        H[:, -3:] = self.original_mesh_points
        self.modified_mesh_points = np.asarray(np.dot(H, self.weights))
