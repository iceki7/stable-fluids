import numpy as np
from scipy.ndimage import map_coordinates, spline_filter
from scipy.sparse.linalg import factorized
from scipy.sparse import identity

from numerical import difference, operator


class Fluid:
    def __init__(self, shape, *quantities, pressure_order=1, advect_order=3):
        self.bvis=1  #swi
        self.mu=0.2 #prm    
        self.shape = shape
        self.dimensions = len(shape)

        # Prototyping is simplified by dynamically 
        # creating advected quantities as needed.
        self.quantities = quantities
        for q in quantities:
            setattr(self, q, np.zeros(shape))

        self.indices = np.indices(shape)
        self.velocity = np.zeros((self.dimensions, *shape))

        laplacian = operator(shape, difference(2, pressure_order))

        self.pressure_solver = factorized(laplacian)
        self.i=identity(250000)
        # print(type(laplacian))
        # print(type(self.i))
        # print(type(self.pressure_solver))
        # print(laplacian.shape)
        # print(self.i)
        self.vis_solver = factorized(self.i-self.mu*laplacian)
        #z return a func to solve sparse linear system。Ax=b  factorized(A)
        
        self.advect_order = advect_order
        #exit()

    def step(self):
        # Advection is computed backwards in time as described in Stable Fluids.
        advection_map = self.indices - self.velocity
        #z 回溯网格中心的粒子上一个step之前所在的位置，然后插值得到之前的速度



        # SciPy's spline filter introduces checkerboard divergence.
        # A linear blend of the filtered and unfiltered fields based
        # on some value epsilon eliminates this error.
        def advect(field, filter_epsilon=10e-2, mode='constant'):
            filtered = spline_filter(field, order=self.advect_order, mode=mode)
            field = filtered * (1 - filter_epsilon) + field * filter_epsilon
            return map_coordinates(field, advection_map, prefilter=False, order=self.advect_order, mode=mode)

        # Apply advection to each axis of the
        # velocity field and each user-defined quantity.
        for d in range(self.dimensions):
            self.velocity[d] = advect(self.velocity[d])
        #z  w2(x)=w1(p(x,t-dt)),advect


        for q in self.quantities:
            setattr(self, q, advect(getattr(self, q)))
     
        # Compute the jacobian at each point in the
        # velocity field to extract curl and divergence.
        jacobian_shape = (self.dimensions,) * 2
        partials = tuple(np.gradient(d) for d in self.velocity)
        jacobian = np.stack(partials).reshape(*jacobian_shape, *self.shape)

        divergence = jacobian.trace()
        #z trace of jacobian matrix(which consists of the partials of vel)

        # If this curl calculation is extended to 3D, the y-axis value must be negated.
        # This corresponds to the coefficients of the levi-civita symbol in that dimension.
        # Higher dimensions do not have a vector -> scalar, or vector -> vector,
        # correspondence between velocity and curl due to differing isomorphisms
        # between exterior powers in dimensions != 2 or 3 respectively.
        curl_mask = np.triu(np.ones(jacobian_shape, dtype=bool), k=1)
        curl = (jacobian[curl_mask] - jacobian[curl_mask.T]).squeeze()

        if(self.bvis):
            for d in range(self.dimensions):
                self.velocity[d] = self.vis_solver(self.velocity[d].flatten()).reshape(self.shape)

        # Apply the pressure correction to the fluid's velocity field.        
        #z  projection  solve a sparse linear system
        #z  laplacian P = Divergence  ,  nabla^2 p = div w3
        pressure = self.pressure_solver(divergence.flatten()).reshape(self.shape)
        self.velocity -= np.gradient(pressure)

        return divergence, curl, pressure
