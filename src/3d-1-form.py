from dolfin import *
import numpy
import sympy
import sympy_interface as S
import tabulate
parameters["form_compiler"]["cpp_optimize"] = True
set_log_level(ERROR)

### utils #################################################################
def my_mixed_function_space(Vs):
    """ My convenient handler for mixed function space. """ 
    M = MixedFunctionSpace(Vs)
    def setter(u, ui):
        # impure setter
        for i in range(len(Vs)):
            assign(u.sub(i), interpolate(ui[i], Vs[i]))
        return u
    def getter(u):
        ui = u.split()
        vi = []
        for i in range(len(Vs)):
            vi.append(interpolate(ui[i], Vs[i]))
        return vi
    return (M, setter, getter)

def compute_error(u_h, u_ext, V):
    u_h_V = interpolate(u_h, V)
    u_ext_V = interpolate(u_ext, V)
    e = Function(V)
    e.vector()[:] = u_h_V.vector() - u_ext_V.vector()
    return e

def calc_rate(hs, data):
    """ Compute the rate of converge by tabulating the successive slopes."""
    hs = numpy.array(hs)
    data = numpy.array(data)
    tmp = numpy.diff(numpy.log(data))/numpy.diff(numpy.log(hs))
    rate = numpy.zeros(data.size)
    rate[1:] = tmp
    return rate

def print_conv_rate_table(hs, ds, names, h_alt = None):
    """ Convergence rate printer. """
    # formatters
    h_fmt    = lambda x: "{0:.4f}".format(x)
    data_fmt = ".4e" # tabulate converts numeric str to number
                     # so this has to be set at the end
    rate_fmt = lambda x: "{0:.2f}".format(x) if x != 0 else ""
    # make table
    if h_alt is None:
        table = [map(h_fmt, hs)]
    else:
        table = [h_alt]
    header = [names[0]]
    for i in range(len(ds)):
        table.append(ds[i])
        table.append(map(rate_fmt, calc_rate(hs, ds[i])))
        header.append(names[i + 1])
        header.append("rate")
    table = zip(*table)
    s = tabulate.tabulate(table, headers = header, floatfmt = data_fmt)
    return s

def random_perturb_mesh(mesh, percentage,
                        deep_copy = True, preserve_boundary = True):
    """
    Randomly perturb a mesh.
    Input
          mesh              - input mesh
          percentage        - maximum amount of the perturbation as a 
                              percentage of hmin
          deep_copy         - whether to copy the mesh before perturbing it
          preserve_boundary - whether to move the vertices on the boundary
    Output
          rmesh             - the perturbed mesh
    """
    # Preparation
    if percentage == 0.0:
        return mesh
    if deep_copy:
        rmesh = Mesh(mesh)
    else:
        rmesh = mesh
    h = rmesh.hmin()
    # Perturb
    xs = rmesh.coordinates()
    dx = numpy.random.rand(*(xs.shape)) * percentage * h
    # Preserve the boundary
    if preserve_boundary:
        boundary_mesh = BoundaryMesh(rmesh, "exterior")
        bv = boundary_mesh.entity_map(0).array()
        dx[bv] = 0.0
    # Move
    rmesh.coordinates()[:] = xs + dx
    return rmesh

### 3d-1-form #############################################################
def make_spaces(pair_name, degree, mesh):
    if pair_name == '++':
        V   = FunctionSpace(mesh, "CG", degree + 1)
        Vb  = FunctionSpace(mesh, "CG", degree + 2)
        DVb = VectorFunctionSpace(mesh, "DG", degree + 1) 
        W   = FunctionSpace(mesh, "N2curl", degree)
        Wb  = FunctionSpace(mesh, "N2curl", degree + 1)
        DWb = FunctionSpace(mesh, "BDM", degree)
    else:
        V   = FunctionSpace(mesh, "CG", degree)
        Vb  = FunctionSpace(mesh, "CG", degree + 1)
        DVb = VectorFunctionSpace(mesh, "DG", degree)         
        W   = FunctionSpace(mesh, "N1curl", degree)
        Wb  = FunctionSpace(mesh, "N1curl", degree + 1)
        DWb = FunctionSpace(mesh, "RT", degree + 1)
    return (V, W, Vb, Wb, DVb, DWb)

def make_data(switch):
    u = sympy.Matrix(sympy.sympify(
        """
        (sin(pi*x)*cos(pi*z),
         cos(pi*x)*sin(pi*y),
         cos(pi*y)*sin(pi*z))
        """))
    if switch[0] == "T":
        l1 = sympy.Matrix(sympy.sympify(
            """
            ((sin(pi*y)*sin(pi*z), sin(pi*z), sin(pi*y)),
             (sin(pi*z), sin(pi*x)*sin(pi*z), sin(pi*x)),
             (sin(pi*y), sin(pi*x), sin(pi*x)*sin(pi*y)))
            """))
    else:
        l1 = sympy.zeros(3, 3)
    if switch[1] == "T":
        l2 = sympy.Matrix(sympy.sympify("(2, 0, 1)"))
    else:
        l2 = sympy.zeros(3, 1)
    if switch[2] == "T":
        l3 = sympy.Matrix(sympy.sympify("(3, 2, -1)"))
    else:
        l3 = sympy.zeros(3, 1)
    if switch[3] == "T":
        l4 = sympy.Matrix(sympy.sympify(
            """
            (( 1,  2, -1),
             ( 3, -3, -3),
             ( 1,  3,  1))
            """))
    else:
        l4 = sympy.zeros(3, 3)
    if switch[4] == "T":
        # l5 = sympy.Matrix(sympy.sympify(
        #     """
        #     (( 10, 0,  0),
        #      ( 0,  10, 0),
        #      ( 0,  0,  0))
        #     """))
        l5 = sympy.Matrix(sympy.sympify(
            """
            ((10,  0,  0),
             ( 0, 10,  0),
             ( 0,  0, 10))
            """))        
    else:
        l5 = sympy.zeros(3, 3)
    # compute data in sympy
    du = S.curl(u)
    sigma = -S.div(u) + l2.dot(u)
    dsigma = S.grad(sigma)
    f = dsigma + l3 * sigma + S.curl(du) + l4 * du + S.curl(l1 * u) + l5 * u
    # convert to FEniCS
    f = Expression(S.sympy2exp(f))
    ext_sols = map(Expression, map(S.sympy2exp, (sigma, dsigma, u, du)))
    lots = map(Expression, map(S.sympy2exp, (l1, l2, l3, l4, l5)))
    return (f, lots, ext_sols)

def solve_1_laplacian(pair_name, mesh, degree, f, lots, ext_sols):
    """ Solve the 1-form Laplacian with lower-order terms l1--l5 and
        right-hand side f using the pair of spaces given by pair_name.
        Then compute the error using the given exact solution.
    """
    # solve
    (l1, l2, l3, l4, l5) = lots
    (V, W, Vb, Wb, DVb, DWb) = make_spaces(pair_name, degree, mesh)
    (M, setter, getter) = my_mixed_function_space([V, W])
    (sigma, u) = TrialFunctions(M)
    (tau,   v) = TestFunctions(M)
    lhs = (sigma * tau - dot(u, grad(tau)) - dot(l2, u) * tau \
           + dot(grad(sigma),v) + dot(l3, v) * sigma \
           + dot(curl(u), curl(v)) + dot(dot(l1, u), curl(v)) \
           + dot(dot(l4, curl(u)), v) + dot(dot(l5, u), v)) * dx
    rhs = dot(f, v) * dx
    A, b = assemble_system(lhs, rhs)
    solver = PETScLUSolver('mumps')
    solver.set_operator(A)
    m = Function(M)
    solver.solve(m.vector(), b)
    (sigma_h, u_h) = getter(m)
    # compute errors
    (sigma_ext, dsigma_ext, u_ext, du_ext) = ext_sols
    error   = compute_error(sigma_h, sigma_ext, Vb)
    esigma  = numpy.sqrt(assemble(inner(error, error) * dx))
    error   = compute_error(project(grad(sigma_h), DVb), dsigma_ext, DVb)
    edsigma = numpy.sqrt(assemble(inner(error, error) * dx))
    error   = compute_error(u_h, u_ext, Wb)
    eu      = numpy.sqrt(assemble(inner(error, error) * dx))
    error   = compute_error(project(curl(u_h), DWb), du_ext, DWb)
    edu     = numpy.sqrt(assemble(inner(error, error) * dx))
    return (esigma, edsigma, eu, edu)

def exp(pair_name, degree, switch, meshes):
    (f, lots, ext_sols) = make_data(switch)
    hs = []; esigmas = []; edsigmas = []; eus = []; edus = []
    for mesh in meshes:
        h = mesh.hmin()
        (esigma, edsigma, eu, edu) = solve_1_laplacian(
            pair_name, mesh, degree, f, lots, ext_sols)
        hs.append(h)
        esigmas.append(esigma)
        edsigmas.append(edsigma)
        eus.append(eu)
        edus.append(edu)
    return (hs, [esigmas, edsigmas, eus, edus])

def mesh_maker(m):
    mesh = UnitCubeMesh(m, m, m)
    mesh = random_perturb_mesh(mesh, 0.20, deep_copy = True,
                               preserve_boundary = True)
    return mesh

pairs = ["++", "+-"]
degrees = [1]
switches = ["FTFFF"]#["FFFFF", "TFFFF", "FTFFF", "FFTFF", "FFFTF", "FFFFT"]
for degree in degrees:
    for pair_name in pairs:
        for switch in switches:
            ms = [2, 4, 8]
            meshes = map(mesh_maker, ms)
            (hs, es) = exp(pair_name, degree, switch, meshes)
            print("[pair: " + pair_name + "   deg: " + str(degree)
                  + "   lots: " + switch + "]")            
            print(print_conv_rate_table(hs, es,
                                        ["h", "sigma", "dsigma", "u", "du"],
                                        h_alt = ms))
            print("")
