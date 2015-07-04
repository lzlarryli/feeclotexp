import sympy

def sympy2exp(exp):
    x, y, z = sympy.symbols('x[0] x[1] x[2]')    
    def to_ccode(f):
        f = f.subs('x', x).subs('y', y).subs('z', z)
        raw = sympy.printing.ccode(f)
        return raw.replace("M_PI", "pi")
    if hasattr(exp, "__getitem__"):
        if exp.shape[0] == 1 or exp.shape[1] == 1:
            return tuple(map(to_ccode, exp))
        else:
            return tuple([tuple(map(to_ccode, exp[i, :]))
                          for i in range(exp.shape[1])])
    else:
        return to_ccode(exp)

def grad(u, dim = 3):
    if dim == 1:
        return sympy.Matrix([u.diff('x')])
    elif dim == 2:
        return sympy.Matrix([u.diff('x'), u.diff('y')])
    elif dim == 3:
        return sympy.Matrix([u.diff('x'), u.diff('y'), u.diff('z')])

def curl(u):
    if hasattr(u, "__getitem__"):
        # 3D vector curl
        return sympy.Matrix([u[2].diff('y') - u[1].diff('z'),
                             u[0].diff('z') - u[2].diff('x'),
                             u[1].diff('x') - u[0].diff('y')])
    else:
        # 2D rotated gradient
        return sympy.Matrix([u.diff('y'), -u.diff('x')])

def rot(u):
    # 2d rot
    return u[1].diff('x') - u[0].diff('y')

def div(u):
    if u.shape[0] == 2:
        return u[0].diff('x') + u[1].diff('y')
    elif u.shape[0] == 3:
        return u[0].diff('x') + u[1].diff('y') + u[2].diff('z')

