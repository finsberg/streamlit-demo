import streamlit as st

try:
    import ufl
except ImportError:
    pass


@st.cache(
    show_spinner=True,
    allow_output_mutation=True,
    hash_funcs={
        ufl.measure.Measure: lambda x: 1
    },  # Don't really know why we need this?
)
def solve(E, mu):
    import dolfin
    import fenics_plotly

    # Optimization options for the form compiler
    dolfin.parameters["form_compiler"]["cpp_optimize"] = True
    ffc_options = {
        "optimize": True,
        "eliminate_zeros": True,
        "precompute_basis_const": True,
        "precompute_ip_const": True,
    }

    # Create mesh and define function space
    mesh = dolfin.UnitCubeMesh(24, 16, 16)
    V = dolfin.VectorFunctionSpace(mesh, "Lagrange", 1)

    # Mark boundary subdomians
    left = dolfin.CompiledSubDomain("near(x[0], side) && on_boundary", side=0.0)
    right = dolfin.CompiledSubDomain("near(x[0], side) && on_boundary", side=1.0)

    # Define Dirichlet boundary (x = 0 or x = 1)
    c = dolfin.Expression(("0.0", "0.0", "0.0"), degree=1)
    r = dolfin.Expression(
        (
            "scale*0.0",
            "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
            "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])",
        ),
        degree=3,
        scale=0.5,
        y0=0.5,
        z0=0.5,
        theta=dolfin.pi / 3,
    )

    bcl = dolfin.DirichletBC(V, c, left)
    bcr = dolfin.DirichletBC(V, r, right)
    bcs = [bcl, bcr]

    # Define functions
    du = dolfin.TrialFunction(V)  # Incremental displacement
    v = dolfin.TestFunction(V)  # Test function
    u = dolfin.Function(V)  # Displacement from previous iteration
    B = dolfin.Constant((0.0, -0.5, 0.0))  # Body force per unit volume
    T = dolfin.Constant((0.1, 0.0, 0.0))  # Traction force on the boundary

    # Kinematics
    I = dolfin.Identity(3)  # Identity tensor
    F = I + dolfin.grad(u)  # Deformation gradient
    C = F.T * F  # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = dolfin.tr(C)
    J = dolfin.det(F)

    # Elasticity parameters
    E, nu = 10.0, 0.3
    mu, lmbda = dolfin.Constant(E / (2 * (1 + nu))), dolfin.Constant(
        E * nu / ((1 + nu) * (1 - 2 * nu))
    )

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu / 2) * (Ic - 3) - mu * dolfin.ln(J) + (lmbda / 2) * (dolfin.ln(J)) ** 2

    # Total potential energy
    Pi = psi * dolfin.dx - dolfin.dot(B, u) * dolfin.dx - dolfin.dot(T, u) * dolfin.ds

    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = dolfin.derivative(Pi, u, v)

    # Compute Jacobian of F
    J = dolfin.derivative(F, u, du)

    # Solve variational problem
    dolfin.solve(F == 0, u, bcs, J=J, form_compiler_parameters=ffc_options)

    # Plot and hold solution

    moved_mesh = dolfin.Mesh(mesh)
    dolfin.ALE.move(mesh, u)
    new_fig = fenics_plotly.plot(mesh, show=False)
    fig = fenics_plotly.plot(moved_mesh, show=False)
    return fig.figure, new_fig.figure


def fenics_page():
    st.header("FEniCS - hyperelasticity examlpe")

    st.markdown(
        """
    This example is based on the example in from the FEniCS demos
    https://fenicsproject.org/olddocs/dolfin/dev/python/demos/hyperelasticity/demo_hyperelasticity.py.html
    """
    )

    try:
        import dolfin
    except ImportError:
        st.error(
            "This demo requires 'dolfin' to be installed, but 'dolfin' cannot be found."
        )
        return

    try:
        import fenics_plotly
    except ImportError:
        st.error(
            "This demo requires 'fenics_plotly' to be installed - python -m pip install fenics_plotly"
        )
        return

    cols = st.columns(2)
    with cols[0]:
        E = st.number_input("E", value=10.0, help="Youngs module")

    with cols[1]:
        nu = st.number_input("nu", value=0.3, help="Poisson ratio")

    if st.button("Compute"):
        fig, new_fig = solve(E, nu)

        st.subheader("Deformed mesh")
        st.plotly_chart(new_fig)

        st.subheader("Original mesh")
        st.plotly_chart(fig)


st.set_page_config(page_title="FEniCS demo")
st.sidebar.title("Streamlit FEniCS demo")
fenics_page()
# About
st.sidebar.markdown(
    """
- [Source code](https://github.com/finsberg/streamlit-demo)
""",
)
