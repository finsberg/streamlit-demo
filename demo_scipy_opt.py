import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import optimize
import matplotlib.cm as cm
from textwrap import dedent


def contour_plots(x_history):
    xmin, xmax = -2, 2
    ymin, ymax = -2, 2
    xx = np.linspace(xmin, xmax, 100)
    yy = np.linspace(ymin, ymax, 100)
    X, Y = np.meshgrid(xx, yy)
    with np.nditer([X, Y, None]) as it:
        for xi, yi, zi in it:
            zi[...] = np.log(optimize.rosen(np.array([xi, yi])))
        Z = it.operands[2]

    fig, ax = plt.subplots()
    im = ax.contourf(X, Y, Z, cmap=cm.jet, extent=(xmin, xmax, ymin, ymax))

    xhist = np.stack(x_history, axis=0)
    ax.plot(xhist[:, 0], xhist[:, 1], "-xk")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    fig.colorbar(im)
    return fig


def plot_cost_function(f_history, df_history):
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].semilogy(f_history)
    ax[0].grid(True)
    ax[0].set_xlabel(r"$k$")
    ax[0].set_ylabel(r"$f(x_k)$")

    ax[1].semilogy(df_history)
    ax[1].grid(True)
    ax[1].set_xlabel(r"$k$")
    ax[1].set_ylabel(r"$df(x_k)$")
    return fig


def optimization_page():
    st.header("Optimization of the rosenbrock function")

    st.markdown(
        dedent(
            """
            Here you can try different methods to minimimze the Rosenbrock function

            $$
            f(x, y) = 100(x^2 - y)^2 + (1-x)^2
            $$

            """
        )
    )
    cols = st.columns(2)
    with cols[0]:
        x00 = st.number_input("Start x1", value=0.5)
        include_jac = st.checkbox("Include gradient info")

    with cols[1]:
        x01 = st.number_input("Start x1", value=-1)
        include_hess = st.checkbox("Include hessian info")

    all_methdos = [
        "Nelder-Mead",
        "Powell",
        "CG",
        "BFGS",
        "L-BFGS-B",
        "TNC",
        "COBYLA",
        "SLSQP",
        "trust-constr",
    ]
    kwargs = {}
    if include_jac:
        kwargs["jac"] = optimize.rosen_der
        all_methdos += [
            "Newton-CG",
            "trust-ncg",
            "dogleg",
            "trust-krylov",
        ]
        if include_hess:
            kwargs["hess"] = optimize.rosen_hess
            all_methdos += ["trust-exact"]

    method = st.selectbox(
        "Optimzation method",
        all_methdos,
    )
    x0 = np.array([x00, x01])

    fun = optimize.rosen
    dfun = optimize.rosen_der

    x_history = [x0]
    f_history = [fun(x0)]
    df_history = [dfun(x0)]

    def cb(xk, *args):
        x_history.append(xk)
        f_history.append(fun(xk))
        df_history.append(dfun(xk))

    if st.button("Run"):
        res = optimize.minimize(fun, x0, method=method, callback=cb, **kwargs)
        st.write(f"Final f={res.fun}, x={res.x}")
        if res.success:
            st.success("Optimization was successfully")
        else:
            st.warning(res.message)
            return

        st.write(f"Number of iterations: {res.nit}")
        st.write(f"Number of function evaluation: {res.nfev}")

        fig = plot_cost_function(f_history, df_history)
        st.pyplot(fig)

        fig = contour_plots(x_history=x_history)
        st.pyplot(fig)


def about():
    st.header("About")
    st.markdown(
        """
    This page is to illustrate how to make a second page
    """
    )


st.set_page_config(page_title="Optization demo")

# Sidebar settings
pages = {
    "Scipy optimize": optimization_page,
    "About": about,
}

st.sidebar.title("Streamlit optimization demo")

# Radio buttons to select desired option
page = st.sidebar.radio("", tuple(pages.keys()))

pages[page]()

# About
st.sidebar.markdown(
    """
- [Source code](#)
""",
)
