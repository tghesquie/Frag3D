import numpy as np
import sympy as sp
import akantu as aka
import h5py
import time
import plotly.graph_objects as go
import plotly.express as px
import os, shutil, sys, argparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs, eigsh
import scipy.sparse as sp


def lump_mass_matrix(M):
    if sp.issparse(M):
        row_sum = np.array(M.sum(axis=1)).ravel()  # (n,)
        M_lumped = sp.diags(row_sum, format=M.format)
    else:
        row_sum = np.sum(M, axis=1)  # (n,)
        M_lumped = np.diag(row_sum)
    return M_lumped


def compute_stable_timestep_bulk(model):
    """
    Computes the stable time step for a given model and mesh.
    """
    model.assembleStiffnessMatrix()
    model.assembleMass()
    dof_manager = model.getDOFManager()

    K = csc_matrix(aka.AkantuSparseMatrix(dof_manager.getMatrix("K")))
    M = csc_matrix(aka.AkantuSparseMatrix(dof_manager.getMatrix("M")))
    M = lump_mass_matrix(M)

    # Find the smallest eigenvalue of the generalized eigenvalue problem
    eigvals, _ = eigsh(A=K, k=1, M=M, which="LM")  # largest magnitude eigenvalue

    lambda_max = eigvals[0]  # hermitian, so real
    omega_max = np.sqrt(lambda_max)
    dt_crit_bulk = 2.0 / omega_max

    print(f"Bulk critical Δt = {dt_crit_bulk:.3e}")

    return dt_crit_bulk


def compute_stable_timestep(model, mesh, contact_penalty):
    """
    Returns the conservative Δt bound using bulk stiffness K plus contact penalties, with Gershgorin's theorem.

    model       : your Akantu model
    facet_conn  : (n_facets×2) array of node‐indices for every potential contact edge
    coords      : (n_nodes×2) array of nodal coordinates
    k_penalty   : penalty coefficient [N/m^3]
    """
    facet_conn = mesh.getConnectivities()(aka._tetrahedron_4)
    coords = mesh.getNodes()

    # --- assemble bulk K, lump M ---
    model.assembleStiffnessMatrix()
    model.assembleMass()
    dof_manager = model.getDOFManager()

    K = csc_matrix(aka.AkantuSparseMatrix(dof_manager.getMatrix("K")))
    M = csc_matrix(aka.AkantuSparseMatrix(dof_manager.getMatrix("M")))
    M = lump_mass_matrix(M)

    # 1) bulk Gershgorin row-sums in DOF-space
    row_sums = np.abs(K).sum(axis=1).A1  # shape = (n_dofs,)

    # 2) lumped masses per DOF
    masses = M.diagonal()  # shape = (n_dofs,)

    # 3) build nodal "area" from facets
    n_nodes = coords.shape[0]
    A_contact = np.zeros(n_nodes)
    for i, j in facet_conn:
        L = np.linalg.norm(coords[i] - coords[j])
        A_contact[i] += 0.5 * L
        A_contact[j] += 0.5 * L

    # 4) nodal penalty stiffness = k_penalty * A_contact
    Kc_nodal = contact_penalty * A_contact  # shape = (n_nodes,)

    # 5) expand to DOF-space by repeating each nodal entry for each DOF per node
    dofs_per_node = row_sums.size // n_nodes  # should be 2 in 2D
    Kc_dof = np.repeat(Kc_nodal, dofs_per_node)  # shape = (n_dofs,)

    # 6) total row-sums = bulk + contact
    row_sums_total = row_sums + Kc_dof

    # 7) Gershgorin bound and critical Δt
    lambda_max = np.max(row_sums_total / masses)
    dt_crit = 2.0 / np.sqrt(lambda_max)
    lambda_max_bulk = np.max(row_sums / masses)
    dt_crit_bulk = 2.0 / np.sqrt(lambda_max_bulk)

    print(f"Conservative critical Δt (bulk+contact): {dt_crit:.3e} s")
    print(f"Conservative critical Δt (bulk only): {dt_crit_bulk:.3e} s")
    print(f"Akantu critical Δt: {model.getStableTimeStep():.3e} s \n")

    return dt_crit, dt_crit_bulk


def compute_contact_penalty(
    model,
    mesh,
    safety_factor: float = 1.0,
    use_min_h: bool = False,
) -> float:
    """
    Estimates a contact penalty parameter based on actual element geometry.

    Parameters
    ----------
    model : Akantu model
        Your simulation model, from which we extract material properties.
    mesh : Akantu mesh
        Provides node coordinates and triangle connectivities.
    safety_factor : float, optional
        Multiplier to scale the reference penalty (default 1.0).
    use_min_h : bool, optional
        If True, use the smallest element characteristic length; otherwise use the mean.

    Returns
    -------
    penalty : float
        Reference contact penalty [N/m^3].
    """
    # 1. Material stiffness
    E = model.getMaterial(0).getReal("E")

    # 2. Mesh geometry
    coords = mesh.getNodes()  # (n_nodes, 2)
    conn = mesh.getConnectivities()(aka._triangle_3)  # (n_elems, 3)

    # 3. Compute each triangle's area using vector cross product in 2D
    tri_pts = coords[conn]  # (n_elems, 3, 2)
    v0 = tri_pts[:, 1] - tri_pts[:, 0]
    v1 = tri_pts[:, 2] - tri_pts[:, 0]
    # cross product z-component for each triangle
    cross_z = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    areas = 0.5 * np.abs(cross_z)  # (n_elems,)

    # 4. Characteristic length for each triangle (equilateral‐equivalent)
    h_elems = np.sqrt((4 * areas) / np.sqrt(3))  # (n_elems,)

    # 5. Choose representative h: mean or minimum
    h_ref = np.min(h_elems) if use_min_h else np.mean(h_elems)

    # 6. Penalty parameter
    penalty = safety_factor * E / h_ref

    print(f"Computed contact penalty: {penalty:.2e} N/m³ " f"(h_ref = {h_ref:.3e} m)")

    return penalty


def getStableTimestepCohesive(model, stable_damage=0.1):

    # Check stable_damage is between 0 and 1
    if stable_damage <= 0 or stable_damage > 1:
        raise ValueError("Stable damage must be > 0 and <=1")

    # Bulk time step
    time_step = model.getStableTimeStep()
    print("Bulk Stable Time Step: {:.2e}".format(time_step), "s")

    # Compute the smallest element size
    E = model.getMaterial(0).getReal("E")
    mu = model.getMaterial(0).getReal("mu")
    lambda_ = model.getMaterial(0).getReal("lambda")
    rho = model.getMaterial(0).getReal("rho")
    nu = model.getMaterial(0).getReal("nu")
    c = np.sqrt((lambda_ + 2 * mu) / rho)

    inradius = c * time_step
    print("Inradius: {:.2e}".format(inradius), "m")

    # Compute the maximum stable cohesive stiffness for a defined damage
    G_c = model.getMaterial(1).getReal("G_c")
    sigma_c = np.max(model.getMaterial(1).getInternalReal("sigma_c")(aka._triangle_3))
    k_coh_max = sigma_c**2 / (2 * G_c * stable_damage) * (1 - stable_damage)

    print("Maximum Cohesive Stiffness: {:.2e}".format(k_coh_max), "N/m")

    # Build test case:
    b = 2 * inradius * np.sqrt(3)
    h = 3 * inradius
    area = 0.5 * b * h
    x, y, b_sp, h_sp, k_coh, delta_t = sp.symbols("x y b h k_coh Delta_t")

    # T3 element 0 - strain/displacement matrix
    x0, y0, x1, y1, x2, y2 = -b / 2, 0, b / 2, 0, 0, h
    area0 = 0.5 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))

    B0 = sp.Matrix(
        [
            [y1 - y2, 0, y2 - y0, 0, y0 - y1, 0],
            [0, x2 - x1, 0, x0 - x2, 0, x1 - x0],
            [x2 - x1, y1 - y2, x0 - x2, y2 - y0, x1 - x0, y0 - y1],
        ]
    ) / (2 * area0)

    # T3 element 1 - strain/displacement matrix
    x3, y3, x4, y4, x5, y5 = -b / 2, 0, b / 2, 0, 0, -h
    area1 = 0.5 * (x3 * (y4 - y5) + x4 * (y5 - y3) + x5 * (y3 - y4))

    B1 = sp.Matrix(
        [
            [y4 - y5, 0, y5 - y3, 0, y3 - y4, 0],
            [0, x4 - x3, 0, x3 - x5, 0, x5 - x4],
            [x4 - x3, y4 - y5, x3 - x5, y5 - y3, x5 - x4, y3 - y4],
        ]
    ) / (2 * area1)

    # Constitutive matrix (assuming plane strain)
    D = (E / ((1 + nu) * (1 - 2 * nu))) * sp.Matrix(
        [[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]]
    )

    # Compute the local stiffness matrices
    K0 = area0 * B0.T * D * B0
    K1 = area1 * B1.T * D * B1

    # Compute the global stiffness matrix
    Kg = sp.zeros(12, 12)
    Kg[0:6, 0:6] = K0
    Kg[6:12, 6:12] = K1

    K_coh = sp.Matrix(
        [
            [k_coh_max, 0, 0, 0, -k_coh_max, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, k_coh_max, 0, 0, 0, -k_coh_max, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-k_coh_max, 0, 0, 0, k_coh_max, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -k_coh_max, 0, 0, 0, k_coh_max, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    indices = [0, 1, 2, 3, 6, 7, 10, 11]
    for i in range(len(indices)):
        for j in range(len(indices)):
            Kg[indices[i], indices[j]] += K_coh[i, j]

    # compute the local lumped mass matrices
    M0 = (rho * area0 / 3) * sp.eye(6)
    M1 = (rho * area1 / 3) * sp.eye(6)

    # compute the global lumped mass matrix
    Mg = sp.zeros(12, 12)
    Mg[0:6, 0:6] = M0
    Mg[6:12, 6:12] = M1
    Mg_inv = Mg.inv()

    # Create I the identity matrix
    I = sp.Matrix(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    # Assemble the U matrix
    U11 = (-(delta_t**2) / 2) * Mg_inv * Kg + I
    U12 = delta_t * I
    U21 = -delta_t * Mg_inv * Kg + (delta_t**3 / 4) * Mg_inv * Kg * Mg_inv * Kg
    U22 = (-(delta_t**2) / 2) * Mg_inv * Kg + I

    U = sp.simplify(sp.Matrix([[U11, U12], [U21, U22]]))

    dt_min, dt_max = 1e-6 * time_step, time_step
    tol = 1e-3 * time_step  # 1e-10

    # Perform binary search
    while dt_max - dt_min > tol:
        dt_mid = (dt_min + dt_max) / 2
        U_dt = np.array(U.subs({delta_t: dt_mid})).astype(np.float64)
        eigenvalues = np.linalg.eigvals(U_dt)
        # eigenvalues = U.subs({delta_t: dt_mid}).eigenvals()
        stable = np.all(eigenvalues <= 1.01)

        if stable:
            dt_min = dt_mid  # Search in the lower half
        else:
            dt_max = dt_mid  # Search in the upper half

        print("Current Cohesive Stable Time Step: {:.5e}".format(dt_mid), "s")
        print("Eigenvalues: ", eigenvalues)

    print("Initial Bulk Stable Time Step: {:.5e}".format(time_step), "s")
    print("Final Cohesive Stable Time Step: {:.5e}".format(dt_mid), "s")

    print("Ratio: ", ((dt_min + dt_max) / 2) / time_step)

    # Return the midpoint as the stable timestep
    return (dt_min + dt_max) / 2


def computeEnergy(model, cumulative_work, previous_work):
    epot = model.getEnergy("potential")
    ekin = model.getEnergy("kinetic")
    edis = model.getEnergy("dissipated")
    erev = model.getEnergy("reversible")
    econ = model.getEnergy("cohesive contact")
    # current_work = model.getEnergy("external work")
    # cumulative_work += (previous_work + current_work) / 2
    return epot + ekin + edis + erev + econ - cumulative_work


def saveCS(model, material_file, mesh_file, inpath):
    try:
        sigma_c_quads = model.getMaterial(1).getInternalReal("sigma_c")(aka._triangle_3)
    except:
        try:
            sigma_c_quads = model.getMaterial(1).getInternalReal("sigma_c")(
                aka._segment_3
            )
        except:
            print("Error: sigma_c not found")
            return
    cs_file = (
        inpath
        + "/input/cohesive_strength/"
        + material_file.split("/")[-1].split(".")[0]
        + "|"
        + mesh_file.split("/")[-1].replace(".msh", "")
        + ".csv"
    )

    with open(cs_file, "w") as f:
        for i in range(len(sigma_c_quads[:, 0])):
            f.write("{} \n".format(sigma_c_quads[i, 0]))
    f.close()


def loadCS(model, material_file, mesh_file, inpath):
    try:
        sigma_c_quads = model.getMaterial(1).getInternalReal("sigma_c")(aka._triangle_3)
    except:
        try:
            sigma_c_quads = model.getMaterial(1).getInternalReal("sigma_c")(
                aka._segment_3
            )
        except:
            print("Error: sigma_c not found")
            return
    cs_file = (
        inpath
        + "/input/cohesive_strength/"
        + material_file.split("/")[-1].split(".")[0]
        + "|"
        + mesh_file.split("/")[-1].replace(".msh", "")
        + ".csv"
    )

    data = np.loadtxt(cs_file, skiprows=0)
    sigma_c_quads[:, 0] = data


def initParaviewDumpers(model, outpath):

    model.setBaseName("tension")
    model.setDirectory(outpath)
    model.addDumpFieldVector("displacement")
    model.addDumpFieldVector("external_force")
    model.addDumpFieldVector("internal_force")
    model.addDumpField("velocity")
    model.addDumpField("grad_u")
    model.addDumpField("stress")

    model.setBaseNameToDumper("cohesive elements", "cohesive")
    model.addDumpFieldVectorToDumper("cohesive elements", "displacement")
    model.addDumpFieldToDumper("cohesive elements", "damage")
    model.addDumpFieldVectorToDumper("cohesive elements", "tractions")
    model.addDumpFieldVectorToDumper("cohesive elements", "opening")
    # model.addDumpFieldVectorToDumper("cohesive elements", "flag_unload_contact")


def initDumpers(model, filename_energy, filename_fragment):

    f_energy = open(filename_energy, "w")
    f_energy.write(
        "ID, Time, Potential Energy, Kinetic Energy, Dissipated Energy, Reversible Energy, Contact Energy, External Work, Total Energy, Fragments Number\n"
    )

    f_frag = open(filename_fragment, "w")
    f_frag.write("ID, Time, Fragments Number, Fragment Mass\n")

    return f_energy, f_frag


def dumpResultsH5(model, n, dt, cumulative_work, h5_file="../output/tmp/data.h5"):
    def get_energy_data():
        energy_data = {
            "epot": model.getEnergy("potential"),
            "ekin": model.getEnergy("kinetic"),
            "edis": model.getEnergy("dissipated"),
            "erev": model.getEnergy("reversible"),
            "econ": model.getEnergy("cohesive contact"),
            "work": cumulative_work,
        }
        energy_data["total_energy"] = (
            energy_data["epot"]
            + energy_data["ekin"]
            + energy_data["edis"]
            + energy_data["erev"]
            + energy_data["econ"]
            - cumulative_work
        )
        return energy_data

    def get_fragment_data():
        fragment_data = aka.FragmentManager(model)
        fragment_data.computeAllData()
        return {
            "nb_fragments": fragment_data.getNbFragment(),
            "fragment_mass": fragment_data.getMass()[:, 0].ravel(),
            "fragment_velocity": fragment_data.getVelocity()[:],
        }

    def get_damage_data():
        try:
            return model.getMaterial(1).getInternalReal("damage")(aka._cohesive_2d_6)
        except:
            try:
                return model.getMaterial(1).getInternalReal("damage")(
                    aka._cohesive_2d_4
                )
            except:
                return [0]

    def get_overshoot_data():
        try:
            return {
                "contact_overshoot": np.sum(
                    model.getMaterial(1).getInternalReal("contact_overshoot")(
                        aka._cohesive_2d_4
                    )
                ),
                "softening_overshoot": np.sum(
                    model.getMaterial(1).getInternalReal("softening_overshoot")(
                        aka._cohesive_2d_4
                    )
                ),
                "initial_unstable_state": np.sum(
                    model.getMaterial(1).getInternalReal("initial_unstable_state")(
                        aka._cohesive_2d_4
                    )
                ),
                "num_contact_overshoot": np.sum(
                    np.where(
                        model.getMaterial(1).getInternalReal("contact_overshoot")(
                            aka._cohesive_2d_4
                        )
                        != 0,
                        1,
                        0,
                    )
                ),
                "num_softening_overshoot": np.sum(
                    np.where(
                        model.getMaterial(1).getInternalReal("softening_overshoot")(
                            aka._cohesive_2d_4
                        )
                        != 0,
                        1,
                        0,
                    )
                ),
                "num_initial_unstable_state": np.sum(
                    np.where(
                        model.getMaterial(1).getInternalReal("initial_unstable_state")(
                            aka._cohesive_2d_4
                        )
                        != 0,
                        1,
                        0,
                    )
                ),
            }

        except Exception as e:
            print(f"Error retrieving overshoot data: {e}")
            return {
                "contact_overshoot": 0,
                "softening_overshoot": 0,
                "initial_unstable_state": 0,
                "num_contact_overshoot": 0,
                "num_softening_overshoot": 0,
                "num_initial_unstable_state": 0,
            }

    # Gather data
    energy_data = get_energy_data()
    fragment_data = get_fragment_data()
    # damage_data = get_damage_data()
    # overshoot_data = get_overshoot_data()

    # Save data to HDF5
    try:
        with h5py.File(h5_file, "a", libver="latest") as h5f:
            h5f.swmr_mode = True
            grp = h5f.create_group(f"step_{n}")
            grp.create_dataset("fragment_mass", data=fragment_data["fragment_mass"])
            grp.create_dataset(
                "fragment_velocity", data=fragment_data["fragment_velocity"]
            )
            attributes = {
                "nb_fragments": fragment_data["nb_fragments"],
                "time": n * dt,
                **energy_data,
                # **overshoot_data,
                # Additional attribute entries can be added here as needed
            }
            for key, value in attributes.items():
                grp.attrs[key] = value
            h5f.flush()  # Ensure data is flushed to disk
    except BlockingIOError:
        print("Oh Oh, file is locked, retrying in 0.1s")
        time.sleep(0.1)  # Wait before retrying


def initUniaxialVelocityField(mesh, model, strain_rate):

    velocity = model.getVelocity()
    nodes = mesh.getNodes()
    lower_bounds = mesh.getLowerBounds()
    upper_bounds = mesh.getUpperBounds()
    L = upper_bounds[0] - lower_bounds[0]
    vel_max = strain_rate * L / 2

    for v, pos in zip(velocity, nodes):
        v[0] = (vel_max / upper_bounds[0]) * pos[0]

    return vel_max


def initRadialVelocityField(mesh, model, strain_rate):

    velocity = model.getVelocity()
    nodes = mesh.getNodes()
    lower_bounds = mesh.getLowerBounds()
    upper_bounds = mesh.getUpperBounds()
    L = upper_bounds[0] - lower_bounds[0]
    vel_max = strain_rate * L / 2

    for v, pos in zip(velocity, nodes):
        v[0] = (vel_max / upper_bounds[0]) * pos[0]
        v[1] = (vel_max / upper_bounds[1]) * pos[1]

    return vel_max


def initRadialVelocityField3D(mesh, model, eps_v_dot, center=[0, 0, 0]):
    alpha = eps_v_dot / 3.0
    vel = model.getVelocity()
    nodes = mesh.getNodes()
    for v, x in zip(vel, nodes):
        xr = [x[0] - center[0], x[1] - center[1], x[2] - center[2]]
        v[0], v[1], v[2] = alpha * xr[0], alpha * xr[1], alpha * xr[2]
    return alpha


def initImpactVelocityField(
    mesh, model, v0, kappa, center=(0.0, 0.0), z_sign=1.0, cutoff=None
):
    """
    Apply a velocity field in z-direction, whose magnitude follows a Gaussian distribution centered at `center`.
    """
    vel = model.getVelocity()
    nodes = mesh.getNodes()

    lower_bounds = mesh.getLowerBounds()
    upper_bounds = mesh.getUpperBounds()
    L = upper_bounds[0] - lower_bounds[0]

    sigma = kappa * L / 2.0

    inv_two_sigma2 = 1.0 / (2.0 * sigma * sigma)
    cx, cy = center

    for v, x in zip(vel, nodes):
        dx = x[0] - cx
        dy = x[1] - cy
        r2 = dx * dx + dy * dy

        if cutoff is not None and r2 > cutoff * cutoff:
            vz = 0.0
        else:
            vz = z_sign * v0 * np.exp(-r2 * inv_two_sigma2)

        # enforce z-only impact
        v[0] = 0.0
        v[1] = 0.0
        v[2] = vz


def computeExternalWorkAKA(model, cumulative_work, previous_work, apply_bc):

    if apply_bc:
        current_work = model.getEnergy("external work")
        cumulative_work += (previous_work + current_work) / 2
        previous_work = current_work

        return cumulative_work, previous_work

    else:
        return cumulative_work, 0


def computeExternalWork(model, mesh, cumulative_work, previous_power, apply_bc, dt):
    if apply_bc:
        # Collect boundary condition nodes
        nodes_bc = []
        for bc in ["x0", "xf", "y0", "yf"]:
            # print(len(mesh.getElementGroup(bc).getNodeGroup().getNodes()))
            # if len(mesh.getElementGroup(bc).getNodeGroup().getNodes()) > 92:
            #    input("Press Enter to continue...")
            nodes_bc.extend(mesh.getElementGroup(bc).getNodeGroup().getNodes())

        # Ensure unique nodes
        unique_nodes_bc = np.unique(nodes_bc)

        # Get velocities and internal forces at the boundary nodes
        velocity_bc = model.getVelocity()[unique_nodes_bc]  # Shape: [nb_nodes, dim]
        internal_force = model.getInternalForce()[unique_nodes_bc] * (
            -1
        )  # Shape: [nb_nodes, dim]

        # Compute power (rate of work) at the current step
        current_power = np.sum(
            np.einsum("ij,ij->i", internal_force, velocity_bc)
        )  # Dot product for each node

        # Update cumulative work using trapezoidal rule
        cumulative_work += dt * 0.5 * (previous_power + current_power)
        # cumulative_work += previous_power * np.einsum("ij,dt

        return cumulative_work, current_power

    else:
        return cumulative_work, 0


class Radial(aka.DirichletFunctor):
    def __init__(self, inc, width):
        super().__init__()
        self.inc = inc
        self.width = width

    def __call__(self, node, flags, primal, coord):
        direction = np.array(coord)
        direction /= self.width / 2

        flags[0] = True
        flags[1] = True
        primal[0] = self.inc * direction[0]
        primal[1] = self.inc * direction[1]


class Free(aka.DirichletFunctor):
    def __init__(self):
        super().__init__()

    def __call__(self, node, flags, primal, coord):
        flags[:] = False


def applyRadialDisplacement(
    mesh, model, n, vel_max, dt, initial_displacement=0, static=False
):

    lower_bounds = mesh.getLowerBounds()
    upper_bounds = mesh.getUpperBounds()
    L = upper_bounds[0] - lower_bounds[0]

    if static:
        young_modulus = model.getMaterial(0).getReal("E")
        critical_stress = np.min(
            model.getMaterial(1).getInternalReal("sigma_c")(aka._triangle_3)
        )

        poisson_ratio = model.getMaterial(0).getReal("nu")
        static_displacement = (
            (L * (1 - poisson_ratio) * critical_stress / young_modulus)
            * 0.5
            * 1
            / np.sqrt(2)
        )
        radial = Radial(static_displacement, L)
        for edge in ["x0", "xf", "y0", "yf"]:
            model.applyBC(radial, edge)
        return static_displacement
    else:
        radial = Radial(((n + 1) * vel_max * dt) + initial_displacement, L)
        for edge in ["x0", "xf", "y0", "yf"]:
            model.applyBC(radial, edge)


def removeRadialDisplacement(mesh, model):

    free = Free()
    for edge in ["x0", "xf", "y0", "yf"]:
        model.applyBC(free, edge)


def createNodeGroups(mesh):
    try:
        lower_bounds = mesh.getLowerBounds()
        upper_bounds = mesh.getUpperBounds()
        nodes = mesh.getNodes()

        node_group_x0 = mesh.createNodeGroup("x0_node", False)
        node_group_xf = mesh.createNodeGroup("xf_node", False)
        node_group_y0 = mesh.createNodeGroup("y0_node", False)
        node_group_yf = mesh.createNodeGroup("yf_node", False)

        # Create masks for the nodes containing the id of the nodes at the boundaries
        mask_x0 = np.where(np.isclose(nodes[:, 0], lower_bounds[0]))[0]
        mask_xf = np.where(np.isclose(nodes[:, 0], upper_bounds[0]))[0]
        mask_y0 = np.where(np.isclose(nodes[:, 1], lower_bounds[1]))[0]
        mask_yf = np.where(np.isclose(nodes[:, 1], upper_bounds[1]))[0]

        for n in mask_x0:
            node_group_x0.add(n)
        for n in mask_xf:
            node_group_xf.add(n)
        for n in mask_y0:
            node_group_y0.add(n)
        for n in mask_yf:
            node_group_yf.add(n)

        mesh.createElementGroupFromNodeGroup("x0", "x0_node", 2)
        mesh.createElementGroupFromNodeGroup("xf", "xf_node", 2)
        mesh.createElementGroupFromNodeGroup("y0", "y0_node", 2)
        mesh.createElementGroupFromNodeGroup("yf", "yf_node", 2)
    except:
        print("Node groups already created.")


############################################################################################################
#                                        Visualization functions                                           #
############################################################################################################


def readH5Data(h5_file):
    data = {"times": []}

    # Open the HDF5 file
    with h5py.File(h5_file, "r") as h5f:
        for step in h5f.keys():
            grp = h5f[step]

            # Load all attributes dynamically
            step_data = {key: grp.attrs[key] for key in grp.attrs}

            # Load additional datasets if they exist
            optional_datasets = [
                "opening",
                "traction",
                "normal",
                "damage",
                "fragment_mass",
                "fragment_velocity",
            ]
            for dataset in optional_datasets:
                if dataset in grp:
                    step_data[dataset] = grp[dataset][:]

            # Append the time separately
            data["times"].append(step_data["time"])

            # Append the step data dictionary
            for key, value in step_data.items():
                if key not in data:
                    data[key] = []
                data[key].append(value)

    # Convert lists to NumPy arrays
    sorted_indices = np.argsort(data["times"])
    data = {
        key: (
            np.array(val)[sorted_indices]
            if isinstance(val[0], (int, float))
            else [val[i] for i in sorted_indices]
        )
        for key, val in data.items()
    }

    return data


def plotScatter(
    x_data,
    y_data,
    labels,
    widths=None,
    dashes=None,
    fills=None,
    mode="lines",
    colorlist=None,
    colorscale=None,
    legend_x=0.98,
    legend_y=0.97,
    type_x="linear",
    type_y="linear",
    range_x=None,
    range_y=None,
    w=400,
    h=400,
    title="",
    xaxis_title="",
    yaxis_title="",
    save=False,
    save_path="plot.png",
    show=True,
    showlegend=True,
):

    fig = go.Figure()

    # Check if colorlist is provided; otherwise, use default colors
    if colorlist:
        colors = colorlist
    elif colorscale:
        colors = px.colors.sequential.Plasma
    else:
        colors = px.colors.qualitative.Plotly

    for i, (x, y, label, width, dash, fill) in enumerate(
        zip(x_data, y_data, labels, widths, dashes, fills)
    ):
        if mode == "lines":
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode=mode,
                    name=label,
                    line=dict(color=colors[i % len(colors)], width=width, dash=dash),
                    fill=fill,
                )
            )
        elif mode == "markers":
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode=mode,
                    name=label,
                    marker=dict(color=colors[i % len(colors)], size=width),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=showlegend,
        template="plotly_white",
        xaxis=dict(
            showgrid=True,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            zeroline=False,
            tickformat=".1e",
            type=type_x,
            range=range_x,
        ),
        yaxis=dict(
            showgrid=True,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            zeroline=False,
            tickformat=".1e",
            type=type_y,
            range=range_y,
        ),
        legend=dict(
            x=legend_x,
            y=legend_y,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor="black",
            borderwidth=1,
        ),
        # paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)'
    )

    if not save:
        fig.update_layout(
            font=dict(size=12, color="Black"),  # Font size  # Font color
            width=600,
            height=600,
        )

    elif save:
        fig.update_layout(
            font=dict(
                family="Latin-Modern",  # Font family
                size=12,  # Font size
                color="Black",  # Font color
            ),
            width=w,
            height=h,
        )

        fig.write_image(save_path, scale=4)
        # fig.write_html(save_path)

    fig.show()


def plotHistogram(
    x_data,
    labels,
    nbins=10,
    opacity=1,
    colorlist=None,
    histnorm="probability",
    title="",
    xaxis_title="",
    yaxis_title="",
    save=False,
    save_path="plot.png",
    show=True,
    showlegend=True,
):

    fig = go.Figure()

    for i, (x, label) in enumerate(zip(x_data, labels)):
        fig.add_trace(
            go.Histogram(
                x=x, name=label, histnorm=histnorm, nbinsx=nbins, opacity=opacity
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=showlegend,
        template="plotly_white",
        xaxis=dict(
            showgrid=True,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            zeroline=False,
        ),
        legend=dict(
            x=0.98,
            y=0.97,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor="black",
            borderwidth=1,
        ),
    )

    if show:
        fig.update_layout(
            font=dict(size=12, color="Black"),  # Font size  # Font color
            width=600,
            height=600,
        )
        fig.show()

    if save:
        fig.update_layout(
            font=dict(
                family="Latin-Modern",  # Font family
                size=12,  # Font size
                color="Black",  # Font color
            ),
            width=400,
            height=400,
        )

        fig.write_image(save_path, scale=4)


def filterMass(fragment_mass, min, max):
    filtered_fragment_mass = []
    nb_fragments = np.zeros(len(fragment_mass))

    for i, f in enumerate(fragment_mass):
        filtered_fragment_mass.append(
            [f[j] for j in range(len(f)) if f[j] > min and f[j] < max]
        )
        nb_fragments[i] = len(filtered_fragment_mass[-1])

    return filtered_fragment_mass, nb_fragments


def parseArguments():
    parser = argparse.ArgumentParser(
        description="Run simulation with specified parameters."
    )
    parser.add_argument(
        "--material_file",
        "-mat",
        type=str,
        default="AD995_cohesive_contact_m10_stable.dat",
        help="Name of the material file.",
    )
    parser.add_argument(
        "--mesh_file",
        "-msh",
        type=str,
        default="plate_0.01x0.01_npz1_P1.msh",
        help="Name of the mesh file.",
    )
    parser.add_argument(
        "--strain_rate",
        "-sr",
        type=float,
        default=2.559e4,
        help="Strain rate for the simulation.",
    )
    parser.add_argument(
        "--velocity",
        "-v",
        type=float,
        default=10.0,
        help="Initial velocity for the impact.",
    )
    parser.add_argument(
        "--safety_factor",
        "-t",
        type=float,
        default=0.2,
        help="Safety factor for the simulation.",
    )
    parser.add_argument(
        "--damping_type",
        "-dt",
        choices=["uniform", "weighted", "weighted_tanh"],
        default=None,
        help="Type of damping to apply on velocities.",
    )
    parser.add_argument(
        "--energy_reference",
        "-er",
        choices=["initial", "previous"],
        default="initial",
        help="Reference energy state for damping calculations.",
    )
    parser.add_argument(
        "--add_previous_weights",
        "-apw",
        action="store_true",
        help="Flag to add previous weights when damping velocities. Effective only if damping is specified.",
    )
    parser.add_argument(
        "--damp_unstable_zone",
        "-duz",
        action="store_true",
        help="Flag to enable damping of the unstable zone.",
    )
    parser.add_argument(
        "--zeta",
        "-z",
        type=float,
        default=0.02,
        help="Damping coefficient for the unstable zone.",
    )
    parser.add_argument(
        "--contact",
        "-c",
        action="store_false",
        default=True,
        help="Flag to enable contact penalty.",
    )
    parser.add_argument(
        "--unit", "-u", action="store_true", help="Switch simulation to unit test case"
    )
    parser.add_argument(
        "--profile",
        "-p",
        action="store_true",
        help="Flag to enable profiling of the code.",
    )
    args = parser.parse_args()
    # Additional logic to check if damping is actually enabled
    if args.damping_type is None:
        # Disable any damping-related effects if no damping type is specified
        args.energy_reference = None
        args.add_previous_weights = False

    return args


def setupDir(nname, args):
    inpath = os.path.dirname(os.path.abspath(__file__)) + "/../"
    if nname == "lsmspc19":
        outpath = inpath + "output/local/clip/"
        if args.unit:
            outpath += "unit_test/"
        else:
            outpath += "test_npz4"
            # outpath += f"strain_rate_{args.strain_rate:.1e}_safety_factor_{args.safety_factor:.1e}_1e17_npz1_1e-5_loop"  # _1e17_npz1_1e-5"
            if args.damping_type != None:
                outpath += f"_dv_{args.damping_type}_{args.energy_reference}"
                if args.add_previous_weights:
                    outpath += "_apw"
                outpath += f"_limit-1e-6"
            outpath += "/"

    else:
        outpath = "/scratch/ghesquie/Frag3D/"
        outpath += f"impact_vel_{args.velocity:.1e}_safety_factor_{args.safety_factor:.1e}_time{args.time:.1e}/"

    # Clear and recreate the output directory
    if os.path.exists(outpath):
        try:
            shutil.rmtree(outpath)
        except Exception as e:
            print(f"Error occurred while clearing the directory: {e}")
            sys.exit(1)
    os.makedirs(outpath)

    return inpath, outpath
