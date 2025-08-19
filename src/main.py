import akantu as aka
import numpy as np
import os
from helper_functions import *

TIME = 4e-6


def runSimulation(material_file, mesh_file, paths, args, free_edges=False):

    # Parse the material file to Akantu
    aka.parseInput(material_file)

    # Create the mesh object and read the mesh file
    spatial_dimension = 3
    mesh = aka.Mesh(spatial_dimension)
    mesh.read(mesh_file)

    # Specify and initialize both static and dynamic models
    model = aka.SolidMechanicsModelCohesive(mesh)
    model.initFull(_analysis_method=aka._static, _is_extrinsic=True)
    model.initNewSolver(aka._explicit_lumped_mass)

    # Set the cohesive strength of the element facets. This is done for reproducibility and comparison between different simulations with the same material file.
    try:
        loadCS(model, material_file, mesh_file, paths[0])
    except:
        saveCS(model, material_file, mesh_file, paths[0])
        loadCS(model, material_file, mesh_file, paths[0])

    # Initialize the dumpers
    initParaviewDumpers(model, paths[1])

    ## STATIC SOLVE ##

    # Apply the boundary conditions
    # static_displacement = applyRadialDisplacement(mesh, model, 0, 0, TIME, static=True)
    static_displacement = 0
    #
    ## Solve the static step
    # model.solveStep("static")
    # model.dump()
    # removeRadialDisplacement(mesh, model)

    ## DYNAMIC SOLVE ##
    # Set the contact penalty
    # compute and assign contact penalty (uses its own safety_factor=10.0)
    # penalty = compute_contact_penalty(model, mesh, safety_factor=10.0, use_min_h=False)
    penalty = 1
    # model.getMaterial(1).setReal("penalty", penalty)

    # Initialize the interpolation functions to compute the cohesive stress at facet level
    model.updateAutomaticInsertion()

    # Set the time step and compute the number of steps
    # dt_crit, dt_crit_bulk = compute_stable_timestep(model, mesh, penalty)
    dt_crit = model.getStableTimeStep()
    time_step = dt_crit * args.safety_factor
    model.setTimeStep(time_step)
    n_steps = int(TIME / time_step)
    print(
        "Time step duration: {:.2e}".format(time_step), "s | Number of steps: ", n_steps
    )

    # Initialize the velocity field
    v = initRadialVelocityField3D(mesh, model, args.strain_rate)

    # Get the stable damage and stable stiffness to check the stability of the simulation
    # initUnstableZoneDamping(model, time_step, None)

    # Initialize necessary variables and apply the boundary conditions
    print("Simulation starting...")
    nb_inserted, cumulative_work, previous_work = 0, 0, 0
    apply_bc = False
    epot_max = -np.inf

    for n in range(n_steps):

        # Boundary Conditions
        if free_edges:
            if apply_bc == True:
                epot = model.getEnergy("potential")
                epot_max = max(epot, epot_max)
                # if epot < 0.99 * epot_max:
                if nb_inserted > 0:
                    removeRadialDisplacement(mesh, model)
                    print("Displacement removed.")
                    apply_bc = False
                else:
                    applyRadialDisplacement(
                        mesh, model, n, v, time_step, static_displacement
                    )
        else:
            applyRadialDisplacement(mesh, model, n, v, time_step)

        # Solve the step and check the cohesive stress
        model.solveStep()
        nb_inserted += model.checkCohesiveStress()
        # model.dump()
        # Compute the external work (integration not done by Akantu)
        cumulative_work, previous_work = computeExternalWork(
            model, mesh, cumulative_work, previous_work, apply_bc, time_step
        )

        # Dump the results n_dump times during the simulation for visualization
        n_dump_paraview = 1000  # n_steps
        n_dump_h5 = 1000

        if n % (int(n_steps / n_dump_paraview)) == 0:
            model.dump()
            model.dump("cohesive elements")

        if n % (int(n_steps / n_dump_h5)) == 0:
            dumpResultsH5(
                model, n, time_step, cumulative_work, h5_file=f"{paths[1]}data.h5"
            )

        # Progress bar
        if n % (int(n_steps / 100)) == 0:

            # Progress and number of fragments print
            fragment_data = aka.FragmentManager(model)
            fragment_data.computeAllData()
            nb_fragments = fragment_data.getNbFragment()
            print(
                "Progress: {:.3f}".format(n / n_steps * 100),
                "%",
                " | Inserted: ",
                nb_inserted,
                " | Fragments: ",
                nb_fragments,
            )


if __name__ == "__main__":

    args = parseArguments()
    args.time = TIME

    # Set mesh file based on --unit flag
    mesh_file = (
        "hollow_sphere_rin9.0e-01_rout1.0e+00_lc5.0e-02_p1.msh"
        if args.unit
        else args.mesh_file
    )

    # Display input file information
    print(f"Material file: {args.material_file}")
    print(f"Mesh file: {args.mesh_file}")
    print(f"Strain rate: {args.strain_rate:.2e} s^-1")
    print(f"Safety factor: {args.safety_factor}")
    print("Contact penalty:", args.contact)
    print("Unit test case:", args.unit)

    ## Output directory as a function of local or cluster run ##
    nname = os.uname().nodename
    print(f"Running on: {nname} \n")
    inpath, outpath = setupDir(nname, args)

    # Display paths
    print(f"Input directory: {inpath}")
    print(f"Output directory: {outpath} \n")

    # Run the simulation
    runSimulation(
        inpath + f"input/material/{args.material_file}",
        inpath + f"input/mesh/{args.mesh_file}",
        [inpath, outpath],
        args,
        free_edges=True,
    )
