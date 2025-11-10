from lammps import lammps

# Create a LAMMPS object
lmp = lammps()

# Run LAMMPS commands
lmp.command("units lj")
lmp.command("atom_style atomic")
lmp.command("lattice fcc 0.8442")
lmp.command("region box block 0 10 0 10 0 10")
lmp.command("create_box 1 box")
lmp.command("create_atoms 1 box")
lmp.command("mass 1 1.0")
lmp.command("velocity all create 1.44 87287 loop geom")
lmp.command("pair_style lj/cut 2.5")
lmp.command("pair_coeff 1 1 1.0 1.0 2.5")
lmp.command("neighbor 0.3 bin")
lmp.command("neigh_modify delay 0 every 20 check no")
lmp.command("fix 1 all nve")
lmp.command("run 100")

# Close the LAMMPS object
lmp.close()