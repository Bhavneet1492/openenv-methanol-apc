# Physics Engine

::: methanol_apc_env.server.reactor_sim

## Simulation Features

| Feature | Implementation |
|---------|---------------|
| ODE integration | 4th-order Runge-Kutta (4 sub-steps) |
| Equation of state | SRK cubic EOS with fugacity corrections |
| Kinetic models | LHHW, Graaf 1988, VBF 1996, Seyfert/BASF, Nestler 2021 |
| Reactor types | ICI 4-bed adiabatic quench, Lurgi isothermal |
| Pressure drop | Ergun equation across packed bed |
| Catalyst deactivation | 3-zone model (normal / above-optimal / sintering) |

## 3 Simultaneous Reactions

| # | Reaction | ΔH |
|---|----------|:--:|
| R1 | CO + 2H₂ → CH₃OH | −90.5 kJ/mol |
| R2 | CO₂ + 3H₂ → CH₃OH + H₂O | −49.5 kJ/mol |
| R3 | CO₂ + H₂ → CO + H₂O | +41.2 kJ/mol |
