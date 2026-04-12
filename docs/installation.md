# Installation

## Default (No External Tools)

```bash
pip install "openenv-core[core]>=0.2.2" numpy fastmcp
```

This gives you the full environment with internal physics models. No external tools needed.

## With DWSIM (Thermodynamic Validation)

```bash
pip install pythonnet
```

Then install DWSIM from [dwsim.org/downloads](https://dwsim.org/index.php/downloads) and set:

```bash
export DWSIM_PATH="/path/to/dwsim"   # Linux/Mac
set DWSIM_PATH=C:\path\to\DWSIM      # Windows
```

## With Cantera (Kinetics Validation)

```bash
pip install cantera
```

Or via conda: `conda install -c cantera cantera`

## With Azure Digital Twins

```bash
pip install azure-digitaltwins-core azure-identity
az login
```

Set your ADT endpoint:
```bash
export AZURE_DIGITAL_TWINS_URL=https://your-instance.api.eus.digitaltwins.azure.net
```

See the [Azure Digital Twins guide](integrations/azure-digital-twins.md) for full setup.

## Development

```bash
git clone https://github.com/Bhavneet1492/openenv-methanol-apc.git
cd openenv-methanol-apc
pip install "openenv-core[core]>=0.2.2" numpy fastmcp pytest
PYTHONPATH=. python -m pytest methanol_apc_env/tests/ -v
```
