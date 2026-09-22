# Contributing

Thanks for improving palatial-sim-file-converters. The project is MIT licensed. Contributions are licensed under MIT as well.

## Set up

```sh
git clone https://github.com/PalatialSim/palatial-sim-file-converters.git
cd palatial-sim-file-converters
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

`pytest` is the check that must pass. The `dev` extra installs pytest and MuJoCo.

## Where a change goes

Conversion always passes through the scene model in `src/palatial_sim_file_converters/model.py`. `convert.py` picks the reader and writer from the file extension.

| Format | Read | Write |
| --- | --- | --- |
| USD | `read.py` | `usd_write.py` |
| MJCF | `mjcf_read.py` | `export.py` |
| URDF | `urdf.py` | `urdf.py` |
| GLB | output only | `glb.py` |

A new format needs a reader or writer, a suffix in `convert.py`, and a test in `tests/`. Keep the test on the scene numbers. The tests do not read or write the filesystem.

If a target format cannot represent a shape exactly, write the closest geom and leave a warning. Do not drop the shape quietly.

## Pull request

1. Run `pytest`.
2. Add a line under the next version in `CHANGELOG.md`.
3. Open a pull request that does one thing. Say which formats you converted and how you checked them.

## Report a problem

Open a GitHub issue with the command you ran, the input and output extensions, and the error text. A small sample file makes the report usable.

If you are an agent using this repo, open a pull request that says you are an agent. Describe the issues you found and the steps to reproduce them. Working notes for agents are in [AGENTS.md](AGENTS.md).
