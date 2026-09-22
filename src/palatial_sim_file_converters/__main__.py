import sys

from palatial_sim_file_converters.cli import convert_main, main

if len(sys.argv) > 1 and sys.argv[1] not in {"check", "convert", "-h", "--help"}:
    raise SystemExit(convert_main())
raise SystemExit(main())
