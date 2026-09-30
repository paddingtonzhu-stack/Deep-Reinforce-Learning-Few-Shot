"""Windows-safe entry point for the upstream Sample Factory corridor checkpoint."""
import os
import sys
import traceback

import numpy as np
import psutil
import torch

if os.name == "nt" and not hasattr(os, "getuid"):
    os.getuid = lambda: 0  # type: ignore[attr-defined]

# The 2023 checkpoint stores a NumPy scalar in optimizer metadata. Keep modern
# weights-only loading enabled and allow only this exact legacy scalar type.
torch.serialization.add_safe_globals(
    [np.core.multiarray.scalar, np.dtype, type(np.dtype(np.float64))]
)

if os.name == "nt":
    from sf_examples.vizdoom.doom.doom_gym import VizdoomEnv

    # ViZDoom's native close can block indefinitely on this Windows host.
    # Child cleanup is handled explicitly in the process-level finally below.
    VizdoomEnv.close = lambda self: None

from sf_examples.vizdoom.enjoy_vizdoom import main  # noqa: E402

if __name__ == "__main__":
    exit_code = 0
    try:
        result = main()
        if isinstance(result, int):
            exit_code = result
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        # ViZDoom 1.2.x can block in DoomGame.close() on Windows.
        sys.stdout.flush()
        sys.stderr.flush()
        if os.name == "nt":
            children = psutil.Process().children(recursive=True)
            for child in children:
                child.terminate()
            _, alive = psutil.wait_procs(children, timeout=2)
            for child in alive:
                child.kill()
            os._exit(exit_code)
