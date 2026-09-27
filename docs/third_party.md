# Third-party provenance

QuantVLA is vendored from the author's integration fork at commit `01e655d`
(`https://github.com/ruanruan-andy/QuantVLA.git`). Its upstream authors,
README attribution, and Apache-2.0 license are retained. Runtime artifacts,
standalone plotting tools, and example data are omitted from this distribution.
Release-local changes normalize workspace paths and PIVOT-Q interface names.

The following dependencies are official Git submodules. Their full commit IDs
and URLs are recorded in `.gitmodules` and `upstream_patches/manifest.json`:

| Dependency | Official source |
|---|---|
| LIBERO | https://github.com/Lifelong-Robot-Learning/LIBERO |
| LIBERO-Plus | https://github.com/sylvestf/LIBERO-plus |
| LeRobot | https://github.com/huggingface/lerobot |
| Omega-QVLA | https://github.com/UCMP13753/Omega-QVLA |
| QVLA | https://github.com/AutoLab-SAI-SJTU/QVLA |

Only the two LIBERO repositories receive the documented compatibility links.
Do not remove or replace upstream licenses and notices. Model weights and data
have their own access and redistribution terms; they are not bundled here.
The primary-project license must be selected by its author before public release.
