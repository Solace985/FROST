# hf_space/weights/

The trained **native-392 MultiTaskHead** checkpoint goes here as
`mt_head_native392.pt` (≈ 501 KB) when assembling the Space.

`assemble_space.py` copies it here automatically from the accepted training run
(`runs/train/brset_20260525_113425/model_checkpoint.pt`, or the path given by
`RETINA_SCREEN_RETF_NATIVE392_MT_CHECKPOINT`). It is **git-ignored in the main
repo** (see `deploy/referable_dr_demo/.gitignore`) so it is never committed to
the private research repo; it is committed only into the separate public Space
repo.

The 84 MB RETFound-Green backbone is **not** placed here — it is public
(Apache-2.0) and is downloaded + SHA-verified by the Dockerfile at build time.
