# NeuroVision AI

A Streamlit research prototype for reviewing four aligned brain MRI modalities (FLAIR, T1, T1ce and T2) and viewing predicted tumour-region masks.

## Current capabilities
- Upload four aligned 3D NIfTI MRI volumes individually or as a ZIP.
- Browse axial slices across the four modalities.
- Run sliding-window 3D U-Net inference when a compatible trained checkpoint is available.
- Overlay predicted **Label 1 in cyan** and **Label 2 in yellow**.
- Hide output class 3 (the configured BraTS label-4 / enhancing-tumour class) from the displayed and exported mask.
- Download the displayed Label 1/Label 2 mask as NIfTI.
- Show the project's previously recorded held-out Dice metrics: Label 1 = 0.1680 and Label 2 = 0.3641.

## Checkpoint setup (required for new predictions)
The model weights are not included in this repository. The app will display the MRI viewer, but new-scan inference requires the actual trained checkpoint file.

1. Put the compatible checkpoint in the app root as `brats3d_unet_balanced_dice_ce_epoch006.pth`, **or** configure the `NEUROVISION_CHECKPOINT` environment variable to its path.
2. Ensure the checkpoint contains a `model_state_dict` matching the architecture in `model_inference.py`.
3. Verify preprocessing, modality order, orientation, label mapping, and performance on held-out cases from the original training pipeline before relying on outputs.

Do not commit patient MRI data or sensitive files to a public repository. If checkpoint size or repository policy makes GitHub unsuitable, use a permitted private artifact store and configure the deployment to access it securely.

## Model verification
The displayed Dice values are previously recorded dataset-level results supplied for this project. They are **not** a per-scan confidence score and do not independently establish clinical performance. The current code is an academic prototype and has not been clinically validated.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with Streamlit Community Cloud
Select this repository, branch `main`, and `app.py` as the entry point. The app will redeploy after commits to the selected branch. Confirm the checkpoint is accessible in the deployed environment before testing new-scan inference.

**Not for clinical decision-making.** Predictions may be inaccurate and must not be used to diagnose, treat, or rule out a brain tumour.
