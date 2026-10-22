# NeuroVision AI

A dark, purple/blue Streamlit research workstation for four-modality brain MRI review.

## Current capabilities
- Upload T1, T1ce, T2 and FLAIR as individual NIfTI files or one ZIP archive.
- Validate that all four inputs are 3D and have matching dimensions.
- Browse axial slices and preview all four modalities.
- Explicitly avoids fabricating tumour masks when trained weights are unavailable.

## Important model status
The provided notebook defines a 3D model and runs forward passes on a randomly initialized model. It does not demonstrate loading a trained checkpoint or show trained inference results. Therefore this app intentionally does not claim to segment tumours yet.

Before enabling segmentation:
1. Supply the actual trained checkpoint (`.pth` / `.pt`).
2. Confirm the exact model architecture and preprocessing used during training.
3. Verify how the three output channels map to segmentation labels.
4. Test on held-out cases with known ground truth.
5. Display only labels 1 and 2; suppress label 4 in the viewer and exported mask.

Do not commit patient MRI files or model weights to a public repository.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy publicly with Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and this README.
2. Visit https://share.streamlit.io and sign in with GitHub.
3. Create an app, select your repository, branch `main`, and entrypoint `app.py`.
4. Deploy. Streamlit will provide the actual public URL.

A public URL cannot be created from this package alone; deployment must be authorized from your GitHub/Streamlit account.
