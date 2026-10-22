import io, zipfile, tempfile, os
from pathlib import Path
import numpy as np
import nibabel as nib
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

st.set_page_config(page_title="NeuroVision AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp {background: radial-gradient(ellipse at 15% 0%, #17183b 0%, #0a0b16 48%, #080a12 100%); color:#eef0ff;}
[data-testid="stSidebar"] {background:#101222; border-right:1px solid #292b4c;}
.block-container {padding-top:1.5rem; max-width:1500px;}
h1,h2,h3,p,label,span,div {font-family: Inter, ui-sans-serif, system-ui, sans-serif;}
h1 {letter-spacing:-1.5px;}
.small-muted {color:#9ca3c7; font-size:0.9rem;}
.panel {background:linear-gradient(145deg,rgba(28,30,59,.92),rgba(14,16,32,.95));border:1px solid #2b2d52;border-radius:18px;padding:16px 18px;}
.tag {display:inline-block;padding:5px 10px;border-radius:20px;background:#202044;border:1px solid #414276;color:#c9caff;font-size:12px;}
div[data-testid="stMetric"] {background:#15172d;border:1px solid #292b4c;padding:14px;border-radius:14px;}
.stButton>button {border-radius:10px;border:1px solid #5754a8;background:linear-gradient(90deg,#6556dc,#3977f6);color:white;font-weight:650;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="tag">RESEARCH WORKSTATION · MULTIMODAL MRI</div>', unsafe_allow_html=True)
st.title("🧠 NeuroVision AI")
st.markdown('<p class="small-muted">Multimodal brain MRI · 3D slice viewer · Tumour-region visualisation</p>', unsafe_allow_html=True)
st.warning("Research prototype only. This tool is not validated for clinical diagnosis. No tumour prediction is generated unless trained model weights are installed and verified.")

with st.sidebar:
    st.markdown("## Workspace")
    st.markdown("**Input modalities**")
    st.caption("T1 · T1ce · T2 · FLAIR")
    st.markdown("---")
    st.markdown("**Segmentation legend**")
    st.markdown("🟣 **Label 1** — model output class 1")
    st.markdown("🔵 **Label 2** — model output class 2")
    st.caption("Enhancing tumour label 4 is excluded from display.")
    st.markdown("---")
    st.markdown("**Model status**")
    weights_path = Path("model_weights.pth")
    if weights_path.exists():
        st.success("Weights file found. Architecture and label mapping still require verification.")
    else:
        st.error("Trained weights not installed")
    st.caption("Upload a checkpoint only after confirming it matches the trained architecture.")

def read_nifti(file_bytes, filename):
    # nibabel.load expects a filesystem path here. Save uploaded/ZIP bytes
    # to a temporary NIfTI file, fully cache the voxel data, then remove it.
    lower_name = Path(filename).name.lower()
    suffix = ".nii.gz" if lower_name.endswith(".nii.gz") else ".nii"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name
        img = nib.load(temp_path)
        # Force data into memory before the temporary file is deleted.
        img.get_fdata(dtype=np.float32)
        return img
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

def load_upload(uploaded):
    data = uploaded.getvalue()
    return read_nifti(data, uploaded.name)

def show_slice(vol, z, title, cmap="gray", vmin=None, vmax=None):
    arr = np.asarray(vol[:, :, z]).T
    fig, ax = plt.subplots(figsize=(5,5), facecolor="#0c0d18")
    ax.set_facecolor("#0c0d18")
    ax.imshow(arr, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    ax.set_title(title, color="#e9eaff", fontsize=11, pad=10)
    ax.axis("off")
    fig.tight_layout()
    return fig

st.markdown("### 01 / Import MRI volumes")
st.markdown('<div class="small-muted">Upload four spatially aligned NIfTI volumes individually, or a ZIP containing files named with their modality.</div>', unsafe_allow_html=True)
upload_mode = st.radio("Upload method", ["Four individual files", "One ZIP archive"], horizontal=True, label_visibility="collapsed")
volumes, headers, affine = {}, {}, None
uploaded_ok = False

if upload_mode == "Four individual files":
    cols = st.columns(4)
    modality_files = {}
    for col, key in zip(cols, ["T1", "T1ce", "T2", "FLAIR"]):
        with col:
            f = st.file_uploader(key, type=["nii", "gz"], key=f"file_{key}")
            if f:
                modality_files[key] = f
    if len(modality_files) == 4:
        try:
            for key, f in modality_files.items():
                img = load_upload(f)
                volumes[key] = img.get_fdata(dtype=np.float32)
                headers[key] = img.header.copy()
                if key == "T1": affine = img.affine.copy()
            uploaded_ok = True
        except Exception as e:
            st.error(f"Could not read a NIfTI volume: {e}")
else:
    zf = st.file_uploader("Upload ZIP containing T1, T1ce, T2 and FLAIR .nii/.nii.gz files", type=["zip"])
    if zf:
        try:
            with zipfile.ZipFile(io.BytesIO(zf.getvalue())) as archive:
                names = [n for n in archive.namelist() if not n.endswith("/") and (n.lower().endswith(".nii") or n.lower().endswith(".nii.gz"))]
                mapping = {}
                for n in names:
                    low = Path(n).name.lower()
                    for key in ["t1ce", "flair", "t1", "t2"]:
                        if key in low and key.upper() not in mapping:
                            mapping[key.upper()] = n
                if all(k in mapping for k in ["T1", "T1CE", "T2", "FLAIR"]):
                    canonical = {"T1":"T1", "T1CE":"T1ce", "T2":"T2", "FLAIR":"FLAIR"}
                    for k, name in mapping.items():
                        img = read_nifti(archive.read(name), name)
                        volumes[canonical[k]] = img.get_fdata(dtype=np.float32)
                        headers[canonical[k]] = img.header.copy()
                        if k == "T1": affine = img.affine.copy()
                    uploaded_ok = True
                else:
                    st.error("Could not identify all four files. Filenames should contain T1, T1ce, T2 and FLAIR.")
        except Exception as e:
            st.error(f"Could not read ZIP: {e}")

if uploaded_ok:
    shapes = {k: v.shape for k, v in volumes.items()}
    if len(set(shapes.values())) != 1:
        st.error(f"Volume dimensions do not match: {shapes}. Please upload aligned volumes in the same space.")
        uploaded_ok = False
    elif any(v.ndim != 3 for v in volumes.values()):
        st.error("Each modality must be a 3D NIfTI volume.")
        uploaded_ok = False

if uploaded_ok:
    shape = next(iter(volumes.values())).shape
    st.markdown("### 02 / Scan overview")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Volume dimensions", f"{shape[0]} × {shape[1]}")
    m2.metric("Axial slices", shape[2])
    m3.metric("Modalities loaded", "4 / 4")
    m4.metric("Segmentation", "Not run")
    z = st.slider("Axial slice", 0, shape[2]-1, shape[2]//2)
    zoom = st.slider("Display contrast", 0.5, 3.0, 1.0, 0.1)
    st.markdown("### 03 / Multimodal viewer")
    cols = st.columns(4)
    for col, key in zip(cols, ["T1", "T1ce", "T2", "FLAIR"]):
        with col:
            st.pyplot(show_slice(volumes[key], z, f"{key} · slice {z}"), use_container_width=True)
    st.markdown("### 04 / Segmentation result")
    st.info("The MRI volumes are loaded successfully. The notebook builds a model architecture, but a trained checkpoint and verified class-to-label mapping have not been provided. Therefore, this version will not invent tumour masks or display fabricated predictions.")
    st.markdown('<div class="panel"><b>Inference status:</b> Waiting for verified trained weights<br><span class="small-muted">Required before real Label 1 / Label 2 overlays can be generated.</span></div>', unsafe_allow_html=True)
    st.markdown("### 05 / Export")
    st.caption("NIfTI export becomes available after a real model inference produces a segmentation volume. No empty or synthetic mask is exported as a prediction.")
else:
    st.markdown("### Your workspace")
    c1,c2,c3 = st.columns(3)
    c1.markdown('<div class="panel"><h3>01 · Import</h3><p class="small-muted">NIfTI / compressed NIfTI, or a ZIP archive.</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="panel"><h3>02 · Inspect</h3><p class="small-muted">Four modalities, shared slice navigation and dimension checks.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="panel"><h3>03 · Segment</h3><p class="small-muted">Label 1 and 2 overlays once trained weights are verified.</p></div>', unsafe_allow_html=True)
    st.caption("Begin by uploading all four modalities. For a ZIP, include filenames containing T1, T1ce, T2 and FLAIR.")

st.markdown("---")
st.markdown('<p class="small-muted">NeuroVision AI · Academic research prototype · Not for clinical decision-making</p>', unsafe_allow_html=True)

