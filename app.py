import io
import os
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st
import torch

from model_inference import load_trained_model, predict_volume

st.set_page_config(page_title="NeuroVision AI", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.stApp{background:radial-gradient(ellipse at 15% 0%,#17183b 0%,#0a0b16 48%,#080a12 100%);color:#eef0ff}
[data-testid="stSidebar"]{background:#101222;border-right:1px solid #292b4c}
.block-container{padding-top:1.5rem;max-width:1500px}
.small-muted{color:#9ca3c7;font-size:.9rem}
.panel{background:linear-gradient(145deg,rgba(28,30,59,.92),rgba(14,16,32,.95));border:1px solid #2b2d52;border-radius:18px;padding:16px}
div[data-testid="stMetric"]{background:#15172d;border:1px solid #292b4c;padding:14px;border-radius:14px}
.stButton>button{border-radius:10px;border:1px solid #5754a8;background:linear-gradient(90deg,#6556dc,#3977f6);color:white;font-weight:650}
</style>
""", unsafe_allow_html=True)
st.markdown('<span style="padding:5px 10px;border-radius:20px;background:#202044;border:1px solid #414276;color:#c9caff">RESEARCH WORKSTATION · MULTIMODAL MRI</span>', unsafe_allow_html=True)
st.title("🧠 NeuroVision AI")
st.markdown('<p class="small-muted">Multimodal brain MRI · 3D slice viewer · AI tumour-region segmentation</p>', unsafe_allow_html=True)
st.warning("Academic research prototype only. Predictions are not clinically validated and must not be used for diagnosis or treatment decisions.")

checkpoint_path=Path(os.environ.get("NEUROVISION_CHECKPOINT","brats3d_unet_balanced_dice_ce_epoch006.pth"))
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
with st.sidebar:
    st.header("Workspace")
    st.markdown("**Input order:** FLAIR · T1 · T1ce · T2")
    st.markdown("**Encoded output:**")
    st.markdown("🟦 Label 1 · 🟨 Label 2")
    if checkpoint_path.is_file(): st.success(f"Checkpoint found: {checkpoint_path.name}")
    else: st.error("Checkpoint missing"); st.caption("Place the .pth file in the app root or set NEUROVISION_CHECKPOINT.")
    st.caption(f"Device: {device}")

def read_nifti(data, filename):
    suffix=".nii.gz" if Path(filename).name.lower().endswith(".nii.gz") else ".nii"
    temp=None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as f:
            f.write(data); temp=f.name
        img=nib.load(temp)
        return img.get_fdata(dtype=np.float32),img.affine.copy(),img.header.copy()
    finally:
        if temp and os.path.exists(temp): os.remove(temp)

def slice_fig(volume,z,title):
    fig,ax=plt.subplots(figsize=(5,5),facecolor="#0c0d18")
    ax.imshow(np.asarray(volume[:,:,z]).T,cmap="gray",origin="lower")
    ax.set_title(title,color="#e9eaff"); ax.axis("off"); fig.tight_layout()
    return fig

def overlay_fig(volume,pred,z):
    fig,ax=plt.subplots(figsize=(7,7),facecolor="#0c0d18")
    ax.imshow(np.asarray(volume[:,:,z]).T,cmap="gray",origin="lower")
    labels=np.asarray(pred[:,:,z]).T
    rgba=np.zeros((*labels.shape,4),dtype=np.float32)
    for cls,color in {1:(0,.85,.95,.58),2:(1.0,.78,0.0,.52)}.items():
        rgba[labels==cls]=color
    ax.imshow(rgba,origin="lower"); ax.set_title(f"AI prediction · axial slice {z}",color="#e9eaff"); ax.axis("off")
    fig.tight_layout(); return fig

def zip_map(archive):
    mapping={}
    names=[n for n in archive.namelist() if not n.endswith("/") and n.lower().endswith((".nii",".nii.gz"))]
    for name in names:
        b=Path(name).name.lower()
        if "t1ce" in b or "t1c" in b: mapping.setdefault("T1ce",name)
        elif "flair" in b: mapping.setdefault("FLAIR",name)
        elif "t1" in b: mapping.setdefault("T1",name)
        elif "t2" in b: mapping.setdefault("T2",name)
    return mapping

st.header("01 / Import MRI volumes")
st.caption("Upload four aligned 3D NIfTI modalities individually or as a ZIP. Input volumes must already be co-registered and have identical voxel grids.")
mode=st.radio("Upload method",["Four individual files","One ZIP archive"],horizontal=True)
volumes={}; affine=None; header=None; loaded=False
if mode=="Four individual files":
    cols=st.columns(4); files={}
    for col,key in zip(cols,("FLAIR","T1","T1ce","T2")):
        with col:
            f=st.file_uploader(f"{key} volume",type=["nii","gz"],key=f"upload_{key}")
            if f: files[key]=f
    if len(files)==4:
        try:
            affines=[]
            for key in ("FLAIR","T1","T1ce","T2"):
                v,a,h=read_nifti(files[key].getvalue(),files[key].name); volumes[key]=v; affines.append(a)
                if key=="FLAIR": affine,header=a,h
            if any(not np.allclose(affines[0],a,atol=1e-3) for a in affines[1:]): st.error("MRI affines differ; upload co-registered modalities.")
            else: loaded=True
        except Exception as e: st.error(f"Could not read NIfTI: {e}")
else:
    f=st.file_uploader("ZIP containing FLAIR, T1, T1ce and T2 NIfTI files",type=["zip"])
    if f:
        try:
            with zipfile.ZipFile(io.BytesIO(f.getvalue())) as archive:
                mapping=zip_map(archive); missing=[k for k in ("FLAIR","T1","T1ce","T2") if k not in mapping]
                if missing: st.error("Could not identify: "+", ".join(missing))
                else:
                    affines=[]
                    for key in ("FLAIR","T1","T1ce","T2"):
                        v,a,h=read_nifti(archive.read(mapping[key]),mapping[key]); volumes[key]=v; affines.append(a)
                        if key=="FLAIR": affine,header=a,h
                    if any(not np.allclose(affines[0],a,atol=1e-3) for a in affines[1:]): st.error("MRI affines differ; upload co-registered modalities.")
                    else: loaded=True
        except Exception as e: st.error(f"Could not read ZIP: {e}")

if loaded:
    shapes={k:v.shape for k,v in volumes.items()}
    if any(v.ndim!=3 for v in volumes.values()) or len(set(shapes.values()))!=1:
        st.error(f"All modalities must be aligned 3D volumes with matching dimensions. Got {shapes}."); loaded=False
if loaded:
    shape=volumes["FLAIR"].shape
    a,b,c,d=st.columns(4); a.metric("Volume dimensions",f"{shape[0]} × {shape[1]} × {shape[2]}"); b.metric("Axial slices",shape[2]); c.metric("Modalities","4 / 4"); d.metric("Checkpoint","Ready" if checkpoint_path.is_file() else "Missing")
    z=st.slider("Axial slice",0,shape[2]-1,shape[2]//2)
    st.header("02 / Multimodal viewer")
    cols=st.columns(4)
    for col,key in zip(cols,("FLAIR","T1","T1ce","T2")):
        with col:
            st.pyplot(slice_fig(volumes[key],z,f"{key} · slice {z}"),use_container_width=True)
    st.header("03 / AI tumour segmentation")
    st.caption("Inference uses overlapping 96³ patches. Correct preprocessing, modality order, orientation and label mapping must match the training pipeline.")
    if not checkpoint_path.is_file():
        st.warning("Viewer is available, but AI prediction is disabled until the trained checkpoint is installed.")
    elif st.button("Run AI segmentation",type="primary"):
        try:
            with st.spinner(f"Loading checkpoint and running inference on {device}…"):
                model,ckpt=load_trained_model(checkpoint_path,device)
                prediction=predict_volume(model,volumes,device,patch_size=96,overlap=.5)
            st.session_state["nv_prediction"]=prediction
            st.session_state["nv_shape"]=shape
            st.success(f"Inference finished using checkpoint epoch {ckpt.get('epoch','unknown')}. This is a model output, not a diagnosis.")
        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.info("Strict checkpoint loading prevents a different architecture from being silently used.")
    prediction=st.session_state.get("nv_prediction")
    if prediction is not None and st.session_state.get("nv_shape")==shape:
        st.header("04 / Predicted tumour overlay")
        # Hide the enhancing-tumour class entirely, as requested; retain only labels 1 and 2.
        display_prediction=prediction.copy()
        display_prediction[display_prediction==3]=0
        st.pyplot(overlay_fig(volumes["FLAIR"],display_prediction,z),use_container_width=True)
        x,y=st.columns(2); x.metric("Label 1 voxels",f"{int((display_prediction==1).sum()):,}"); y.metric("Label 2 voxels",f"{int((display_prediction==2).sum()):,}")
        st.caption("Overlay shows only labels 1 and 2. This research prototype can produce false positives and is not suitable for clinical decisions.")
        tmp=None
        try:
            with tempfile.NamedTemporaryFile(suffix=".nii.gz",delete=False) as f: tmp=f.name
            h=header.copy(); h.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(display_prediction.astype(np.uint8),affine,h),tmp)
            with open(tmp,"rb") as f: mask_bytes=f.read()
            st.download_button("Download predicted mask (.nii.gz)",mask_bytes,"neurovision_predicted_mask_labels_1_2_only.nii.gz","application/gzip")
        finally:
            if tmp and os.path.exists(tmp): os.remove(tmp)
        st.header("05 / Remote specialist review")
        st.text_area("Review notes (prototype; avoid identifiable patient information)",key="nv_review_notes")
else:
    st.markdown('<div class="panel"><b>Workspace ready</b><p class="small-muted">Upload four aligned MRI modalities to view slices and run inference with the installed checkpoint.</p></div>',unsafe_allow_html=True)
st.markdown("---")
st.markdown('<p class="small-muted">NeuroVision AI · Academic research prototype · Not for clinical decision-making</p>',unsafe_allow_html=True)
