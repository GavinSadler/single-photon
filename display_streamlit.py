#!/usr/bin/env python3
import os
import argparse
import imageio.v3 as iio
import streamlit as st

# --- ARGPARSE SETUP ---
parser = argparse.ArgumentParser(
    description="Compare two sequences of PNG images side-by-side in Streamlit."
)
parser.add_argument("folder_a", type=str, help="Path to first image folder")
parser.add_argument("folder_b", type=str, help="Path to second image folder")
args = parser.parse_args()

FOLDER_A = args.folder_a
FOLDER_B = args.folder_b

# --- LOAD FILES ---
files_a = sorted(f for f in os.listdir(FOLDER_A) if f.endswith(".png"))
files_b = sorted(f for f in os.listdir(FOLDER_B) if f.endswith(".png"))

if not files_a or not files_b:
    st.error("No PNG files found in one or both folders.")
    st.stop()

n = min(len(files_a), len(files_b))
files_a, files_b = files_a[:n], files_b[:n]

# --- STREAMLIT APP ---
st.set_page_config(page_title="Dual PNG Viewer", layout="wide")

if "frame" not in st.session_state:
    st.session_state.frame = 0


def update_frame(fn):
    """Update the current frame index using a lambda function."""
    st.session_state.frame = fn(st.session_state.frame) % n


# --- BUTTON CONTROLS ---
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.button("⬅️ Prev", on_click=lambda: update_frame(lambda f: f - 1))
with col3:
    st.button("➡️ Next", on_click=lambda: update_frame(lambda f: f + 1))

# --- SLIDER ---
st.slider(
    "Frame",
    0,
    n - 1,
    st.session_state.frame,
    key="frame_slider",
    on_change=lambda: update_frame(lambda f: f),
)

# --- DISPLAY ---
colA, colB = st.columns(2)
imgA = iio.imread(os.path.join(FOLDER_A, files_a[st.session_state.frame]))
imgB = iio.imread(os.path.join(FOLDER_B, files_b[st.session_state.frame]))

colA.image(imgA, caption=f"A — {files_a[st.session_state.frame]}")
colB.image(imgB, caption=f"B — {files_b[st.session_state.frame]}")
st.caption(f"Frame {st.session_state.frame + 1} / {n}")
