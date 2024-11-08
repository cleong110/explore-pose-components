import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import pandas as pd
import numpy as np
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from pathlib import Path
from pyzstd import decompress
from PIL import Image
import cv2
import mediapipe as mp
import torch

mp_holistic = mp.solutions.holistic
FACEMESH_CONTOURS_POINTS = [str(p) for p in sorted(set([p for p_tup in list(mp_holistic.FACEMESH_CONTOURS) for p in p_tup]))]

def pose_normalization_info(pose_header):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
                                            p2=("POSE_LANDMARKS", "LEFT_SHOULDER"))

    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(p1=("BODY_135", "RShoulder"), p2=("BODY_135", "LShoulder"))

    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(p1=("pose_keypoints_2d", "RShoulder"),
                                                p2=("pose_keypoints_2d", "LShoulder"))


def pose_hide_legs(pose):
    if pose.header.components[0].name == "POSE_LANDMARKS":
        point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        # pylint: disable=protected-access
        points = [
            pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
            for n in point_names
            for side in ["LEFT", "RIGHT"]
        ]
        pose.body.confidence[:, :, points] = 0
        pose.body.data[:, :, points, :] = 0
        return pose
    else:
        raise ValueError("Unknown pose header schema for hiding legs")


def preprocess_pose(pose):
    pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"], 
                        {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS})

    pose = pose.normalize(pose_normalization_info(pose.header))
    pose = pose_hide_legs(pose)
    
    # from sign_vq.data.normalize import pre_process_mediapipe, normalize_mean_std
    # from pose_anonymization.appearance import remove_appearance

    # pose = remove_appearance(pose)
    # pose = pre_process_mediapipe(pose)
    # pose = normalize_mean_std(pose)

    feat = np.nan_to_num(pose.body.data)
    feat = feat.reshape(feat.shape[0], -1)

    pose_frames = torch.from_numpy(np.expand_dims(feat, axis=0)).float()

    return pose_frames


# @st.cache_data(hash_funcs={UploadedFile: lambda p: str(p.name)})
def load_pose(uploaded_file:UploadedFile)->Pose:

    # with input_path.open("rb") as f_in:
    if uploaded_file.name.endswith(".zst"):
        return Pose.read(decompress(uploaded_file.read()))
    else:
        return Pose.read(uploaded_file.read())

@st.cache_data(hash_funcs={Pose: lambda p: np.array(p.body.data)})
def get_pose_frames(pose:Pose, transparency: bool = False):
    v = PoseVisualizer(pose)
    frames = [frame_data for frame_data in v.draw()]

    if transparency:
        cv_code = v.cv2.COLOR_BGR2RGBA
    else:
        cv_code = v.cv2.COLOR_BGR2RGB
    images = [Image.fromarray(v.cv2.cvtColor(frame, cv_code)) for frame in frames]
    return frames, images

def get_pose_gif(pose:Pose, step:int=1, fps:int=None):
    if fps is not None:
        pose.body.fps = fps
    v = PoseVisualizer(pose)
    frames = [frame_data for frame_data in v.draw()]
    frames = frames[::step]
    return v.save_gif(None,frames=frames)


uploaded_file = st.file_uploader("gimme a .pose file", type=[".pose", ".pose.zst"])



if uploaded_file is not None:
    with st.spinner(f"Loading {uploaded_file.name}"):
        pose = load_pose(uploaded_file)
        frames, images = get_pose_frames(pose=pose)
    st.success("done loading!")
    # st.write(f"pose shape: {pose.body.data.shape}")
        

    header = pose.header
    st.write("### File Info")
    with st.expander(f"Show full Pose-format header from {uploaded_file.name}"):
        
        st.write(header)
    # st.write(pose.body.data.shape)
    # st.write(pose.body.fps)

    st.write(f"### Selection")

    components = pose.header.components

    component_names = [component.name for component in components]
    chosen_component_names = component_names

    component_selection = st.radio("How to select components?", options=["manual", "signclip"])
    if component_selection == "manual":
        st.write(f"### Component selection: ")
        chosen_component_names = st.pills("Components to visualize", options=component_names, selection_mode="multi", default=component_names)
        
        # st.write(chosen_component_names)
        
        
        
        st.write("### Point selection:")
        point_names = []
        new_chosen_components =[]        
        points_dict = {}
        for component in pose.header.components:
                with st.expander(f"points for {component.name}"):
                    
                    if component.name in chosen_component_names:
                        
                        st.write(f"#### {component.name}")
                        selected_points = st.multiselect(f"points for component {component.name}:",options=component.points, default=component.points)
                        if selected_points == component.points:
                            st.write(f"All selected, no need to add a points dict entry for {component.name}")
                        else:
                            st.write(f"Adding dictionary for {component.name}")            
                            points_dict[component.name] = selected_points
        
                    
        # selected_points = st.multiselect("points to visualize", options=point_names, default=point_names)
        if chosen_component_names:
            
            if not points_dict:
                points_dict=None
            # else: 
            #     st.write(points_dict)
            # st.write(chosen_component_names)

            pose = pose.get_components(chosen_component_names,points=points_dict)
            # st.write(pose.header)

    elif component_selection == "signclip":
        st.write("Selected landmarks used for SignCLIP. (Face countours only)")
        pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"], 
                    {"FACE_LANDMARKS": FACEMESH_CONTOURS_POINTS})

        # pose = pose.normalize(pose_normalization_info(pose.header)) Visualization goes blank
        pose = pose_hide_legs(pose)
        with st.expander("Show facemesh contour points:"):
            st.write(f"{FACEMESH_CONTOURS_POINTS}")
        with st.expander(f"Show header:"):
            st.write(pose.header)
        # st.write(f"signclip selected, new header:")
        # st.write(pose.body.data.shape)
        # st.write(pose.header)
    else:
        pass
        

    
    st.write(f"### Visualization")
    width=st.select_slider("select width of images",list(range(1,pose.header.dimensions.width +1)),value=pose.header.dimensions.width/2)
    step=st.select_slider("Step value to select every nth image",list(range(1,len(frames))),value=1)
    fps=st.slider("fps for visualization: ", min_value=1.0, max_value=pose.body.fps,value=pose.body.fps)
    visualize_clicked = st.button(f"Visualize!")
    
    

    if visualize_clicked:
        
        st.write(f"Generating gif...")

        # st.write(pose.body.data.shape)
        
        st.image(get_pose_gif(pose=pose, step=step, fps=fps))

        with st.expander("See header"):
            st.write(f"### header after filtering:")
            st.write(pose.header)

        # st.write(pose.body.data.shape)
        

        
        # st.write(visualize_pose(pose=pose)) # bunch of ndarrays
        # st.write([Image.fromarray(v.cv2.cvtColor(frame, cv_code)) for frame in frames])

        # for i, image in enumerate(images[::n]):
        #     print(f"i={i}")
        #     st.image(image=image, width=width)
        
