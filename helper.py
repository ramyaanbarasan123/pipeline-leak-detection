from ultralytics import YOLO
import streamlit as st
import cv2
from pytube import YouTube
import pandas as pd

import settings


def load_model(model_path):
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):

    image = cv2.resize(image, (720, int(720*(9/16))))

    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)

    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_youtube_video(conf, model):
  
    source_youtube = st.sidebar.text_input("YouTube Video url")


    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))



def play_webcam(conf, model):

    source_webcam = settings.WEBCAM_PATH
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):

    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())


    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def display_dashboard():
    images_l ={
        "Dataset" : "images/dataset.png",
        "Confusion Matrix" : "images/confusion_matrix.png",
        "F1 Curve" : "images/F1_curve.png",
        "R Curve" : "images/R_curve.png",
        "Validation" : "images/batch_val.png",

    }
    images_r ={
        "Training Graph" : "images/training_graph.png",
        "Results" : "images/results.png",
        "P Curve" : "images/P_curve.png",
        "PR Curve" : "images/PR_curve.png",
    }
    col1, col2 = st.columns(2)

    for image_name, image_path in images_l.items():
        with col1:
            with st.expander(f"{image_name}", expanded=False):
                st.image(image_path, caption=f"Image {image_name}", width=604)

    for image_name, image_path in images_r.items():
        with col2:
            with st.expander(f"{image_name}", expanded=False):
                st.image(image_path, caption=f"Image {image_name}", width=604)
    
    df = pd.read_csv("images/results.csv")
    st.subheader("Result CSV Data")
    st.dataframe(df)
               
