import streamlit as st
import subprocess
from pydub import AudioSegment
import math
import glob
import openai
import os

has_transcript = os.path.exists("./.cache/podcast.txt")

@st.cache_data
def transcribe_chunks(chunk_folder, destination):
    if has_transcript:
        return
    files = glob.glob(f"./{chunk_folder}/*.mp3")
    files.sort()
    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            text_file.write(transcript["text"])

@st.cache_data
def extract_audio_from_video(video_path):
    if has_transcript:
        return
    audio_path = video_path.replace("mp4", "mp3")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        audio_path,
    ]
    subprocess.run(command)

@st.cache_data
def cut_audio_in_chunks(audio_path, chunk_size, chunks_folder):
    if has_transcript:
        return
    track = AudioSegment.from_mp3(audio_path)
    chunk_length = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_length)

    for i in range(chunks):
        start_time = i * chunk_length
        end_time = (i + 1) * chunk_length

        chunk = track[start_time:end_time]

        chunk.export(f"./{chunks_folder}/chunk{i}.mp3", format="mp3")


st.set_page_config(
    page_title="MeetingGPT",
    page_icon="💼",
)

st.markdown("""
    # MeetingGPT
    
    Welcome to MeetingGPT, upload a video and I will give you a transcript, a summary and a chat bot to ask
    any question about it.
    
    Get started by uploading a video file in the sidebar.
""")

with st.sidebar:
    video = st.file_uploader(
        "Video",
        type=["mp4", "avi", "mkv", "mov", ],
    )
if video:
    with st.status("Loading video..."):
        video_path = f"./.cache/{video.name}"
        audio_path = video_path.replace("mp4", "mp3")
        with open(video_path, "wb") as f:
            f.write(video.read())
    with st.status("Extracting audio..."):
        extract_audio_from_video(video_path)
    chunk_folder = "./.cache/chunks"
    with st.status("Cutting audio segments..."):
        cut_audio_in_chunks(audio_path, 10, chunk_folder)
    with st.status("Transcribing audio..."):
        transcript_path = video_path.replace("mp4", "txt")
        transcribe_chunks(chunk_folder, transcript_path)
