from pytube import YouTube
import pandas as pd
from gradio_client import Client


video_url = "https://www.youtube.com/watch?v=QTMhuD0XEcY" 
audio_file = YouTube(video_url).streams.filter(only_audio=True).first().download(filename="audio.mp4")

client = Client("https://sanchit-gandhi-whisper-jax.hf.space/")
result = client.predict(
				"audio.mp4",	# str (filepath or URL to file) in 'inputs' Audio component
				"transcribe",	# str in 'Task' Radio component
				False,	# bool in 'Return timestamps' Checkbox component
				api_name="/predict"
)
str_res = result[0]