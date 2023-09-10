from pytube import YouTube, extract
from gradio_client import Client
from youtube_transcript_api import YouTubeTranscriptApi


def transcribe_video(video_url):
    # Download audio from the given video URL
    yt_id = extract.video_id(video_url)
    if YouTubeTranscriptApi.list_transcripts(yt_id).find_transcript(['en']):
        return "\n".join([i['text'] for i in YouTubeTranscriptApi.get_transcript(yt_id)])
    
    
    audio_file = YouTube(video_url).streams.filter(only_audio=True).first().download(filename="audio.mp4")

    # Initialize Gradio Client
    client = Client("https://sanchit-gandhi-whisper-jax.hf.space/")

    # Predict transcription using Gradio Client
    result = client.predict(
        "audio.mp4",    # str (filepath or URL to file) in 'inputs' Audio component
        "transcribe",    # str in 'Task' Radio component
        False,           # bool in 'Return timestamps' Checkbox component
        api_name="/predict"
    )

    # Extract the transcription from the result
    str_res = result[0]
    
    return str_res