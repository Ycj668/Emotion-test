
from openai import OpenAI
import httpx,os
import pyttsx3
import pygame
import threading

def create_chat_completion(user_message):
    os.environ["http_proxy"] = "http://localhost:7890"
    os.environ["https_proxy"] = "http://localhost:7890"

    client = OpenAI(
        api_key="sk-2a0yFGJGm3EbUrjNnI7jTOGvVkOYkFFKOWaKQluqQHJ4vNO0",

        base_url="https://api.moonshot.cn/v1",
    )

    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system", "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，"},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3,
        max_tokens=70,
    )
    song = completion.choices[0].message.content
    song_text = wrap_text_by_words(song, 5)
    lines = [line.strip() for line in song_text.split('\n')]
    processed_text = '\n'.join(lines)
    print(processed_text)
    engine = pyttsx3.init()
    engine.setProperty('rate', 120)
    engine.setProperty('volume', 0.9)


    sentences = song.split('. ')
    for i, sentence in enumerate(sentences):
        if i < len(sentences) - 1:
            sentences[i] = sentence + '?'

    song_to_speak = '. '.join(sentences)

    engine.say(song_to_speak)
    engine.runAndWait()  # 等待语音播放完成

    return None

def play_background_music(idx):

    pygame.mixer.init()
    if idx == 0:
        music_file = 'D:\代码\emotion-offline\BGM\搞笑.mp3'
    elif idx == 1:
        music_file = 'D:\代码\emotion-offline\BGM\欢乐.mp3'
    elif idx == 2:
        music_file = 'D:\代码\emotion-offline\BGM\激励.mp3'
    else:
        music_file = 'D:\代码\emotion-offline\BGM\舒缓.mp3'
    pygame.mixer.music.load(music_file)
    pygame.mixer.music.set_volume(0.3)
    pygame.mixer.music.play()
    pygame.mixer.music.play(-1)

    return None

def stop_background_music():
    pygame.mixer.music.stop()

    return None

def run(idx, user_message):
    play_background_music(idx, )
    speak_thread = threading.Thread(target=create_chat_completion, args=(user_message,))  # 使用线程  防止堵塞
    speak_thread.start()
    speak_thread.join()
    stop_background_music()
    pygame.mixer.quit()

    return None


def wrap_text_by_words(text, words_per_line):
    words = text.split()
    wrapped_text = ''
    current_line = ''

    for word in words:
        if len(current_line.split()) < words_per_line:
            current_line += ' ' + word
        else:
            wrapped_text += current_line + '\n'
            current_line = word
    if current_line:
        wrapped_text += current_line

    return wrapped_text



