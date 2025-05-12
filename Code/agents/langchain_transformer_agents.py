# Reference: Transformers Agent - Is this Hugging Face's LangChain Competitor? Sam Witteveen
# https://colab.research.google.com/drive/1HGpp1OI-o_ppHi2bHZsvV6QX9k5gsTIK?usp=sharing

# Assuming OPENAI_API_KEY set in Environment variables

from langchain.llms import OpenAI, HuggingFaceHub
from langchain import PromptTemplate
from langchain import LLMChain
import pandas as pd
from colorama import Fore
from gtts import gTTS
from playsound import playsound
from transformers.tools import HfAgent
from transformers.tools import OpenAiAgent
import requests
from PIL import Image
from transformers import Tool
from huggingface_hub import list_models
from transformers.tools import HfAgent
import os


class CatImageFetcher(Tool):
    name = "cat_fetcher"
    description = (
        "This is a tool that fetches an actual image of a cat online. It takes no input, and returns the image of a cat.")

    inputs = []
    outputs = ["text"]

    def __call__(self):
        return Image.open(requests.get('https://cataas.com/cat', stream=True).raw).resize((256, 256))


def play_audio(audio):
    myobj = gTTS(text=audio, lang='en', slow=False)
    myobj.save("speech_converted.mp3")
    playsound("speech_converted.mp3")


def show_image(img):
    img.show()


# So far we've been using the tools that the agent has access to. These tools are the following:
#
# Document question answering: given a document (such as a PDF) in image format, answer a question on this document (
# Donut) Text question answering: given a long text and a question, answer the question in the text (Flan-T5) 
# Unconditional image captioning: Caption the image! (BLIP) Image question answering: given an image, 
# answer a question on this image (VILT) Image segmentation: given an image and a prompt, output the segmentation 
# mask of that prompt (CLIPSeg) Speech to text: given an audio recording of a person talking, transcribe the speech 
# into text (Whisper) Text to speech: convert text to speech (SpeechT5) Zero-shot text classification: given a text 
# and a list of labels, identify to which label the text corresponds the most (BART) Text summarization: summarize a 
# long text in one or a few sentences (BART) Translation: translate the text into a given language (NLLB) We also 
# support the following community-based tools:
#
# Text downloader: to download a text from a web URL
# Text to image: generate an image according to a prompt, leveraging stable diffusion
# Image transformation: transforms an image
# We can therefore use a mix and match of different tools by explaining in natural language what we would like to do.

def chat_session1(agent):
    agent.prepare_for_new_chat()

    print(agent.chat("who is the possible new twitter ceo based on this article at " +
                     "https://techcrunch.com/2023/05/11/elon-musk-says-he-has-found-a-new-ceo-for-twitter/"))

    print(agent.chat("who was the former twitter ceo based on that article"))
    print(agent.chat("who is the current twitter ceo based on that article"))
    print(agent.chat("Translate the title of that article into French"))
    print(agent.chat("Summarize that article for me"))


def main():
    agent = OpenAiAgent(model="text-davinci-003")
    print("OpenAI is initialized 💪")

    # cat = agent.run("Generate an image of a royal bengal tiger sitting down resting")
    # show_image(cat)

    # ToDO: not downloading from the given site
    # audio = agent.run("Read out loud the summary of ndtv.com", return_code=True)
    # print(audio)
    # play_audio(audio)

    # .run does not keep memory across runs, but performs better for multiple operations at once
    # .chat keeps memory across runs, but performs better at single instructions
    cat = agent.chat("Show me an image of a panther cat")
    show_image(cat)

    cat = agent.chat("Transform the image so that the background is in the snow")
    show_image(cat)

    chat_session1(agent)


if __name__ == "__main__":
    main()
