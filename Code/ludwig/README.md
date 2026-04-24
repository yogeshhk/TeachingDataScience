# Ludwig

Low/No code platform for running and fine-tuning LLMs apart from the usual ML/DL workflows, all config based.

[My Slack](https://app.slack.com/client/T01PN6M1TKK/C01PN6M2RSM)

## Installation on Windows
On Windows, installation appears to be a bit sensitive ... here is a sequence that seems to be working..

```
conda create -n ludwig python=3.10
conda activate ludwig
conda install cuda -c nvidia
conda install cudatoolkit
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Check `import torch; print(torch.cuda.is_available())` Must be True
# LD_LIBRARY_PATH has paths to <cuda version>/lib x64 and Win32
# pip install bitsandbyes==0.40.2 gives error 'CUDA Setup failed despite GPU being available' need windows version
# pip install bitsandbytes-windows gives error 'no attribute 'cuDeviceGetCount''
# pip install git+https://github.com/Keith-Hon/bitsandbytes-windows gives same error

pip install bitsandbytes==0.40.2 --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
pip install ludwig
```

## Vertex AI
To run on Google Cloud, to utize the credits that I have, steps are:
- Go to Vertex AI, Find `Colab Enterprise` there.
- In `Runtime Template` find existing GPU based template or create one, with `n1-std-8` with either T4 or VT100 GPUs, say 4.
- Based on the template, create `runtime`, give name.
- Create or import notebook, change runtime to the recently created one.
- After experiment is over, download the notebook along with outputs.
- Shutdown the notebook, delete the runtime, so that billing does not continue.

## References
- [Official Website](https://ludwig.ai/latest/)
- [Mistral Finetuning on text 2 text](https://predibase.com/blog/fine-tuning-mistral-7b-on-a-single-gpu-with-ludwig)
- Ludwig AI: The Easiest Way To Train A Custom LLM [Part 1](https://medium.com/mlearning-ai/ludwig-ai-the-easiest-way-to-train-a-custom-llm-part-1-49c7fc134ebc), [Part II](https://medium.com/mlearning-ai/ludwig-ai-the-easiest-way-to-train-a-custom-llm-part-2-caf2235f0689)
- [No More Hard Coding: Use Declarative Configuration to Build and Fine-tune Custom Mistral 7B on Your Data](https://levelup.gitconnected.com/no-more-hard-coding-use-declarative-configuration-to-build-and-fine-tune-custom-llms-on-your-data-6418b243fad7)
- [ML NLP Tutorials](https://www.youtube.com/playlist?list=PL_lyFNnob30u8h9DPXQOyJQ9nGbqwYmZK)
- [Efficient Fine-Tuning for Llama-v2-7b on a Single GPU](https://www.youtube.com/watch?v=g68qlo9Izf0)
- [Efficient Fine-Tuning of LLMs on single T4 GPU using Ludwig](https://community.analyticsvidhya.com/c/datahour/efficient-fine-tuning-of-llms-on-single-t4-gpu-using-ludwig)
- Ludwig 0.8: Hands On Webinar [Notebook](https://colab.research.google.com/drive/1lB4ALmEyvcMycE3Mlnsd7I3bc0zxvk39#scrollTo=xb1aLHZRFrwA)