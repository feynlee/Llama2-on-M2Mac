# Llama2 on M1/M2 Mac

Some demo scripts for running Llama2 on M1/M2 Macs.

## Installation

Simply run the install script to install Llama2:

```sh
install.sh
```

You could replace the model used with a different one from here: [Llama-2-13B-chat-GGML](https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/tree/main).

## WebUI Demo

Make sure you have `streamlit` and `langchain` installed and then execute the Python script:

```sh
pip install -r requirements.txt
streamlit run chat_with_llama2-WebUI.py
```
