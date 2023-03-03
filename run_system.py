#!/usr/bin/env python3

# prerequisites: as described in https://alphacephei.com/vosk/install and also python module `sounddevice` (simply run command `pip install sounddevice`)
# Example usage using Dutch (nl) recognition model: `python test_microphone.py -m nl`
# For more help run: `python test_microphone.py -h`

import argparse
import queue
import sys
import sounddevice as sd
import json
import asyncio
#from punctuator import Punctuator
#p = Punctuator('model.pcl')

from revChatGPT.V1 import Chatbot
from deepmultilingualpunctuation import PunctuationModel


model_punc = PunctuationModel()

from vosk import Model, KaldiRecognizer
#from chatgpt_wrapper import ChatGPT

chatbot = Chatbot(config={
  "email": "example_mail@gmail.com",
  "password": "password"
})

q = queue.Queue()

    
def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "-l", "--list-devices", action="store_true",
    help="show list of audio devices and exit")
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    "-f", "--filename", type=str, metavar="FILENAME",
    help="audio file to store recording to")
parser.add_argument(
    "-d", "--device", type=int_or_str,
    help="input device (numeric ID or substring)")
parser.add_argument(
    "-r", "--samplerate", type=int, help="sampling rate")
parser.add_argument(
    "-m", "--model", type=str, help="language model; e.g. en-us, fr, nl; default is en-us")
args = parser.parse_args(remaining)





try:
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, "input")
        # soundfile expects an int, sounddevice provides a float:
        args.samplerate = int(device_info["default_samplerate"])
        
    if args.model is None:
        model = Model(lang="en-us")
    else:
        model = Model(lang=args.model)

    if args.filename:
        dump_fn = open(args.filename, "wb")
    else:
        dump_fn = None

    final_output = ""
    timeout = 800000
    counter = 0
    with sd.RawInputStream(samplerate=args.samplerate, blocksize = 8000, device=args.device,
            dtype="int16", channels=1, callback=callback):
        print("#" * 80)
        print("Press Ctrl+C to stop the recording")
        print("#" * 80)
       
        
        rec = KaldiRecognizer(model, args.samplerate)
        while True:
            counter += 8000
            #print(counter)
            data = q.get()
            if rec.AcceptWaveform(data):
                output_res = rec.Result()
                output = json.loads(output_res)
                #print(output["text"])
                final_output = final_output + str(output["text"]) + " "
                #print(final_output)
                #output += str(json.loads(rec.Result())["text"]) 
                #print(output)
            #else:
                #print(rec.PartialResult())
            #if dump_fn is not None:
            #    dump_fn.write(data)
            
            if counter*2 > timeout:
                break
            else:
            	continue
                
    
    #print(final_output)
    
    ######################### STT output is stored in final_output and passed to a deepmultilingualpunctuation model to get Punctutations right ######################################
    result = model_punc.restore_punctuation(final_output)
    #print(result)
    
    
    print("Aditya: {0}".format(result))
    #asyncio.coroutine(ask_chat_gpt(token, result))
    
    ######################### After getting a Punctuation output, you need to pass the result into the ChatGPT ######################################
    prev_text = ""

    for data in chatbot.ask(result):
            message = data["message"][len(prev_text) :]
            #print(message)
            print(message, end="", flush=True)
            prev_text = data["message"]
            #print(prev_text)
    print()
    

except Exception as e:
    parser.exit(type(e).__name__ + ": " + str(e))
