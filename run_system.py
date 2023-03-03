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

token = "eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0.._XjROWiqYlKxl-JE.tvPAuyJrny5USQ-USOgPI5esD6vrB-ki0Jb4VUwbnZ-QI-hViB9AyNmG-rg0m-TdBHHFIUx-m_r8rWSla5MxWFaVhlNXJ0dqnJYBgPNWDoEYPHcmUCo_Z2zSzaCBBmxniAuXCW9kAQzLU6OIxd39ywKZdRxDjFY5F_MLYmb2hXl5xouLRBgBj-tgvq3Jm2mEB6qMTDd7V62f6EXg1yEMlVwlDJ5stco5PTwJoAWl-Neln7Er3hX3Le71d5xCjZw6vQWeQ0GwgYtjMUkuL4rGbPX_NXer-gv74mpz6A4_a2c6G4DAidgoMA95UpD9PTJVsn5IvsKUWJfHsizGWvi2gy0S0eEhut3kN1UU6gVsqEbjs0OrhGxFO83xXErwQ2zzj3BsB-QOSm0j0EKS7lhnOzQi8nTCRs-5yyiYNm3GVResXztLyn1_6lStDX9Q1bn7ef-IGPGEbvq2q0Fx5PBP0QTUvbTxTVQjta9VUiZcN32GbhGiSJA0JeAH3TdVCqxksaXy46ED0KKbs4TvrMw0jyETVmUIW7cssdUZy2KLgottcZ6adTMGk4l22Z2ygHJ6Rq0NMrNyFUUMXwNgDNURpIbP0D1kpmtNcxAwNExGKKUt1uHJKGqadz_htpJIEoZeVf99M94MPHLDr5F9-5_IFA15ShDRXDiTVhx69DmUeW6HXer4IzFR51yJSTGkKRTu1oYJ5h2_kqfS0EKeXdHQ8Xb8ha593hUuXz3ogeIqsPRDdxIHzHnzPwjR2ulYPwJk-k-m1mqpfPBA_fXumd2twptkUWgpDm0u5Xdw0FfYEjHS5TQqjPzlyiDmo-pDiIdBg0cVFUGVvh_jL9lp5SKrhMmoixl0zC3uKT5Q1ZAXW6QrMnrSjvoKuD332Bfd0zljDKxUzG0DtMYbBFYbhXLjsG0c6bXcqwnoZq-DoKVOqRIlBl8fuDg96s7-sffAwZyIap0txu3YNogyRYk96q3h4EPPb5e1dP45RSf_x8bo6TLUiPXYB3dGoaTc2U-d37U3DSozBvnkG1BXpk_2NVz9AqK_gxVEEwC4o6tGU0tHDD9y5t1SMh_OLCkLfK9octN8m4BvcPdJUW2_1HbN80rNszCFxW12hXmSsRoRnQzGPWCskqoiC7RLtQjiFlqm9sAm-cv7VaZHrAgyPJ8ZTSWqyVEmAKZ1IbR1RVTgr5wB9hMA75oRMKq9V1Odk-55FQEoWineT0CsKZYKdw2h-EWKIXAL4F29DZtk-uO0a7lcKjmpagBNmBQB6IcSasEouJVR4EyErAX6iIr5_r9z1QS1vJyl7cD6LRV7p_chATyIautha-Th5qA0smusxggVVXHQVv0o94PYz_L06zZGh0GsVdYEZiCIrMMKLXX5MtDTKtEpWGeEmpBrtAxkmBSbtUXz9uIeHWoPcmZpJKT2G__iz9DybkS4_7XQkK6JJbdyWH7ReSKsk0Uf7fT-pNNcalNbjVU0Wc7wqjqKSJzgo03aFf8o-76BrDWiWBDkckiLJ0iRZ7NW2pW-3IYq2OIgpt50PidakdE0JE1oLp2s1N6LQssigjhvFIUzZfYZ1yw92DEUK59RJDH6APIXYtm6UtJdTXsbWLkN2h8dcLefLxNc_5LtD8EgmsOK1anyXKkGOEcBQFTP2UETyCFchiBkBxXQVHKTr06vQFsGZV63dwxP3tNFJgv1T2bwpbo_60bTo2cjGrGdCgG8ZyLwL2nTQfu9BhDgN_oD1D0lud_5R_HayeS5e9s4096BvBUaHHb3InxDAnXSdIQPWa4EclzUWF5w_KpXrUuE06jLtCsBTQFRvhQMKxlBZf-0eETDK30Z30QO6WSoSP-ve8FUxMaN_9bSauYPhM2q4iJ1LRYxoy0kjFIj1IhkKOUVJ3VMzuge-ugX1138pgoAhNS-zvcEkWt2BbBcdNEuLMwR1Ty1J4qxMrkMMg7RWfqIU4GT1lriIV46L726p1k4MQWbMwiOVwqX38zPuIrWIwMzs4vLY4YmOQQsBEVPOwcWW3CnhEC6RkR-t_EiClByl5ZSW41YG7bicbCeI0faDkWS4prnIR_egT2yYeOtaHY43d7bbmu5VA883AEbUigzWNhNZf-7Jy8_Wg8K4I72F3LMHdBYWRuDu3zwavdn8b3BKInINjNAALm134O8ekEgRgljWABwtac1e7hIzXwt3VJRth140rqOKaqgqzsSOVrnqomO_3zNdkImdncWxAwPJmITEyRlRUBgjSdxiSCkz4rERAGCPHsDx7onBxOyYaZdLbhzfyZ87Y9QJYOSkiMxbwXc5eBjvzsISTjN7Xz5U535-N2WRCGCBWlep2P3F_sqRt8qSQsVDjbZINp85jTIQZwE6WfytSeurmtPHmIij0Hs2pbMOFsCPOSLeJaPSCN0UW_rIQ1X7j3aEo9n6TXXMyfc_7NlEp9InRlhNI-AwSLO6Cm3ybKObKbT7UJ3tGkRLe1DdL-X9Vyv6Y2kBuLNbye_uDDl3DV9M6tP8cCjN5j5YY5E38BpTQ0wRF2NSXG-Prvv4W3y6OTE3NxZKp7_jeXTJSRnyCusrvVs5VbYHczH5MaTwh1WUMiXdoMT9nu5Rb3FhyFaRVzwyKpEpijF2MSEbwocpsXAZqwSMeRTAeuArRrqOzG1wQibc8W2_cxF12Vu-HdaIiscGXc3nPiGW21F48IuLbhwaY-cpXSb926l-HkawZo0rj-rBD83yrDvXUVleKPuPn4wxUt65ktqaQSCH5w.stuHWo07ZY9BZQGDol4J_w"


model_punc = PunctuationModel()

from vosk import Model, KaldiRecognizer
from chatgpt_wrapper import ChatGPT

chatbot = Chatbot(config={
  "email": "raikaradityainvestment@gmail.com",
  "password": "Letmein#2605"
})

q = queue.Queue()

def ask_chat_gpt(token, prompt):
    global gpt_response
    chat = Chatbot(token, "https://gpt.pawan.krd")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(chat.wait_for_ready())
    response = loop.run_until_complete(chat.ask(prompt))
    chat.close()
    loop.stop()
    
    gpt_response = response['answer']

    return
    

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
    
    result = model_punc.restore_punctuation(final_output)
    #print(result)
    
    
    print("Aditya: {0}".format(result))
    #asyncio.coroutine(ask_chat_gpt(token, result))
    
    
    prev_text = ""

    for data in chatbot.ask(result):
            message = data["message"][len(prev_text) :]
            #print(message)
            print(message, end="", flush=True)
            prev_text = data["message"]
            #print(prev_text)
    print()
    
    
    
    
   
    
              
    
except KeyboardInterrupt:
    #output = json.loads(rec.FinalResult())

    

    print("\nDone")
    print(final_output)
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ": " + str(e))
