import traceback
from flask import Flask, request, jsonify, abort, Response
from flask_cors import CORS
import traceback
import litellm
from util import handle_error
from litellm import completion 
import os, dotenv, time 
import json
import concurrent.futures
import threading
dotenv.load_dotenv()

# TODO: set your keys in .env or here:
# os.environ["OPENAI_API_KEY"] = "" # set your openai key here
# os.environ["ANTHROPIC_API_KEY"] = "" # set your anthropic key here
# os.environ["TOGETHER_AI_API_KEY"] = "" # set your together ai key here
# see supported models / keys here: https://docs.litellm.ai/docs/providers
######### ENVIRONMENT VARIABLES ##########
verbose = True

# litellm.token = "" # get your own - https://admin.litellm.ai/
# litellm.use_client = True
# litellm.caching_with_models = True # CACHING: caching_with_models Keys in the cache are messages + model. - to learn more: https://docs.litellm.ai/docs/caching/

############ HELPER FUNCTIONS ###################################

def print_verbose(print_statement):
    if verbose:
        print(print_statement)

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'received!', 200

def data_generator(response):
    for chunk in response:
        yield f"data: {json.dumps(chunk)}\n\n"

def simple_llm_call(model, messages):
    completion(model=model, messages=messages)
    return 

def batch_model_calls(model_list, messages): 
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_result = {executor.submit(simple_llm_call, model, messages): model for model in model_list}
        for future in concurrent.futures.as_completed(future_to_result):
            try:
                results.append(future.result())
            except Exception as exc:
                print(f'Generated an exception: {exc}')
    return results

@app.route('/chat/completions', methods=["POST"])
def api_completion():
    data = request.json
    if data.get('stream') == "True":
        data['stream'] = True # convert to boolean
    try:
        # GET MODELS
        selected_model = "gpt-3.5-turbo" # by default always return the response from gpt-3.5-turbo

        # PROMPT LOGIC
        if "prompt" not in data:
            data["prompt"] = "What's the weather like in San Francisco?"
        system_prompt = "Only respond to questions about code. Say 'I don't know' to anything outside of that."
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": data.pop("prompt")}]

        # COMPLETION CALL
        response = completion(model=selected_model, messages=messages)

        # CALL REST OF MODELS - to log what their results would have been 
        model_list = [{"model": "facebook/opt-125m", "provider": "openai", "api_base": "http://localhost:8000/v1"}]
        threading.Thread(target=batch_model_calls, args=(model_list, messages)).start()

        if 'stream' in data and data['stream'] == True: # use generate_responses to stream responses
            return Response(data_generator(response), mimetype='text/event-stream')
    except Exception as e:
        # call handle_error function
        print_verbose(f"Got Error api_completion(): {traceback.format_exc()}")
        ## LOG FAILURE
        end_time = time.time() 
        traceback_exception = traceback.format_exc()
        return handle_error(data=data)
    return response

@app.route('/get_models', methods=["POST"])
def get_models():
    try:
        return litellm.model_list
    except Exception as e:
        traceback.print_exc()
        response = {"error": str(e)}
    return response, 200

if __name__ == "__main__":
  from waitress import serve
  serve(app, host="0.0.0.0", port=4000, threads=500)

