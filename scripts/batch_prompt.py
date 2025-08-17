# This script takes a path to a txt file or a directory of txt files,
# and generates images based on the prompts in those files.

import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import os
import argparse
import re
from tqdm import tqdm

server_address = "127.0.0.1:8188"
client_id = str(uuid.uuid4())

def queue_prompt(prompt, prompt_id):
    p = {"prompt": prompt, "client_id": client_id, "prompt_id": prompt_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    urllib.request.urlopen(req)

def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = str(uuid.uuid4())
    queue_prompt(prompt, prompt_id)
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

def main():
    parser = argparse.ArgumentParser(description='Generate images from prompt files.')
    parser.add_argument('path', type=str, help='Path to a txt file or a directory containing txt files.')
    parser.add_argument('--workflow', type=str, default='workflows/flux_api.json', help='Path to the workflow API JSON file.')
    args = parser.parse_args()

    try:
        with open(args.workflow, 'r') as f:
            prompt_text = f.read()
    except FileNotFoundError:
        print(f"Workflow file not found at {args.workflow}")
        return

    all_txt_files = []
    if os.path.isdir(args.path):
        file_pattern = re.compile(r'_\d{5}\.txt$')
        for root, _, files in os.walk(args.path):
            for file in files:
                if file_pattern.search(file):
                    all_txt_files.append(os.path.join(root, file))
    elif os.path.isfile(args.path) and args.path.endswith('.txt'):
        all_txt_files.append(args.path)
    else:
        print("Invalid path. Please provide a valid txt file or a directory.")
        return

    print(f"Found {len(all_txt_files)} total matching .txt files.")

    txt_files_to_process = []
    for txt_file in all_txt_files:
        output_dir = os.path.dirname(txt_file)
        base_name = os.path.splitext(os.path.basename(txt_file))[0]
        
        output_exists = False
        if os.path.isdir(output_dir):
            for fname in os.listdir(output_dir):
                if fname.startswith(base_name) and fname.endswith('.png'):
                    output_exists = True
                    break
        
        if not output_exists:
            txt_files_to_process.append(txt_file)

    print(f"Found {len(txt_files_to_process)} files to process after checking for existing outputs.")
    
    if not txt_files_to_process:
        return

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    
    for txt_file in tqdm(txt_files_to_process, desc="Processing prompts"):
        with open(txt_file, 'r') as f:
            prompt_from_file = f.readline().strip()

        prompt = json.loads(prompt_text)
        # Set the text prompt for our positive CLIPTextEncode
        prompt["6"]["inputs"]["text"] = prompt_from_file
        
        # Set the seed for our KSampler node to a random value
        # prompt["3"]["inputs"]["seed"] = uuid.uuid4().int & (1<<32)-1

        images = get_images(ws, prompt)
        
        output_dir = os.path.dirname(txt_file)
        for i, (node_id, image_list) in enumerate(images.items()):
            for j, image_data in enumerate(image_list):
                # Use the txt file name as the base for the output image name
                base_name = os.path.splitext(os.path.basename(txt_file))[0]
                output_filename = f"{base_name}_{i}_{j}.png"
                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, 'wb') as img_file:
                    img_file.write(image_data)
                print(f"Saved image to {output_path}")

    ws.close()


if __name__ == "__main__":
    main()
