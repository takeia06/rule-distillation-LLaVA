import argparse
import json, os
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image
from transformers import BitsAndBytesConfig # 8-bit量子化モデルロード用

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True, help="Path to the LLaVA base model (e.g., llava-hf/llava-1.5-7b-hf)")
parser.add_argument('--model_path', default=None, type=str, help="Path to the fine-tuned or distilled LLaVA model checkpoint. If None, use base_model directly.")
parser.add_argument('--data_file', default=None, type=str, help="A JSON file that contains inputs for inference.")
parser.add_argument('--image_base_path', default="", type=str, help="Base path for image files.")
parser.add_argument('--predictions_file', default='./predictions.json', type=str, help="Output file for predictions.")
parser.add_argument('--gpus', default="0", type=str, help="GPU to use (e.g., '0' or '0,1').")
parser.add_argument('--only_cpu', action='store_true', help='Only use CPU for inference.')
parser.add_argument('--max_new_tokens', type=int, default=128, help="Maximal generated tokens.")
parser.add_argument('--temperature', type=float, default=0.1, help="Generation temperature.")
parser.add_argument('--top_p', type=float, default=0.75, help="Generation top_p.")
parser.add_argument('--top_k', type=int, default=40, help="Generation top_k.")
parser.add_argument('--num_beams', type=int, default=1, help="Number of beams for beam search. Set to 1 for greedy decoding.")
parser.add_argument('--prompt_template', default='alpaca', type=str, help="The prompt template to use.")

args = parser.parse_args()

if args.only_cpu:
    args.gpus = ""
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

def generate_response(model, processor, data_point, device, max_new_tokens, temperature, top_p, top_k, num_beams):
    instruction = data_point.get("instruction", "")
    input_image_path_relative = data_point.get("input", "")

    image_path = os.path.join(args.image_base_path, input_image_path_relative)

    print(f"Attempting to load image from: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}. Skipping this data point.")
        return "Image not found."

    if not instruction:
        instruction = "Is there a crack in this road image? Output 'Abnormal' if there is a crack, otherwise 'No abnormality'."

    # LLaVA-1.5のプロンプトフォーマット
    prompt = "A chat between a curious user and an AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    chat_text = f"{prompt}\nUSER: <image>\n{instruction}\nASSISTANT:"
    print(f"Processing chat text: {chat_text}")

    try:
        # プロセッサを使って画像とテキストを一緒に処理
        inputs = processor(
            images=image,
            text=chat_text,
            return_tensors="pt",
            add_special_tokens=True,
        )

        # 必要に応じて画像特徴量をfloat16に変換
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

        # デバイスに移動
        inputs = {k: v.to(device=device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}

    except Exception as e:
        print(f"Error during input processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Error during input processing."

    with torch.no_grad():
        try:
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "use_cache": True,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "eos_token_id": processor.tokenizer.eos_token_id,
            }

            if temperature > 0 and num_beams == 1:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p
                gen_kwargs["top_k"] = top_k
            else:
                gen_kwargs["do_sample"] = False

            output_ids = model.generate(
                **inputs,
                **gen_kwargs
            )
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Error during generation."

    full_decoded_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=False)
    print(f"--- Full Decoded Text ---")
    print(full_decoded_text)
    print(f"-------------------------")

    response_start_marker = "ASSISTANT:"
    if response_start_marker in full_decoded_text:
        response_raw = full_decoded_text.split(response_start_marker, 1)[1].strip()
    else:
        print(f"Warning: '{response_start_marker}' not found in decoded text. Returning full decoded text.")
        response_raw = full_decoded_text.strip()

    response_raw = response_raw.replace("</s>", "").strip()

    return response_raw

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() and not args.only_cpu else 'cpu'

    model_path = args.model_path if args.model_path else args.base_model

    processor = AutoProcessor.from_pretrained(
        model_path,
        use_fast=False,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        quantization_config=quantization_config,
    )

    model.eval()

    if not args.data_file:
        print("Please provide a --data_file for inference.")
        exit()

    with open(args.data_file, 'r', encoding='utf-8') as f:
        examples = json.load(f)

    print("Start inference.")
    results = []
    for index, example in enumerate(examples):
        predicted_output = generate_response(
            model, processor, example, device, args.max_new_tokens, args.temperature,
            args.top_p, args.top_k, args.num_beams
        )

        results.append({
            "instruction": example.get("instruction", ""),
            "input": example.get("input", ""),
            "original_output": example.get("output", ""),
            "predicted_output": predicted_output
        })

        print(f"======={index}=======")
        print(f"Instruction: {example.get('instruction', '')}\n")
        print(f"Input: {example.get('input', '')}\n")
        print(f"Original Output: {example.get('output', '')}\n")
        print(f"Predicted Output: {predicted_output}\n")

    dirname = os.path.dirname(args.predictions_file)
    os.makedirs(dirname, exist_ok=True)
    with open(args.predictions_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("Finish inference.")