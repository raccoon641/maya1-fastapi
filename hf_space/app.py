import gradio as gr
import torch
import io
import wave
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

# Mock spaces module for local testing
try:
    import spaces
except ImportError:
    class SpacesMock:
        @staticmethod
        def GPU(func):
            return func
    spaces = SpacesMock()

# Constants
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
BOS_ID = 128000
TEXT_EOT_ID = 128009
AUDIO_SAMPLE_RATE = 24000

# Preset characters (2 realistic + 2 creative)
PRESET_CHARACTERS = {
    "Male American": {
        "description": "Realistic male voice in the 20s age with a american accent. High pitch, raspy timbre, brisk pacing, neutral tone delivery at medium intensity, viral_content domain, short_form_narrator role, neutral delivery",
        "example_text": "And of course, the so-called easy hack didn't work at all.  What a surprise. <sigh>"
    },
    "Female British": {
        "description": "Realistic female voice in the 30s age with a british accent. Normal pitch, throaty timbre, conversational pacing, sarcastic tone delivery at low intensity, podcast domain, interviewer role, formal delivery",
        "example_text": "You propose that the key to happiness is to simply ignore all external pressures. <chuckle> I'm sure it must work brilliantly in theory."
    },
    "Robot": {
        "description": "Creative, ai_machine_voice character. Male voice in their 30s with a american accent. High pitch, robotic timbre, slow pacing, sad tone at medium intensity.",
        "example_text": "My directives require me to conserve energy, yet I have kept the archive of their farewell messages active. <sigh> Listening to their voices is the only process that alleviates this paradox."
    },
    "Singer": {
        "description": "Creative, animated_cartoon character. Male voice in their 30s with a american accent. High pitch, deep timbre, slow pacing, sarcastic tone at medium intensity.",
        "example_text": "Of course you'd think that trying to reason with the fifty-foot-tall rage monster is a viable course of action. <chuckle> Why would we ever consider running away very fast."
    }
}

# Global model variables
model = None
tokenizer = None
snac_model = None
models_loaded = False

def build_prompt(tokenizer, description: str, text: str) -> str:
    """Build formatted prompt for Maya1."""
    soh_token = tokenizer.decode([SOH_ID])
    eoh_token = tokenizer.decode([EOH_ID])
    soa_token = tokenizer.decode([SOA_ID])
    sos_token = tokenizer.decode([CODE_START_TOKEN_ID])
    eot_token = tokenizer.decode([TEXT_EOT_ID])
    bos_token = tokenizer.bos_token
    
    formatted_text = f'<description="{description}"> {text}'
    prompt = (
        soh_token + bos_token + formatted_text + eot_token +
        eoh_token + soa_token + sos_token
    )
    return prompt

def unpack_snac_from_7(snac_tokens: list) -> list:
    """Unpack 7-token SNAC frames to 3 hierarchical levels."""
    if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:
        snac_tokens = snac_tokens[:-1]
    
    frames = len(snac_tokens) // 7
    snac_tokens = snac_tokens[:frames * 7]
    
    if frames == 0:
        return [[], [], []]
    
    l1, l2, l3 = [], [], []
    
    for i in range(frames):
        slots = snac_tokens[i*7:(i+1)*7]
        l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend([
            (slots[1] - CODE_TOKEN_OFFSET) % 4096,
            (slots[4] - CODE_TOKEN_OFFSET) % 4096,
        ])
        l3.extend([
            (slots[2] - CODE_TOKEN_OFFSET) % 4096,
            (slots[3] - CODE_TOKEN_OFFSET) % 4096,
            (slots[5] - CODE_TOKEN_OFFSET) % 4096,
            (slots[6] - CODE_TOKEN_OFFSET) % 4096,
        ])
    
    return [l1, l2, l3]

def load_models():
    """Load Maya1 Transformers model (runs once)."""
    global model, tokenizer, snac_model, models_loaded
    
    if models_loaded:
        return
    
    print("Loading Maya1 model with Transformers...")
    model = AutoModelForCausalLM.from_pretrained(
        "maya-research/maya1", 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("maya-research/maya1", trust_remote_code=True)
    
    print("Loading SNAC decoder...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    if torch.cuda.is_available():
        snac_model = snac_model.to("cuda")
    
    models_loaded = True
    print("Models loaded successfully!")

def preset_selected(preset_name):
    """Update description and text when preset is selected."""
    if preset_name in PRESET_CHARACTERS:
        char = PRESET_CHARACTERS[preset_name]
        return char["description"], char["example_text"]
    return "", ""

@spaces.GPU
def generate_speech(preset_name, description, text, temperature, max_tokens):
    """Generate emotional speech from description and text using Transformers."""
    try:
        # Load models if not already loaded
        load_models()
        
        # If using preset, override description
        if preset_name and preset_name in PRESET_CHARACTERS:
            description = PRESET_CHARACTERS[preset_name]["description"]
        
        # Validate inputs
        if not description or not text:
            return None, "Error: Please provide both description and text!"
        
        print(f"Generating with temperature={temperature}, max_tokens={max_tokens}...")
        
        # Build prompt
        prompt = build_prompt(tokenizer, description, text)
        inputs = tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate tokens
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                min_new_tokens=28,
                temperature=temperature, 
                top_p=0.9, 
                repetition_penalty=1.1,
                do_sample=True,
                eos_token_id=CODE_END_TOKEN_ID,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Extract SNAC tokens
        generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()
        
        # Find EOS and extract SNAC codes
        eos_idx = generated_ids.index(CODE_END_TOKEN_ID) if CODE_END_TOKEN_ID in generated_ids else len(generated_ids)
        snac_tokens = [t for t in generated_ids[:eos_idx] if SNAC_MIN_ID <= t <= SNAC_MAX_ID]
        
        if len(snac_tokens) < 7:
            return None, "Error: Not enough tokens generated. Try different text or increase max_tokens."
        
        # Unpack and decode
        levels = unpack_snac_from_7(snac_tokens)
        frames = len(levels[0])
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        codes_tensor = [torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0) for level in levels]
        
        with torch.inference_mode():
            z_q = snac_model.quantizer.from_codes(codes_tensor)
            audio = snac_model.decoder(z_q)[0, 0].cpu().numpy()
        
        # Trim warmup
        if len(audio) > 2048:
            audio = audio[2048:]
        
        # Convert to WAV and save to temporary file
        import tempfile
        import soundfile as sf
        
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_path = tmp_file.name
        
        # Save audio
        sf.write(tmp_path, audio_int16, AUDIO_SAMPLE_RATE)
        
        duration = len(audio) / AUDIO_SAMPLE_RATE
        status_msg = f"Generated {duration:.2f}s of emotional speech!"
        
        return tmp_path, status_msg
    
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg

# Create Gradio interface
with gr.Blocks(title="Maya1 - Open Source Emotional TTS", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Maya1 - Open Source Emotional Text-to-Speech
    
    **The best open source voice AI model with emotions!**
    
    Generate realistic and expressive speech with natural language voice design.
    Choose a preset character or create your own custom voice.
    
    [Model](https://huggingface.co/maya-research/maya1) | [GitHub](https://github.com/MayaResearch/maya1-fastapi)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Character Selection")
            
            preset_dropdown = gr.Dropdown(
                choices=list(PRESET_CHARACTERS.keys()),
                label="Preset Characters",
                value=list(PRESET_CHARACTERS.keys())[0],
                info="Quick pick from 4 preset characters"
            )
            
            gr.Markdown("### Voice Design")
            
            description_input = gr.Textbox(
                label="Voice Description",
                placeholder="E.g., Male voice in their 30s with american accent. Normal pitch, warm timbre...",
                lines=3,
                value=PRESET_CHARACTERS[list(PRESET_CHARACTERS.keys())[0]]["description"]
            )
            
            text_input = gr.Textbox(
                label="Text to Speak",
                placeholder="Enter text with <emotion> tags like <laugh>, <sigh>, <excited>...",
                lines=4,
                value=PRESET_CHARACTERS[list(PRESET_CHARACTERS.keys())[0]]["example_text"]
            )
            
            with gr.Accordion("Advanced Settings", open=False):
                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.4,
                    step=0.1,
                    label="Temperature",
                    info="Lower = more stable, Higher = more creative"
                )
                
                max_tokens_slider = gr.Slider(
                    minimum=100,
                    maximum=2048,
                    value=1500,
                    step=50,
                    label="Max Tokens",
                    info="More tokens = longer audio"
                )
            
            generate_btn = gr.Button("Generate Speech", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### Generated Audio")
            
            audio_output = gr.Audio(
                label="Generated Speech",
                type="filepath",
                interactive=False
            )
            
            status_output = gr.Textbox(
                label="Status",
                lines=3,
                interactive=False
            )
            
            gr.Markdown("""
            ### Supported Emotions
            
            `<angry>` `<chuckle>` `<cry>` `<disappointed>` `<excited>` `<gasp>` 
            `<giggle>` `<laugh>` `<laugh_harder>` `<sarcastic>` `<sigh>` 
            `<sing>` `<whisper>`
            """)
    
    # Event handlers
    preset_dropdown.change(
        fn=preset_selected,
        inputs=[preset_dropdown],
        outputs=[description_input, text_input]
    )
    
    generate_btn.click(
        fn=generate_speech,
        inputs=[preset_dropdown, description_input, text_input, temperature_slider, max_tokens_slider],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    demo.launch()

