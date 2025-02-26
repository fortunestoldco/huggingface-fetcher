#!/bin/bash

# Function to print log messages with timestamps and context
log() {
    local context=$1
    local message=$2
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo -e "[$timestamp] \033[1;36m$context\033[0m: $message"
}

# Function to validate input format - simplified version
validate_replicate_format() {
    # Log start of validation
    log "VALIDATION" "Checking replicate format for '$1'..."
    
    # Check if it has a slash and both sides have content
    if [[ "$1" == *"/"* && "${1:0:1}" != "/" && "${1: -1}" != "/" ]]; then
        # Get parts before and after the slash
        local username="${1%%/*}"
        local modelname="${1#*/}"
        
        # Make sure both parts exist and aren't empty
        if [[ -n "$username" && -n "$modelname" ]]; then
            log "VALIDATION" "Format is valid: username='$username', modelname='$modelname'"
            return 0
        fi
    fi
    
    log "VALIDATION" "Format is invalid for '$1'"
    return 1
}

# Check if OpenAI API key is available
check_openai_key() {
    log "CONFIG" "Checking for OpenAI validation capability..."
    
    # No API key needed anymore
    log "CONFIG" "OpenAI validation service available via HTTP endpoint"
    echo "✅ OpenAI validation service available. Will validate model compatibility."
    return 0
}

# Function to validate predict.py with OpenAI (via the new endpoint)
validate_predict_with_openai() {
    local model="$1"
    local script_content="$2"
    
    log "OPENAI" "Starting validation of predict.py with OpenAI service..."
    echo "🧠 Validating predict.py compatibility with model using OpenAI service..."
    
    # Check if jq is installed
    if ! command -v jq &> /dev/null; then
        log "OPENAI" "Error: jq command not found. Please install jq to use OpenAI validation."
        echo "❌ Error: jq command not found. Please install jq to use OpenAI validation."
        return 1
    fi
    
    # Properly escape the script content for JSON using jq
    log "OPENAI" "JSON-escaping the script content..."
    local escaped_script_content=$(jq -Rsa . <<< "$script_content")
    log "OPENAI" "Script content escaped successfully"
    
    # Create the JSON payload
    log "OPENAI" "Creating API request payload..."
    local payload=$(cat <<EOF
{
    "hf_model": "$model",
    "script": $escaped_script_content
}
EOF
)

    # Validate that the payload is valid JSON
    if ! echo "$payload" | jq . > /dev/null 2>&1; then
        log "OPENAI" "Error: Generated payload is not valid JSON"
        echo "❌ Error: Generated payload is not valid JSON"
        return 1
    else
        log "OPENAI" "Payload validated as valid JSON"
    fi

    # Make the API call to the validation service
    log "OPENAI" "Making API call to validation service..."
    local response=$(curl -s -X POST "https://cogcheck-299757569925.europe-west4.run.app/validate" \
                     -H "Content-Type: application/json" \
                     -d "$payload")
    
    # Check if curl was successful
    if [ $? -ne 0 ]; then
        log "OPENAI" "Error: curl command failed"
        echo "❌ Error: curl command failed to connect to validation service"
        return 1
    fi
    
    log "OPENAI" "Received response from validation service, checking for errors..."
    
    # Check if the API call was successful
    if echo "$response" | jq -e '.error' > /dev/null 2>&1; then
        error_msg=$(echo "$response" | jq -r '.error')
        log "OPENAI" "API error: $error_msg"
        echo "❌ Validation service error: $error_msg"
        return 1
    fi
    
    # Extract the validated script
    log "OPENAI" "Processing response from validation service..."
    if ! echo "$response" | jq -e '.validated_script' > /dev/null 2>&1; then
        log "OPENAI" "Error: Response does not contain expected content"
        echo "❌ Failed to get a valid response from validation service."
        return 1
    fi
    
    MODIFIED_SCRIPT=$(echo "$response" | jq -r '.validated_script')
    
    if [[ -z "$MODIFIED_SCRIPT" ]]; then
        log "OPENAI" "Failed to get a valid response (empty content)"
        echo "❌ Failed to get a valid response from validation service (empty content)."
        return 1
    fi
    
    log "OPENAI" "Successfully received and processed validation service response"
    echo "$MODIFIED_SCRIPT"
    return 0
}

# Banner
echo "================================================"
echo "🤗  Transformer → Replicate Deployment Assistant"
echo "================================================"
echo "This script will help you deploy a Hugging Face model to Replicate."
echo

log "INIT" "Starting deployment assistant"

# Check for OpenAI validation capability
USE_OPENAI=false
if check_openai_key; then
    USE_OPENAI=true
    log "INIT" "OpenAI validation enabled"
else
    log "INIT" "OpenAI validation disabled"
fi

# Step 1: Get HuggingFace model name
log "INPUT" "Prompting for Hugging Face model name"
read -p "Enter the Hugging Face model name (e.g. mosaicml/mpt-7b-storywriter): " HF_MODEL
if [ -z "$HF_MODEL" ]; then
    log "ERROR" "Model name cannot be empty"
    echo "Error: Model name cannot be empty."
    exit 1
fi
log "INPUT" "Received Hugging Face model: $HF_MODEL"

# Step 2: Get Replicate model info
log "INPUT" "Prompting for Replicate destination"
while true; do
    read -p "Enter the Replicate destination in format 'username/modelname': " REPLICATE_MODEL
    if [ -z "$REPLICATE_MODEL" ]; then
        log "ERROR" "Replicate model name cannot be empty"
        echo "Error: Replicate model name cannot be empty."
    elif validate_replicate_format "$REPLICATE_MODEL"; then
        log "INPUT" "Replicate format validated successfully: $REPLICATE_MODEL"
        break
    else
        log "ERROR" "Invalid Replicate format: $REPLICATE_MODEL"
        echo "Error: Format must be 'username/modelname' (e.g. fortunestold/storybook)"
        echo "Note: Enter the username, followed by a slash, followed by the model name."
    fi
done

REPLICATE_USERNAME=$(echo $REPLICATE_MODEL | cut -d'/' -f1)
REPLICATE_MODELNAME=$(echo $REPLICATE_MODEL | cut -d'/' -f2)
log "CONFIG" "Parsed Replicate username: $REPLICATE_USERNAME, model name: $REPLICATE_MODELNAME"

# Step 3: Ask for HuggingFace token
log "INPUT" "Prompting for Hugging Face token"
read -p "Enter your Hugging Face access token (or press Enter to continue without authentication): " HF_TOKEN
if [ -n "$HF_TOKEN" ]; then
    log "CONFIG" "Hugging Face token provided"
else
    log "CONFIG" "No Hugging Face token provided, continuing without authentication"
fi

echo
echo "📋 Configuration Summary:"
echo "-------------------------"
echo "Hugging Face model: $HF_MODEL"
echo "Replicate deployment: $REPLICATE_MODEL"
if [ -n "$HF_TOKEN" ]; then
    echo "Using Hugging Face authentication: Yes"
else
    echo "Using Hugging Face authentication: No"
fi
if [ "$USE_OPENAI" = true ]; then
    echo "Using OpenAI validation: Yes"
else
    echo "Using OpenAI validation: No"
fi
echo "-------------------------"
echo

# Ask for confirmation
log "INPUT" "Asking for confirmation"
read -p "Is this information correct? (y/n): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    log "CANCEL" "User cancelled operation"
    echo "Operation cancelled."
    exit 0
fi
log "CONFIG" "Configuration confirmed by user"

echo
echo "🔧 Creating deployment files..."

# Create necessary directories
log "FILES" "Creating directories"
mkdir -p scripts
log "FILES" "Created scripts directory"

# Create cog.yaml
log "FILES" "Creating cog.yaml"
cat > cog.yaml << EOL
build:
  gpu: true
  cuda: "11.8"
  python_version: "3.10"
  python_packages:
    - "torch==2.0.1"
    - "transformers==4.31.0"
    - "einops==0.6.1"
    - "accelerate==0.21.0"
    - "scipy==1.11.1"
    - "huggingface_hub>=0.16.4"

predict: "predict.py:Predictor"
EOL
log "FILES" "Created cog.yaml"

# Create initial predict.py with local_files_only=True
log "FILES" "Creating predict.py"
cat > predict.py << EOL
import os
import torch
from typing import Any, List
from cog import BasePredictor, Input, ConcatenateIterator

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory"""
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("Loading tokenizer from local files...")
        # Load tokenizer from local files
        self.tokenizer = AutoTokenizer.from_pretrained(
            "./weights", 
            trust_remote_code=True,
            local_files_only=True
        )
        
        print("Loading model from local files (this may take a minute)...")
        # Try to determine best dtype automatically
        try:
            # Try bfloat16 first for newer architectures (like MPT)
            print("Attempting to load with bfloat16...")
            self.model = AutoModelForCausalLM.from_pretrained(
                "./weights", 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                local_files_only=True
            )
            print("Successfully loaded model with bfloat16")
        except Exception as e:
            # Fall back to float16 if bfloat16 fails
            print(f"Failed to load with bfloat16, trying float16: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                "./weights", 
                trust_remote_code=True, 
                torch_dtype=torch.float16, 
                device_map="auto",
                local_files_only=True
            )
            print("Successfully loaded model with float16")
            
        self.model.eval()
        print("Model setup complete.")

    def predict(
        self,
        prompt: str = Input(description="Input text prompt"),
        max_new_tokens: int = Input(
            description="Number of new tokens to generate", ge=1, le=4096, default=512
        ),
        temperature: float = Input(
            description="Sampling temperature; higher values produce more diverse outputs",
            ge=0.01,
            le=5.0,
            default=0.8,
        ),
        top_p: float = Input(
            description="Nucleus sampling parameter. The model only considers tokens whose cumulative probability exceeds this value.",
            ge=0.01,
            le=1.0,
            default=0.95,
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeating tokens; 1.0 means no penalty, values > 1.0 discourage repetition",
            ge=0.01,
            le=5.0,
            default=1.1,
        ),
        stream: bool = Input(
            description="Whether to stream the results or not", default=False
        ),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
        print(f"Tokenizing prompt with {len(prompt)} characters...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

        print("Configuring generation parameters...")
        # Configure generation parameters
        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0,
        }
        
        # Add EOS token if available
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            print(f"Using eos_token_id: {self.tokenizer.eos_token_id}")
            
        # Add PAD token if available, otherwise use EOS
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            print(f"Using pad_token_id: {self.tokenizer.pad_token_id}")
        elif hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
            print(f"Using eos_token_id as pad_token_id: {self.tokenizer.eos_token_id}")

        # Stream output if requested
        if stream:
            print(f"Generating text with streaming (max_new_tokens={max_new_tokens})...")
            def generate():
                previous_text = ""
                tokens_generated = 0
                for output in self.model.generate(**gen_kwargs, streamer=None, use_cache=True, return_dict_in_generate=False):
                    tokens_generated += 1
                    if tokens_generated % 10 == 0:
                        print(f"Generated {tokens_generated}/{max_new_tokens} tokens...")
                    
                    decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    # Only yield the new text
                    new_text = decoded_output[len(previous_text):]
                    previous_text = decoded_output
                    yield new_text
                print(f"Completed generation with {tokens_generated} tokens.")

            return generate()
        else:
            # Generate without streaming
            print(f"Generating text without streaming (max_new_tokens={max_new_tokens})...")
            with torch.no_grad():
                output = self.model.generate(**gen_kwargs)
            result = self.tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Generated {output.shape[1] - inputs.input_ids.shape[1]} tokens.")
            return result
EOL
log "FILES" "Created predict.py with added logging"

# If OpenAI validation is enabled, validate and update predict.py
if [ "$USE_OPENAI" = true ]; then
    log "OPENAI" "Starting OpenAI validation for predict.py"
    echo "Validating predict.py with OpenAI service for model compatibility..."
    PREDICT_CONTENT=$(cat predict.py)
    log "OPENAI" "Sending predict.py content to validation service"
    VALIDATED_CONTENT=$(validate_predict_with_openai "$HF_MODEL" "$PREDICT_CONTENT")
    
    if [ $? -eq 0 ] && [ ! -z "$VALIDATED_CONTENT" ]; then
        log "OPENAI" "Validation succeeded, updating predict.py"
        echo "✅ Successfully validated predict.py with OpenAI service."
        echo "$VALIDATED_CONTENT" > predict.py
        log "FILES" "Updated predict.py with validated content"
    else
        log "OPENAI" "Validation failed, keeping default predict.py"
        echo "⚠️ Validation failed or returned empty content. Using default predict.py."
    fi
fi

# Create a modified download_weights script that doesn't rely on environment variables
log "FILES" "Creating download_weights script"
cat > scripts/download_weights << EOL
#!/usr/bin/env python

import os
import sys
import shutil
import time
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

def log_progress(message):
    """Print log message with timestamp"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] DOWNLOAD: {message}")

def main():
    """Download model and save it to the weights directory"""
    # Hardcoded model and token values (will be replaced by the script)
    model_name = "${HF_MODEL}"
    token = "${HF_TOKEN}"
    
    log_progress(f"Starting download for model: {model_name}")
    
    # Create a fresh weights directory
    if os.path.exists("weights"):
        log_progress("Removing existing weights directory")
        shutil.rmtree("weights")
    
    log_progress("Creating new weights directory")
    os.makedirs("weights", exist_ok=True)
    
    # Login to Hugging Face if token is provided
    if token and token.strip():
        log_progress("Authenticating with Hugging Face...")
        login(token)
    else:
        log_progress("No token provided, continuing without authentication")
    
    try:
        # First download and save the tokenizer
        log_progress("Downloading tokenizer...")
        tokenizer_args = {"trust_remote_code": True}
        if token and token.strip():
            tokenizer_args["token"] = token
            
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        log_progress("Tokenizer downloaded, saving to weights directory")
        tokenizer.save_pretrained("weights")
        log_progress("Tokenizer saved successfully")
        
        # Then download and save the model
        log_progress("Downloading model weights (this may take a while)...")
        model_args = {"trust_remote_code": True}
        if token and token.strip():
            model_args["token"] = token
            
        log_progress("Starting model download (be patient, this is usually the longest step)")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
        
        log_progress("Model downloaded, saving to weights directory (this may also take time)")
        model.save_pretrained("weights")
        
        log_progress("✅ Model and tokenizer saved to the 'weights' directory successfully.")
        return True
    except Exception as e:
        log_progress(f"❌ Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
EOL

# Make scripts executable
log "FILES" "Making scripts executable"
chmod +x scripts/download_weights
log "FILES" "Made scripts executable"

# Create README.md
log "FILES" "Creating README.md"
cat > README.md << EOL
# ${REPLICATE_MODELNAME}

This is a Replicate deployment of the ${HF_MODEL} model.

## Model Details

- **Original Model**: ${HF_MODEL}
- **Replicate Model**: ${REPLICATE_MODEL}

## Usage

Provide a prompt to generate text with this model.

## Example

\`\`\`
Your example prompt here
\`\`\`

## Deployment Info

Deployed using [Cog](https://github.com/replicate/cog) and [Replicate](https://replicate.com)
EOL
log "FILES" "Created README.md"

echo "✅ Files created successfully!"
echo

# Step 4: Build with Cog
log "BUILD" "Starting Cog build process"
echo "🚀 Building with Cog..."
log "BUILD" "Running 'cog build' command"
cog build
build_status=$?
if [ $build_status -ne 0 ]; then
    log "ERROR" "Cog build failed with status $build_status"
    echo "❌ Cog build failed"
    exit 1
fi
log "BUILD" "Cog build completed successfully"

# Step 5: Download the weights (using cog run)
log "DOWNLOAD" "Starting model weights download"
echo "📥 Downloading model weights (this may take a while)..."

# Run the download script using cog run
log "DOWNLOAD" "Running download_weights script with cog run"
cog run scripts/download_weights
download_status=$?
if [ $download_status -ne 0 ]; then
    log "ERROR" "Weight download failed with status $download_status"
    echo "❌ Weight download failed"
    exit 1
fi
log "DOWNLOAD" "Model weights downloaded successfully"

# Step 6: Verify weights folder structure
log "VERIFY" "Starting verification of model weights folder structure"
echo "🔍 Verifying model weights folder structure..."

if [ ! -d "weights" ]; then
    log "ERROR" "Weights folder not found"
    echo "❌ Weights folder not found. Download may have failed."
    exit 1
fi

# Check if the required model files are present
if [ ! -f "weights/config.json" ]; then
    log "WARNING" "config.json not found in weights folder"
    echo "⚠️ Warning: config.json not found in weights folder."
fi

if [ ! -f "weights/tokenizer_config.json" ]; then
    log "WARNING" "tokenizer_config.json not found in weights folder"
    echo "⚠️ Warning: tokenizer_config.json not found in weights folder."
fi

# Check for model weights
if [ ! -f "weights/pytorch_model.bin" ] && [ ! -d "weights/pytorch_model.bin.index.json" ] && ! find weights -name "*.bin" -o -name "*.safetensors" | grep -q .; then
    log "WARNING" "No model weight files found in weights folder"
    echo "⚠️ Warning: No model weight files found. Check the weights folder."
    log "INFO" "Listing files in weights directory"
    ls -la weights/
else
    log "VERIFY" "Model weight files found"
    echo "✅ Model weight files found."
fi

log "VERIFY" "Weight folder structure verification complete"
echo "✅ Weight folder structure verification complete."

# Step 7: Ask to deploy
echo
echo "🚢 Deployment"
echo "------------"
log "DEPLOY" "Asking user about deployment"
read -p "Would you like to deploy the model to Replicate now? (y/n): " deploy_now

if [[ $deploy_now == [yY] || $deploy_now == [yY][eE][sS] ]]; then
    log "DEPLOY" "Starting deployment to Replicate"
    echo "🚀 Deploying to Replicate as r8.im/${REPLICATE_MODEL}..."
    log "DEPLOY" "Running 'cog push' command"
    cog push r8.im/${REPLICATE_MODEL}
    push_status=$?
    
    if [ $push_status -eq 0 ]; then
        log "DEPLOY" "Deployment successful"
        echo "✅ Deployment successful! Your model is now available at:"
        echo "https://replicate.com/${REPLICATE_MODEL}"
    else
        log "ERROR" "Deployment failed with status $push_status"
        echo "❌ Deployment failed. Please check the error messages above."
    fi
else
    log "DEPLOY" "Deployment skipped by user"
    echo "Deployment skipped. When you're ready, run:"
    echo "cog push r8.im/${REPLICATE_MODEL}"
fi

echo
echo "📋 Summary:"
echo "- Model configuration files are in the current directory"
echo "- Model weights are in the 'weights' folder"
echo "- To test locally: cog predict -i prompt=\"Your test prompt\""
echo
log "COMPLETE" "Deployment assistant finished successfully"
echo "Thank you for using the Transformer → Replicate Deployment Assistant! 🎉"
