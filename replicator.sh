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

# Function to load configuration
load_config() {
    log "CONFIG" "Loading configuration..."
    
    # Check if config file exists
    if [ -f "config.json" ]; then
        # Check if jq is installed
        if ! command -v jq &> /dev/null; then
            log "CONFIG" "Warning: jq not installed, cannot parse config file"
            echo "⚠️ Warning: jq is not installed. Cannot load saved configuration."
            return 1
        fi
        
        # Load values from config.json
        REPLICATE_USERNAME=$(jq -r '.replicate_username // empty' config.json)
        REPLICATE_TOKEN=$(jq -r '.replicate_token // empty' config.json)
        HF_TOKEN=$(jq -r '.hf_token // empty' config.json)
        VALIDATION_ENDPOINT=$(jq -r '.validation_endpoint // empty' config.json)
        
        # Set default validation endpoint if not defined
        if [ -z "$VALIDATION_ENDPOINT" ]; then
            VALIDATION_ENDPOINT="https://validate.fortunestold.co/hf"
            log "CONFIG" "Using default validation endpoint: $VALIDATION_ENDPOINT"
        else
            log "CONFIG" "Loaded validation endpoint: $VALIDATION_ENDPOINT"
        fi
        
        if [ -n "$REPLICATE_USERNAME" ]; then
            log "CONFIG" "Loaded Replicate username: $REPLICATE_USERNAME"
            echo "✅ Loaded Replicate username: $REPLICATE_USERNAME"
        else
            log "CONFIG" "No Replicate username found in config"
        fi
        
        if [ -n "$REPLICATE_TOKEN" ]; then
            log "CONFIG" "Loaded Replicate token from config"
            echo "✅ Loaded Replicate token from config"
        else
            log "CONFIG" "No Replicate token found in config"
        fi
        
        if [ -n "$HF_TOKEN" ]; then
            log "CONFIG" "Loaded Hugging Face token from config"
            echo "✅ Loaded Hugging Face token from config"
        else
            log "CONFIG" "No Hugging Face token found in config"
        fi
        
        return 0
    else
        log "CONFIG" "No config file found"
        # Set default validation endpoint
        VALIDATION_ENDPOINT="https://validate.fortunestold.co/hf"
        log "CONFIG" "Using default validation endpoint: $VALIDATION_ENDPOINT"
        echo "ℹ️ No saved configuration found. You'll be prompted for credentials."
        return 1
    fi
}

# Function to save configuration
save_config() {
    log "CONFIG" "Saving configuration..."
    
    # Check if jq is installed
    if ! command -v jq &> /dev/null; then
        log "CONFIG" "Warning: jq not installed, cannot save config file"
        echo "⚠️ Warning: jq is not installed. Cannot save configuration."
        return 1
    fi
    
    # Create or update config.json
    local config_data="{}"
    
    if [ -f "config.json" ]; then
        config_data=$(cat config.json)
    fi
    
    if [ -n "$REPLICATE_USERNAME" ]; then
        config_data=$(echo "$config_data" | jq --arg username "$REPLICATE_USERNAME" '.replicate_username = $username')
    fi
    
    if [ -n "$REPLICATE_TOKEN" ]; then
        config_data=$(echo "$config_data" | jq --arg token "$REPLICATE_TOKEN" '.replicate_token = $token')
    fi
    
    if [ -n "$HF_TOKEN" ]; then
        config_data=$(echo "$config_data" | jq --arg token "$HF_TOKEN" '.hf_token = $token')
    fi
    
    if [ -n "$VALIDATION_ENDPOINT" ]; then
        config_data=$(echo "$config_data" | jq --arg endpoint "$VALIDATION_ENDPOINT" '.validation_endpoint = $endpoint')
    fi
    
    echo "$config_data" > config.json
    log "CONFIG" "Configuration saved to config.json"
    echo "✅ Configuration saved to config.json"
    
    return 0
}

# Function to edit credentials
edit_credentials() {
    log "CONFIG" "Editing credentials..."
    echo "📝 Edit Credentials"
    echo "-----------------"
    
    # Show current Replicate username if available
    if [ -n "$REPLICATE_USERNAME" ]; then
        echo "Current Replicate username: $REPLICATE_USERNAME"
    else
        echo "No Replicate username currently set"
    fi
    
    # Prompt for new Replicate username
    read -p "Enter Replicate username (leave empty to keep current): " new_username
    
    if [ -n "$new_username" ]; then
        REPLICATE_USERNAME="$new_username"
        log "CONFIG" "Updated Replicate username to: $REPLICATE_USERNAME"
        echo "✅ Updated Replicate username to: $REPLICATE_USERNAME"
    fi
    
    # Show current Replicate token status
    if [ -n "$REPLICATE_TOKEN" ]; then
        echo "Current Replicate token: [HIDDEN]"
    else
        echo "No Replicate token currently set"
    fi
    
    # Prompt for new Replicate token
    read -p "Enter Replicate token (leave empty to keep current): " new_replicate_token
    
    if [ -n "$new_replicate_token" ]; then
        REPLICATE_TOKEN="$new_replicate_token"
        log "CONFIG" "Updated Replicate token"
        echo "✅ Updated Replicate token"
    fi
    
    # Show current HF token status
    if [ -n "$HF_TOKEN" ]; then
        echo "Current Hugging Face token: [HIDDEN]"
    else
        echo "No Hugging Face token currently set"
    fi
    
    # Prompt for new HF token
    read -p "Enter Hugging Face token (leave empty to keep current): " new_token
    
    if [ -n "$new_token" ]; then
        HF_TOKEN="$new_token"
        log "CONFIG" "Updated Hugging Face token"
        echo "✅ Updated Hugging Face token"
    fi
    
    # Show current validation endpoint
    echo "Current validation endpoint: $VALIDATION_ENDPOINT"
    
    # Prompt for new validation endpoint
    read -p "Enter validation endpoint (leave empty to keep current): " new_endpoint
    
    if [ -n "$new_endpoint" ]; then
        VALIDATION_ENDPOINT="$new_endpoint"
        log "CONFIG" "Updated validation endpoint to: $VALIDATION_ENDPOINT"
        echo "✅ Updated validation endpoint to: $VALIDATION_ENDPOINT"
    fi
    
    # Save the updated configuration
    save_config
    
    echo "Press Enter to return to main menu..."
    read
}

# Function to check and install dependencies
check_install_dependencies() {
    log "DEPS" "Checking for dependencies..."
    echo "🔍 Checking for required dependencies..."
    
    local missing_deps=()
    
    # Check for Docker
    if ! command -v docker &> /dev/null; then
        log "DEPS" "Docker is not installed"
        echo "❌ Docker is not installed"
        missing_deps+=("docker")
    else
        log "DEPS" "Docker is installed: $(docker --version)"
        echo "✅ Docker is installed: $(docker --version)"
    fi
    
    # Check for Cog
    if ! command -v cog &> /dev/null; then
        log "DEPS" "Cog is not installed"
        echo "❌ Cog is not installed"
        missing_deps+=("cog")
    else
        log "DEPS" "Cog is installed: $(cog --version)"
        echo "✅ Cog is installed: $(cog --version)"
    fi
    
    # If all dependencies are installed, return
    if [ ${#missing_deps[@]} -eq 0 ]; then
        log "DEPS" "All dependencies are installed"
        echo "✅ All required dependencies are installed."
        echo "Press Enter to return to main menu..."
        read
        return 0
    fi
    
    # Ask if user wants to install missing dependencies
    echo
    echo "The following dependencies are missing:"
    for dep in "${missing_deps[@]}"; do
        echo "- $dep"
    done
    echo
    
    read -p "Would you like to install the missing dependencies? (y/n): " install_deps
    
    if [[ $install_deps != [yY] && $install_deps != [yY][eE][sS] ]]; then
        log "DEPS" "User declined to install dependencies"
        echo "Installation skipped. Press Enter to return to main menu..."
        read
        return 1
    fi
    
    log "DEPS" "User agreed to install dependencies"
    echo "Downloading the setup scripts..."
    echo "If the installation fails, quit this helper and re-run with sudo."
    echo "Press Q to cancel (waiting 10 seconds)..."
    
    # Wait for 10 seconds, allowing user to cancel
    local start_time=$(date +%s)
    local elapsed=0
    local key=""
    
    # Enable non-blocking read
    if [ -t 0 ]; then  # Check if stdin is a terminal
        stty -echo -icanon time 0 min 0
    fi
    
    while [ $elapsed -lt 10 ]; do
        # Calculate remaining time
        elapsed=$(($(date +%s) - start_time))
        remaining=$((10 - elapsed))
        
        # Print countdown (overwrite the same line)
        echo -ne "\rContinuing in $remaining seconds... (Press Q to cancel) "
        
        # Check for keypress
        if [ -t 0 ]; then  # Only if stdin is a terminal
            key=$(dd bs=1 count=1 2>/dev/null)
            if [[ ${key,,} == "q" ]]; then
                # Restore terminal settings
                stty sane
                echo -e "\nCancelled by user."
                echo "Press Enter to return to main menu..."
                read
                return 1
            fi
        else
            # If not in a terminal, just sleep
            sleep 1
        fi
        
        # Short delay
        sleep 0.1
    done
    
    # Restore terminal settings
    if [ -t 0 ]; then
        stty sane
    fi
    
    echo -e "\nProceeding with installation..."
    
    # Install Docker if needed
    if [[ " ${missing_deps[*]} " =~ " docker " ]]; then
        log "DEPS" "Installing Docker..."
        echo "🔄 Installing Docker..."
        curl -fsSL https://get.docker.com | sh
        
        if [ $? -eq 0 ]; then
            log "DEPS" "Docker installed successfully"
            echo "✅ Docker installed successfully"
        else
            log "DEPS" "Docker installation failed"
            echo "❌ Docker installation failed. You may need to run with sudo."
        fi
    fi
    
    # Install Cog if needed
    if [[ " ${missing_deps[*]} " =~ " cog " ]]; then
        log "DEPS" "Installing Cog..."
        echo "🔄 Installing Cog..."
        sh <(curl -fsSL https://cog.run/install.sh)
        
        if [ $? -eq 0 ]; then
            log "DEPS" "Cog installed successfully"
            echo "✅ Cog installed successfully"
        else
            log "DEPS" "Cog installation failed"
            echo "❌ Cog installation failed. You may need to run with sudo."
        fi
    fi
    
    echo "Dependency installation completed."
    echo "Press Enter to return to main menu..."
    read
    return 0
}

# Function to fetch the predict.py file from the validation service
fetch_predict_py() {
    local model="$1"
    
    log "FETCH" "Fetching predict.py for model: $model"
    echo "🧠 Fetching optimized predict.py for model: $model"
    echo "Using validation endpoint: $VALIDATION_ENDPOINT"
    
    # Make the GET request and capture the response
    log "FETCH" "Making GET request to validation service..."
    response=$(curl -s -G "$VALIDATION_ENDPOINT" \
                  --data-urlencode "model=$model")
    
    # Check if curl was successful and if we got a response
    if [ $? -eq 0 ] && [ -n "$response" ]; then
        # Save the response to predict.py
        echo "$response" > predict.py
        
        # Check if the file was successfully saved and has content
        if [ -s "predict.py" ]; then
            log "FETCH" "Successfully downloaded predict.py"
            echo "✅ Successfully generated predict.py for $model"
            return 0
        else
            log "ERROR" "Generated predict.py is empty"
            echo "❌ Error: Generated predict.py file is empty."
            return 1
        fi
    else
        log "ERROR" "Failed to generate predict.py"
        echo "❌ Error: Failed to generate predict.py for $model"
        return 1
    fi
}

# Function to deploy a new model
deploy_new_model() {
    log "DEPLOY" "Starting new model deployment workflow"
    
    # Step 1: Get HuggingFace model name
    log "INPUT" "Prompting for Hugging Face model name"
    read -p "Enter the Hugging Face model name (e.g. mosaicml/mpt-7b-storywriter): " HF_MODEL
    if [ -z "$HF_MODEL" ]; then
        log "ERROR" "Model name cannot be empty"
        echo "Error: Model name cannot be empty."
        return 1
    fi
    log "INPUT" "Received Hugging Face model: $HF_MODEL"

    # Step 2: Get Replicate model info
    # If we already have a username, use it as default
    if [ -n "$REPLICATE_USERNAME" ]; then
        log "INPUT" "Using saved Replicate username: $REPLICATE_USERNAME"
        read -p "Enter the Replicate model name (will be deployed as $REPLICATE_USERNAME/modelname): " REPLICATE_MODELNAME
        
        if [ -z "$REPLICATE_MODELNAME" ]; then
            log "ERROR" "Replicate model name cannot be empty"
            echo "Error: Replicate model name cannot be empty."
            return 1
        fi
        
        REPLICATE_MODEL="$REPLICATE_USERNAME/$REPLICATE_MODELNAME"
        log "INPUT" "Set Replicate model to: $REPLICATE_MODEL"
    else
        # No saved username, ask for full path
        log "INPUT" "Prompting for Replicate destination"
        while true; do
            read -p "Enter the Replicate destination in format 'username/modelname': " REPLICATE_MODEL
            if [ -z "$REPLICATE_MODEL" ]; then
                log "ERROR" "Replicate model name cannot be empty"
                echo "Error: Replicate model name cannot be empty."
            elif validate_replicate_format "$REPLICATE_MODEL"; then
                log "INPUT" "Replicate format validated successfully: $REPLICATE_MODEL"
                
                # Extract and save the username for future use
                REPLICATE_USERNAME=$(echo $REPLICATE_MODEL | cut -d'/' -f1)
                REPLICATE_MODELNAME=$(echo $REPLICATE_MODEL | cut -d'/' -f2)
                log "CONFIG" "Saved Replicate username: $REPLICATE_USERNAME"
                
                # Save the configuration with the new username
                save_config
                
                break
            else
                log "ERROR" "Invalid Replicate format: $REPLICATE_MODEL"
                echo "Error: Format must be 'username/modelname' (e.g. fortunestold/storybook)"
                echo "Note: Enter the username, followed by a slash, followed by the model name."
            fi
        done
    fi

    # Step 3: Check for HuggingFace token, ask if not available
    if [ -z "$HF_TOKEN" ]; then
        log "INPUT" "Prompting for Hugging Face token"
        read -p "Enter your Hugging Face access token (or press Enter to continue without authentication): " HF_TOKEN
        
        if [ -n "$HF_TOKEN" ]; then
            log "CONFIG" "Hugging Face token provided, saving to config"
            save_config
        else
            log "CONFIG" "No Hugging Face token provided, continuing without authentication"
        fi
    else
        log "CONFIG" "Using saved Hugging Face token"
    fi

    echo
    echo "📋 Configuration Summary:"
    echo "-------------------------"
    echo "Hugging Face model: $HF_MODEL"
    echo "Replicate deployment: $REPLICATE_MODEL"
    echo "Validation endpoint: $VALIDATION_ENDPOINT"
    if [ -n "$HF_TOKEN" ]; then
        echo "Using Hugging Face authentication: Yes"
    else
        echo "Using Hugging Face authentication: No"
    fi
    if [ -n "$REPLICATE_TOKEN" ]; then
        echo "Using Replicate authentication: Yes"
    else
        echo "Using Replicate authentication: No"
    fi
    echo "-------------------------"
    echo

    # Ask for confirmation
    log "INPUT" "Asking for confirmation"
    read -p "Is this information correct? (y/n): " confirm
    if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
        log "CANCEL" "User cancelled operation"
        echo "Operation cancelled."
        return 1
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

    # Fetch predict.py from the validation service
    log "FILES" "Fetching predict.py from validation service"
    if ! fetch_predict_py "$HF_MODEL"; then
        log "WARNING" "Failed to fetch predict.py, creating a default version"
        echo "⚠️ Failed to fetch predict.py, creating a default version instead."
        
        # Create a default predict.py file
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
        log "FILES" "Created default predict.py"
    else
        log "FILES" "Successfully fetched predict.py from validation service"
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
        return 1
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
        return 1
    fi
    log "DOWNLOAD" "Model weights downloaded successfully"

    # Step 6: Verify weights folder structure
    log "VERIFY" "Starting verification of model weights folder structure"
    echo "🔍 Verifying model weights folder structure..."

    if [ ! -d "weights" ]; then
        log "ERROR" "Weights folder not found"
        echo "❌ Weights folder not found. Download may have failed."
        return 1
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
        
        # Use Replicate token if available
        if [ -n "$REPLICATE_TOKEN" ]; then
            log "DEPLOY" "Using saved Replicate token"
            export REPLICATE_API_TOKEN="$REPLICATE_TOKEN"
        fi
        
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
        if [ -n "$REPLICATE_TOKEN" ]; then
            echo "REPLICATE_API_TOKEN=\"your_token\" cog push r8.im/${REPLICATE_MODEL}"
        else
            echo "cog push r8.im/${REPLICATE_MODEL}"
        fi
    fi

    echo
    echo "📋 Summary:"
    echo "- Model configuration files are in the current directory"
    echo "- Model weights are in the 'weights' folder"
    echo "- To test locally: cog predict -i prompt=\"Your test prompt\""
    echo
    log "COMPLETE" "Deployment assistant finished successfully"
    echo "Thank you for using the Transformer → Replicate Deployment Assistant! 🎉"
    
    return 0
}

# Main script entry point

# Banner
echo "================================================"
echo "🤗  Transformer → Replicate Deployment Assistant"
echo "================================================"

log "INIT" "Starting deployment assistant"

# Load the configuration if it exists
load_config

# Main menu
while true; do
    clear
    echo "🤗 Transformer → Replicate Deployment Assistant"
    echo "================================================"
    echo
    echo "Current Configuration:"
    if [ -n "$REPLICATE_USERNAME" ]; then
        echo "- Replicate Username: $REPLICATE_USERNAME"
    else
        echo "- Replicate Username: Not set"
    fi
    
    if [ -n "$REPLICATE_TOKEN" ]; then
        echo "- Replicate Token: [SAVED]"
    else
        echo "- Replicate Token: Not set"
    fi
    
    if [ -n "$HF_TOKEN" ]; then
        echo "- Hugging Face Token: [SAVED]"
    else
        echo "- Hugging Face Token: Not set"
    fi
    
    echo "- Validation Endpoint: $VALIDATION_ENDPOINT"
    echo
    echo "Main Menu:"
    echo "  N - Deploy a New Model"
    echo "  D - Install Dependencies"
    echo "  X - Edit Credentials"
    echo "  Q - Quit"
    echo
    read -p "Select an option: " choice
    
    case ${choice,,} in
        n)
            deploy_new_model
            echo "Press Enter to return to the main menu..."
            read
            ;;
        d)
            check_install_dependencies
            ;;
        x)
            edit_credentials
            ;;
        q)
            log "EXIT" "User quit the application"
            echo "Exiting. Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid selection. Press Enter to try again..."
            read
            ;;
    esac
done
