#!/usr/bin/env bash

# Checks to see if variable is set and non-empty.
env_var_exists() {
  if [[ -n "${!1}" ]]; then
    return 0
  else
    return 1
  fi
}

# Define the directory path for WSL2 - 这部分将被删除，因为我们不在WSL2中
# lib_path="/usr/lib/wsl/lib/"

# If it is run with the sudo command, get the complete LD_LIBRARY_PATH environment variable of the system and assign it to the current environment.
if [ -n "$SUDO_USER" ] || [ -n "$SUDO_COMMAND" ]; then
    echo "The sudo command resets the non-essential environment variables, we keep the LD_LIBRARY_PATH variable."
    export LD_LIBRARY_PATH=$(sudo -i printenv LD_LIBRARY_PATH)
fi

# This gets the directory the script is run from so pathing can work relative to the script where needed.
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)

# Step into GUI local directory
cd "$SCRIPT_DIR" || exit 1

# Activate conda environment - 替换为您的conda环境名称
conda_env_name="kohya"
source activate "$conda_env_name"

# Check if LD_LIBRARY_PATH environment variable exists
if [[ -z "${LD_LIBRARY_PATH}" ]]; then
    # Set the ANSI escape sequence for yellow text
    YELLOW='\033[0;33m'
    # Set the ANSI escape sequence to reset text color
    RESET='\033[0m'
    
    echo -e "${YELLOW}Warning: LD_LIBRARY_PATH environment variable is not set.${RESET}"
fi

# Determine the requirements file based on the system
# 此部分保持不变
# ... [省略的部分] ...

#Set STARTUP_CMD as conda python if not specified
if [[ -z "$STARTUP_CMD" ]]
then
    STARTUP_CMD=python  # 这里假定conda环境已经设置好了Python路径
fi

# Validate the requirements and run the script if successful
if python "$SCRIPT_DIR/setup/validate_requirements.py" -r "$REQUIREMENTS_FILE"; then
    "${STARTUP_CMD}" $STARTUP_CMD_ARGS "$SCRIPT_DIR/kohya_gui.py" "$@"
else
    echo "Validation failed. Exiting..."
    exit 1
fi
