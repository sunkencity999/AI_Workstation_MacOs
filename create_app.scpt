tell application "Finder"
    set appPath to ((path to desktop) as text) & "Shell GPT Interface.app"
    
    try
        -- Create a new AppleScript application
        set newApp to make new application file at folder "local_shellgpt_llm" of (path to documents folder) with properties {name:"Shell GPT Interface", visible:true}
        
        -- Get the path to the script as text
        set appPath to (POSIX path of (newApp as alias))
        
        -- Create the launcher script content
        set scriptContent to "#!/bin/bash

cd \"" & (POSIX path of (path to documents folder)) & "mac_computer_use/local_shellgpt_llm\"
source venv/bin/activate

# Ensure Ollama is running
if ! pgrep -x \"ollama\" > /dev/null; then
    ollama serve &
    sleep 3
fi

# Launch streamlit
streamlit run sgpt_interface.py --server.port 8527 --server.headless true
"
        
        -- Write the script to a temporary file
        do shell script "echo '" & scriptContent & "' > /tmp/launcher.sh"
        do shell script "chmod +x /tmp/launcher.sh"
        
        -- Create the AppleScript runner
        set scriptPath to appPath & "/Contents/Resources/Scripts/main.scpt"
        set scriptContent to "tell application \"Terminal\"
    activate
    do script \"/tmp/launcher.sh\"
end tell"
        
        do shell script "mkdir -p " & quoted form of (appPath & "/Contents/Resources/Scripts")
        do shell script "echo '" & scriptContent & "' > " & quoted form of scriptPath
        
        -- Set execute permissions
        do shell script "chmod +x " & quoted form of appPath
        
        display dialog "Application created successfully at " & appPath
    on error errMsg number errNum
        display dialog "Error creating application: " & errMsg & " (Error " & errNum & ")"
    end try
end tell