import subprocess
script = r'''
tell application "Google Chrome" to activate
delay 1
tell application "System Events"
    tell process "Google Chrome"
        set frontmost to true
        try
            set v to value of attribute "AXDocument" of group 1 of group 1 of tab group 1 of splitter group 1 of window 1
            return v
        on error errMsg
            return "ERR " & errMsg
        end try
    end tell
end tell
'''
res=subprocess.run(['osascript','-e',script], text=True, capture_output=True)
print(res.stdout)
print(res.stderr)
print('exit', res.returncode)
