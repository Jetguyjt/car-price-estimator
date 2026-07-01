import subprocess, time
url = 'https://www.cars.com/shopping/results/?zip=98103' + '&maximum_distance=9999&sort=listed_at_desc&stock_type=used&list_price_min=0&list_price_max=5000'
script = f'''
tell application "Safari"
    if (count of windows) = 0 then make new document
    set URL of current tab of front window to "{url}"
end tell
'''
subprocess.run(['osascript','-e',script], check=False)
time.sleep(8)
script2 = r'''
tell application "Safari"
    tell current tab of front window
        set t to name
        set u to URL
        try
            set bodyText to do JavaScript "document.body.innerText.slice(0,1000)"
        on error errMsg
            set bodyText to "JSERR: " & errMsg
        end try
    end tell
end tell
return "TITLE=" & t & linefeed & "URL=" & u & linefeed & bodyText
'''
res = subprocess.run(['osascript','-e',script2], text=True, capture_output=True)
print(res.stdout)
print(res.stderr)
print('exit', res.returncode)
