import paramiko
import sys

def verify_and_deploy():
    hostname = "209.38.249.154"
    username = "root"
    password = r"06050022003dD@d"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        client.connect(hostname=hostname, username=username, password=password, timeout=15)
        
        print("1. Pulling latest code...")
        client.exec_command("cd /root/DZemo && git pull origin main")
        
        print("2. Running standalone verification of the 'Better Way' scraper...")
        # We'll create a small one-liner to call the function
        # We need to set up the event loop too since it's async
        test_cmd = """
import asyncio
from app import scrape_facebook_comments_better_way
import json

async def run_test():
    url = 'https://www.facebook.com/ramy.jus/posts/pfbid0LrGiVSKCSKfSv2QDUWePv5D1qTKVuGZW4KUsjUHe85JSGpSexaS1tfRhBiVh4Nwyl'
    comments = await scrape_facebook_comments_better_way(url)
    print(f'VERIFICATION_COUNT:{len(comments)}')

if __name__ == "__main__":
    asyncio.run(run_test())
"""
        # Save it to a file on server first to avoid shell quoting issues
        sftp = client.open_sftp()
        with sftp.file('/root/DZemo/verify_script.py', 'w') as f:
            f.write(test_cmd)
        sftp.close()
        
        # Run it
        stdin, stdout, stderr = client.exec_command("/root/DZemo/venv/bin/python3 /root/DZemo/verify_script.py")
        out = stdout.read().decode('utf-8', errors='replace')
        err = stderr.read().decode('utf-8', errors='replace')
        print(f"STDOUT: {out}")
        print(f"STDERR: {err}")
        
        if "VERIFICATION_COUNT" in out:
            count = int(out.split("VERIFICATION_COUNT:")[1].strip())
            print(f"Successfully scraped {count} comments in verification test.")
            if count >= 30: # 30 is much better than the previous 4-12
                print("Count meets threshold. Restarting service...")
                client.exec_command("systemctl restart dzemotion.service")
                print("Service restarted.")
            else:
                print("Count still too low. Investigating...")
        else:
            print("Verification failed or timed out.")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    verify_and_deploy()
