import os

def write_log(msg: str, dir: str):

    os.makedirs(os.path.dirname(dir), exist_ok=True)
    
    if not msg.startswith("\n"):
        msg = "\n" + msg

    with open(dir, "a") as file:
        file.write(msg)