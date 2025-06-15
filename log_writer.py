def write_log(msg: str, dir: str):
    
    if not msg.startswith("\n"):
        msg = "\n" + msg

    with open(dir, "a") as file:
        file.write(msg)