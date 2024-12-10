import subprocess

def run(command):
    # Run the command
    result = subprocess.run(command)

    # Print the output and errors
    print("Output:")
    print(result.stdout)
    print("Errors:")
    print(result.stderr)

    return result

x = run(["ls"])
print(x.stderr)
